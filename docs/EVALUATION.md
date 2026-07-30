# Retrieval Evaluation

Measured on 2026-07-29/30. Everything below is reproducible from the commands in
[Reproducing](#reproducing); raw runs are committed under `backend/eval/results/`.

## Headline

Adding a BM25 sparse leg to the dense-only retriever is a large, statistically significant
win. Cross-encoder reranking is not.

| Configuration | hit@1 | hit@3 | hit@10 | MRR@10 | nDCG@10 | retrieval p50 | p95 |
|---|---|---|---|---|---|---|---|
| `dense_legacy` — **baseline, as shipped** | 0.76 | 0.84 | 0.90 | 0.8123 | 0.8339 | 13 ms | 35 ms |
| `sparse` — BM25 only | **0.96** | 0.98 | **1.00** | **0.9740** | **0.9804** | **3 ms** | **7 ms** |
| `hybrid` — dense + BM25, RRF fused | 0.92 | 0.98 | **1.00** | 0.9529 | 0.9645 | 14 ms | 36 ms |
| `hybrid` + rerank top-30 | 0.94 | 0.98 | 0.98 | 0.9600 | 0.9652 | 231 ms | 432 ms |
| `hybrid` + rerank top-50 | 0.94 | 0.98 | 0.98 | 0.9600 | 0.9652 | 361 ms | 587 ms |

Paired bootstrap over the 50 queries (10 000 resamples) plus a sign test, on nDCG@10:

| Comparison | Δ nDCG@10 | 95% CI | queries better / worse | p | Verdict |
|---|---|---|---|---|---|
| baseline → `hybrid` | **+0.1306** | [+0.059, +0.212] | 11 / 0 | 0.0010 | **significant** |
| baseline → `sparse` | **+0.1464** | [+0.067, +0.237] | 12 / 1 | 0.0034 | **significant** |
| `hybrid` → `sparse` | +0.0158 | [−0.013, +0.045] | 4 / 1 | 0.3750 | not significant |
| `hybrid` → `hybrid`+rerank | +0.0007 | [−0.055, +0.044] | 4 / 2 | 0.6875 | not significant |

**Shipped configuration:** `RETRIEVAL_MODE=hybrid`, `ENABLE_RERANK=false`.

`sparse` alone scored highest on every metric, but its edge over `hybrid` is not significant
(4 queries better, 1 worse). Hybrid is shipped anyway, because the gold set under-represents
exactly the query type the dense leg exists to handle — see
[what this does not measure](#what-this-does-not-measure). Choosing sparse-only on a
non-significant 4-query margin would be fitting the retriever to the eval.

## Findings

### 1. The dense-only baseline was the weakest configuration tested

BM25 with IDF beat the 384-dim bi-encoder by +0.20 hit@1. The baseline failed 12 of 50 queries
at rank 1, two of them past rank 20 (`q004`, a cautionary-statement passage; `q025`, an exhibit
index). BM25 failed 2.

The corpus explains it: filings are dense with rare discriminative tokens — `Bencap`, `Xience`,
`Llano`, `8.125%`, `P-25`, `iStar`. IDF makes a single rare term decisive, while a 384-dim
bi-encoder compresses a sentence toward a semantic centroid and loses precisely those tokens.

### 2. `Modifier.IDF` is load-bearing, and verified end to end

fastembed emits only the term-frequency half of BM25. Inspected directly, document vectors
carry TF weights and query vectors carry all `1.0`:

```
doc   values[:8]  [1.6434199 1.6434199 1.6434199 ...]
query values      [1 1 1 1 1]
```

The IDF factor comes from Qdrant, and only if the sparse field declares it. Confirmed live on
the collection: `sparse: {"sparse": {"modifier": "idf"}}`. Without it the sparse leg degrades
to term counting with no rarity weighting.

### 3. Reranking changed nothing measurable, and the retrieval leg stopped mattering

Reranking moved nDCG@10 by +0.0007 (p=0.69) and *lowered* hit@10 from 1.00 to 0.98 by demoting
one relevant chunk out of the top 10 (`q031`). Depth 50 scored identically to depth 30 while
costing 325 ms instead of 197 ms, so the deeper setting is pure waste.

More telling: `dense`+rerank, `sparse`+rerank and `hybrid`+rerank all produced **identical**
metrics, agreeing on the rank-1 chunk for **50/50** queries. Every leg already places the answer
inside the top 30, so the cross-encoder — not the retriever — decides the final order. Its
ceiling here (0.9652) sits below plain `sparse` (0.9804).

A note on cost: 197 ms is *not* the reason to leave reranking off. Generation dominates
end-to-end latency at **6 518 ms p50 / 12 750 ms p95** (llama3.1:8b, 50 queries), so the
reranker is ~3% of a response. The reason to leave it off is that it buys no measurable quality.

An earlier hypothesis — that `q031` regressed because its long compensation paragraph was
truncated at the cross-encoder's 512-token limit — was tested and **rejected**: the pair
tokenises to 306 tokens, well inside the limit. The regression is genuine misjudgement on a
multi-topic passage, most likely MS MARCO domain mismatch.

### 4. RRF fusion was non-deterministic; fixed

Identical hybrid requests returned different orderings. RRF ties are routine — a chunk ranked
1st by dense and 2nd by sparse fuses to the same score as one ranked 2nd then 1st — and Qdrant
leaves tied points unordered. Measured across five identical eval runs:

| | hit@1 | MRR@10 | nDCG@10 |
|---|---|---|---|
| before | 0.90 – 0.94 (spread 0.040) | spread 0.020 | spread 0.015 |
| after | 0.92 (spread 0.000) | 0.000 | 0.000 |

Fixed in `HybridStore` by over-fetching `2 × limit`, sorting on `(-score, chunk_id)`, then
truncating — over-fetching matters because ties straddling the cut-off otherwise change *which*
points come back, not just their order. Dense-only and sparse-only were already deterministic.

This is also a correctness point for the product: without it, the same question returned
differently ordered citations on a refresh.

### 5. Two data bugs in the indexer

- **Section labels were falsified.** `section` is a `ClassLabel`, so the raw value is an integer
  index. `str(row.get("section") or "unknown")` mapped index `0` (`section_1`, Business) to
  `"unknown"` — 195 of 2000 chunks — and labelled the rest with indexes that look like Item
  numbers but are not: stored `"10"` is `section_8`, not Item 10.
- **`chunk_id` collided.** 2000 points shared 1993 ids, because the id derives from a chunk
  index that was always `0`, so repeated boilerplate sentences hashed identically.

### 6. The indexer never actually chunked

`pack_sentences([sentence], max_chars=1800)` receives a one-element list, so it is a no-op:
every "chunk" was a single sentence (median 147 characters). The dataset was also shuffled
before selection, which makes multi-sentence chunking impossible regardless of the budget —
consecutive sentences from a filing are scattered. Both are fixed in the rewritten indexer,
which groups by filing and packs by token count.

## Chunking A/B

10-K Item boundaries **are preserved** by the dataset, as structured metadata rather than
header text: `section` is a `ClassLabel` over `['section_1', 'section_1A', …, 'section_15']`
and `sentenceID` is `<docID>_section_<Item>_<index>`. So the Item arm splits on that label
instead of regex-matching `ITEM\s+1A\.?\s+RISK\s+FACTORS` out of prose — same intent, exact
boundaries, no parsing risk.

All arms use the same 60 filings (67 803 sentences) and the same retrieval strategy (`hybrid`),
so only chunking varies. Relevance is projected per-arm by sentence containment, which is
chunking-invariant; all 50 anchors matched in every arm.

| Arm | chunks | hit@1 | hit@3 | hit@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| one sentence per chunk (current behaviour) | 67 803 | 0.30 | 0.70 | 0.82 | 0.5107 | 0.5903 |
| fixed 256/32 | 10 489 | 0.44 | 0.60 | 0.86 | 0.5456 | **0.6075** |
| Item-bounded 256/32 | 11 028 | 0.44 | 0.64 | 0.84 | 0.5449 | 0.6068 |
| fixed 512/64 | 5 241 | 0.28 | 0.60 | 0.82 | 0.4632 | 0.5260 |
| Item-bounded 512/64 | 5 848 | 0.32 | 0.60 | 0.82 | 0.4882 | 0.5431 |

| Comparison | Δ nDCG@10 | 95% CI | better / worse | p | Verdict |
|---|---|---|---|---|---|
| fixed → Item-bounded (256/32) | −0.0007 | [−0.046, +0.046] | 6 / 8 | 0.79 | not significant |
| 512/64 → 256/32 (fixed) | +0.0815 | [−0.005, +0.166] | 22 / 12 | 0.12 | not significant |
| sentence → 256/32 (fixed) | +0.0172 | [−0.052, +0.093] | 13 / 16 | 0.71 | not significant |

**Conclusion: no chunking variant is significantly better than another at n=50.**

- **Item boundaries make no measurable difference.** The effect is not merely small, it changes
  sign with configuration: Item-bounded is ahead at 512/64 (+0.017), level at 256/32 (−0.001),
  and behind under sparse-only retrieval (−0.013). That is noise.
- **Chunk size is the only lever with a directional signal.** 512 → 256 tokens is +0.08 nDCG
  with 22 queries better and 12 worse, and the CI only just includes zero. It is not significant
  at n=50, but it has a mechanism: **all-MiniLM-L6-v2 truncates at 256 tokens**
  (`max_seq_length = 256`), so the back half of every 512-token chunk never reaches the dense
  vector and is reachable only through BM25. A 512-token budget on this encoder is a
  mis-specification regardless of what the metrics say.

The indexer therefore defaults its budget to the encoder's own `max_seq_length` rather than a
hardcoded number, and keeps Item-bounded splitting — not for retrieval quality, which is a wash,
but because a chunk that never straddles two Items has an unambiguous `section` for citation.

Absolute numbers in this table are much lower than the retrieval table and are **not
comparable** to it: this corpus is 60 complete filings (5k–68k chunks) rather than a 2000-chunk
sample of partial ones, so there are far more near-miss distractors.

## Gold set

`backend/eval/gold/gold.jsonl` — 50 queries, JSONL, multiple relevant chunks permitted.

Built in two stages, both committed so the process is auditable:

1. **Sample + draft** (`scripts/build_gold_set.py`) — 50 chunks sampled stratified by section,
   filtered to 260–1400 characters, excluding signature blocks and cross-reference boilerplate
   that no known-item query could distinguish. llama3.1:8b drafts one question per chunk.
2. **Manual rewrite** (`eval/gold/rewrites.tsv`) — all 50 rewritten by hand. Each record keeps
   `draft_query` next to the final `query`.

The rewrite was not cosmetic. Measured query→chunk content-word containment:

| | mean | median |
|---|---|---|
| Llama drafts | 0.650 | 0.667 |
| hand rewrites | 0.506 | 0.500 |

a 22% reduction in vocabulary copied from the source chunk. The rewrite also fixed drafts that
opened with an unresolvable pronoun (`"What was his role at Westinghouse…"`) and added a company
or date discriminator to drafts answerable by boilerplate recurring across filings
(disclosure controls, safe-harbour language).

Relevance is additionally expanded to any chunk whose normalised text equals the anchor's, since
a query answered by one copy of a repeated sentence is answered by any copy.

## Metrics

Written from their definitions in `eval/metrics.py`; no retrieval library involved.

```
Recall@k = |relevant ∩ retrieved@k| / |relevant|
hit@k    = 1 if relevant ∩ retrieved@k else 0
MRR@10   = mean of 1 / rank of first relevant
nDCG@10  = DCG@10 / IDCG@10,  DCG = Σ 1/log2(i+1) over relevant hits
```

`hit@k` was added because Recall@1 is capped at `1/|relevant|`: with chunk overlap, several
chunks can contain the anchor sentence for reasons unrelated to retrieval quality, so a perfect
ranking still scores below 1.0 (measured ceiling on the chunking arms: 0.81–0.83). Both are
reported; hit@k is the one to read across chunking arms.

Latency is `time.perf_counter()` around each stage, accumulated over all 50 queries, then
`np.percentile(x, [50, 95])`. Two measurement details:

- **Models are warmed before timing.** The first torch forward pass is several times slower than
  steady state; unwarmed, that single call landed in the p95 and reported `embed_dense` p95 as
  221 ms instead of 34 ms.
- **End-to-end totals are summed per query, then percentiled** — not summed across per-stage
  p95s, which would overstate the tail by assuming one query is simultaneously worst at
  everything.
- Dense retrieval, sparse retrieval and fusion happen in a single Qdrant round trip and cannot
  be timed apart; the `dense` and `sparse` strategies give the per-leg cost.

## What this does not measure

- **Query style.** This is a *known-item* gold set: every query was written from a specific
  sentence, so it is entity- and figure-heavy. That is the regime where BM25 is strongest.
  Vaguer conceptual questions ("what are the main supply-chain risks?") have no single anchor
  sentence and are where the dense leg should earn its place — they are absent here. **The
  dense-vs-sparse gap is very likely overstated**, and a conceptual-query slice is the highest
  value next addition to this eval.
- **n=50.** Roughly ±0.055 on a proportion near 0.95. Differences under ~0.05 hit@1 are not
  resolvable; this is why the sparse-vs-hybrid and all chunking comparisons come back
  inconclusive.
- **Answer quality.** Only retrieval is scored. Whether the LLM uses the retrieved context
  correctly, and whether single-sentence chunks starve it of context that 256-token chunks
  would supply, is unmeasured — and is a real argument for larger chunks that these retrieval
  metrics cannot see.
- **Corpus scale.** 2000 chunks (retrieval table) and 5k–68k chunks (chunking table). Approximate
  HNSW behaviour and the dense/sparse balance can both shift by an order of magnitude in size.

## Reproducing

```bash
cd backend
docker compose up -d qdrant          # from the repo root

# 1. Freeze the corpus and build the gold set
QDRANT_COLLECTION=sec_filings python -m eval.corpus
python -m scripts.build_gold_set --n 50 --seed 17      # Llama drafts
python -m scripts.finalize_gold_set                    # + manual rewrites

# 2. Baseline, before changing anything
python -m eval.run_eval --strategy dense_legacy --collection sec_filings

# 3. Hybrid collection over the identical corpus, then the strategy sweep
python -m scripts.index_hybrid --collection sec_filings_hybrid
for s in dense sparse hybrid; do
  python -m eval.run_eval --strategy $s --collection sec_filings_hybrid
done
python -m eval.run_eval --strategy hybrid --collection sec_filings_hybrid --rerank --rerank-depth 30

# 4. Compare, and test whether the gaps are real
python -m eval.compare --baseline dense_legacy
python -m eval.significance --baseline dense_legacy --candidate hybrid

# 5. Chunking A/B
python -m scripts.build_chunked_corpora --max-tokens 256 --overlap-tokens 32
python -m scripts.map_gold_to_corpus --chunks eval/data/corpus_item_256_32.jsonl
python -m scripts.index_hybrid --chunks eval/data/corpus_item_256_32.jsonl --collection chunk_item_256_32
python -m eval.run_eval --strategy hybrid --collection chunk_item_256_32 \
  --gold eval/gold/gold_corpus_item_256_32.jsonl --tag chunk_item_256_32_hybrid
```

`dense` run against the rebuilt collection reproduces `dense_legacy` to four decimal places on
every metric, which is the control confirming the reindex is corpus-identical and that later
deltas are attributable to retrieval strategy alone.

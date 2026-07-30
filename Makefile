.DEFAULT_GOAL := help

.PHONY: dev down logs index gold eval-index eval eval-compare test-backend lint-backend lint-frontend build clean help

dev: ## Start all services (dev mode)
	docker compose --profile full up -d

down: ## Stop all services
	docker compose --profile full down

logs: ## Follow all container logs
	docker compose --profile full logs -f

index: ## Run the indexing script inside the API container
	docker compose exec api python -m scripts.index_sec_dataset

gold: ## Rebuild the eval gold set (Llama drafts, then the committed manual rewrites)
	docker compose exec api python -m scripts.build_gold_set --n 50 --seed 17
	docker compose exec api python -m scripts.finalize_gold_set

eval-index: ## Freeze the live corpus and index it into the hybrid eval collection
	docker compose exec api python -m eval.corpus
	docker compose exec api python -m scripts.index_hybrid --collection sec_filings_hybrid

eval: ## Evaluate baseline vs hybrid vs rerank against the gold set (needs eval-index first)
	docker compose exec api python -m eval.run_eval --strategy dense_legacy --collection sec_filings
	docker compose exec api python -m eval.run_eval --strategy sparse --collection sec_filings_hybrid
	docker compose exec api python -m eval.run_eval --strategy hybrid --collection sec_filings_hybrid
	docker compose exec api python -m eval.run_eval --strategy hybrid --collection sec_filings_hybrid \
		--rerank --rerank-depth 30

eval-compare: ## Print the metric/latency table across saved eval runs
	docker compose exec api python -m eval.compare --baseline dense_legacy

test-backend: ## Run pytest inside the API container
	docker compose exec api python -m pytest tests/ -v

lint-backend: ## Run black --check and isort --check inside the API container
	docker compose exec api black --check .
	docker compose exec api isort --check .

lint-frontend: ## Run npm run lint inside the frontend container
	docker compose exec frontend npm run lint

build: ## Build production images
	docker compose -f docker-compose.prod.yml build

clean: ## Stop containers and remove volumes
	docker compose --profile full down -v

help: ## List all targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

export interface QueryRequest {
  query: string;
}

export interface Citation {
  score: number;
  chunk_id?: string;
  company?: string;
  year?: string;
  filingDate?: string;
  docID?: string;
  section?: string;
  snippet?: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
}

export interface HealthResponse {
  status: string;
}

export interface ErrorResponse {
  detail: string;
}
import axios, { AxiosResponse } from 'axios';
import { QueryRequest, QueryResponse, HealthResponse } from '@/types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const ragApi = {
  // Health check
  health: (): Promise<AxiosResponse<HealthResponse>> =>
    api.get<HealthResponse>('/health'),

  // Query the RAG system
  query: (data: QueryRequest): Promise<AxiosResponse<QueryResponse>> =>
    api.post<QueryResponse>('/query', data),
};

export default api;
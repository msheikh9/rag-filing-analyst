import { useState, useCallback } from 'react';
import { ragApi } from '@/services/api';
import { QueryResponse } from '@/types/api';

interface UseRagQueryState {
  loading: boolean;
  results?: QueryResponse;
  error?: string;
  lastQuery?: string;
}

interface UseRagQueryReturn extends UseRagQueryState {
  executeQuery: (query: string) => Promise<void>;
  clearResults: () => void;
}

export function useRagQuery(): UseRagQueryReturn {
  const [state, setState] = useState<UseRagQueryState>({
    loading: false,
    results: undefined,
    error: undefined,
    lastQuery: undefined,
  });

  const executeQuery = useCallback(async (query: string) => {
    if (!query.trim()) {
      setState(prev => ({ ...prev, error: 'Query cannot be empty' }));
      return;
    }

    setState(prev => ({ 
      ...prev, 
      loading: true, 
      error: undefined,
      lastQuery: query 
    }));

    try {
      const response = await ragApi.query({ query: query.trim() });
      setState(prev => ({
        ...prev,
        loading: false,
        results: response.data,
        error: undefined,
      }));
    } catch (error: any) {
      console.error('Query failed:', error);
      
      let errorMessage = 'Failed to execute query';
      
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      } else if (error.code === 'ECONNREFUSED') {
        errorMessage = 'Cannot connect to the backend server. Please check if it\'s running.';
      }

      setState(prev => ({
        ...prev,
        loading: false,
        results: undefined,
        error: errorMessage,
      }));
    }
  }, []);

  const clearResults = useCallback(() => {
    setState({
      loading: false,
      results: undefined,
      error: undefined,
      lastQuery: undefined,
    });
  }, []);

  return {
    ...state,
    executeQuery,
    clearResults,
  };
}
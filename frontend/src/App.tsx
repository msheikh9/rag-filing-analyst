import React, { useState, useEffect } from 'react';
import { Header } from '@/components/Header';
import { QueryForm } from '@/components/QueryForm';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { useRagQuery } from '@/hooks/useRagQuery';
import { ragApi } from '@/services/api';
import { CheckCircle, AlertCircle, Loader } from 'lucide-react';

function App() {
  const [healthStatus, setHealthStatus] = useState<'checking' | 'healthy' | 'error'>('checking');
  const { loading, results, error, lastQuery, executeQuery } = useRagQuery();

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await ragApi.health();
        setHealthStatus('healthy');
      } catch (error) {
        console.error('Health check failed:', error);
        setHealthStatus('error');
      }
    };
    checkHealth();
  }, []);

  const handleQuery = async (query: string) => {
    await executeQuery(query);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 flex flex-col">
      <Header />

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero */}
        <div className="text-center mb-10">
          <h2 className="text-3xl sm:text-4xl font-bold text-secondary-900 tracking-tight">
            Ask anything about SEC filings
          </h2>
          <p className="mt-3 text-lg text-secondary-500 max-w-2xl mx-auto">
            AI-powered analysis of 10-K filings with cited sources
          </p>

          {/* Health Status */}
          <div className="mt-4 inline-flex items-center space-x-1.5 text-xs">
            {healthStatus === 'checking' && (
              <>
                <Loader className="h-3 w-3 animate-spin text-secondary-400" />
                <span className="text-secondary-400">Connecting...</span>
              </>
            )}
            {healthStatus === 'healthy' && (
              <>
                <CheckCircle className="h-3 w-3 text-emerald-500" />
                <span className="text-emerald-600">System online</span>
              </>
            )}
            {healthStatus === 'error' && (
              <>
                <AlertCircle className="h-3 w-3 text-red-500" />
                <span className="text-red-600">Backend unavailable</span>
              </>
            )}
          </div>
        </div>

        {/* Search */}
        <div className="mb-10">
          <QueryForm onSubmit={handleQuery} loading={loading} />
        </div>

        {/* Results */}
        {(results || error) && (
          <div className="animate-fade-in">
            <ResultsDisplay
              results={results}
              query={lastQuery}
              error={error}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="py-6 text-center text-xs text-secondary-400">
        Built with React, FastAPI & Ollama
      </footer>
    </div>
  );
}

export default App;

import React, { useState } from 'react';
import { SearchInput } from '@/components/ui/SearchInput';
import { Button } from '@/components/ui/Button';
import { ArrowRight } from 'lucide-react';

interface QueryFormProps {
  onSubmit: (query: string) => void;
  loading?: boolean;
}

export function QueryForm({ onSubmit, loading }: QueryFormProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !loading) {
      onSubmit(query.trim());
    }
  };

  const examples = [
    'What are the key risk factors?',
    'Summarize business operations',
    'Revenue trends and outlook',
  ];

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="flex space-x-2">
        <div className="flex-1">
          <SearchInput
            placeholder="Ask a question about SEC filings..."
            value={query}
            onChange={setQuery}
            onClear={() => setQuery('')}
            loading={loading}
            className="text-base py-3"
          />
        </div>
        <Button
          type="submit"
          disabled={!query.trim() || loading}
          loading={loading}
          size="lg"
          className="flex items-center space-x-2 px-5"
        >
          <ArrowRight className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs text-secondary-400">Try:</span>
        {examples.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => setQuery(example)}
            className="text-xs px-2.5 py-1 rounded-full border border-secondary-200 text-secondary-500 hover:border-primary-300 hover:text-primary-600 transition-colors"
          >
            {example}
          </button>
        ))}
      </div>
    </form>
  );
}

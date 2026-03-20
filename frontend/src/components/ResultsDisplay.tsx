import React from 'react';
import ReactMarkdown from 'react-markdown';
import { QueryResponse } from '@/types/api';
import { CitationCard } from '@/components/CitationCard';
import { AlertCircle, MessageSquare } from 'lucide-react';

interface ResultsDisplayProps {
  results?: QueryResponse;
  query?: string;
  error?: string;
}

export function ResultsDisplay({ results, query, error }: ResultsDisplayProps) {
  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-5">
        <div className="flex items-center space-x-2 text-red-700">
          <AlertCircle className="h-5 w-5" />
          <span className="font-medium">Error</span>
        </div>
        <p className="text-red-600 mt-2 text-sm">{error}</p>
      </div>
    );
  }

  if (!results) return null;

  const { answer, citations } = results;

  return (
    <div className="space-y-6">
      {/* Answer */}
      <div className="rounded-xl border border-secondary-200 bg-white p-6 shadow-sm">
        <div className="flex items-center space-x-2 mb-4">
          <MessageSquare className="h-4 w-4 text-primary-600" />
          <h3 className="text-sm font-semibold text-primary-700 uppercase tracking-wide">Answer</h3>
        </div>
        {query && (
          <p className="text-xs text-secondary-400 mb-4 border-b border-secondary-100 pb-3">
            {query}
          </p>
        )}
        <div className="prose prose-sm prose-slate max-w-none">
          <ReactMarkdown>{answer}</ReactMarkdown>
        </div>
      </div>

      {/* Citations */}
      {citations && citations.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-secondary-700 uppercase tracking-wide">
              Sources
            </h3>
            <span className="text-xs text-secondary-400">
              {citations.length} citation{citations.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="space-y-3">
            {citations.map((citation, index) => (
              <CitationCard
                key={`${citation.chunk_id}-${index}`}
                citation={citation}
                index={index}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

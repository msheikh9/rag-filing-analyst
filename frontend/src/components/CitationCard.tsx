import React from 'react';
import { Citation } from '@/types/api';
import { formatDate, formatScore } from '@/utils';

interface CitationCardProps {
  citation: Citation;
  index: number;
}

export function CitationCard({ citation, index }: CitationCardProps) {
  const { score, company, year, filingDate, section, snippet } = citation;

  return (
    <div className="rounded-lg border border-secondary-200 bg-white p-4 hover:border-secondary-300 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3 text-xs text-secondary-500 flex-wrap">
          <span className="font-mono font-medium text-secondary-700">#{index + 1}</span>
          {company && <span>CIK {company}</span>}
          {year && <span>{year}</span>}
          {filingDate && <span>{formatDate(filingDate)}</span>}
          {section && section !== 'unknown' && (
            <span>Section {section}</span>
          )}
        </div>
        <span className="shrink-0 text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
          {formatScore(score)}
        </span>
      </div>

      {snippet && (
        <p className="mt-3 text-sm text-secondary-600 leading-relaxed border-l-2 border-primary-200 pl-3">
          {snippet}
        </p>
      )}
    </div>
  );
}

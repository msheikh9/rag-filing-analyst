import React from 'react';
import { Brain, FileText } from 'lucide-react';

export function Header() {
  return (
    <header className="bg-white/80 backdrop-blur-sm border-b border-secondary-200 sticky top-0 z-10">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-14">
          <div className="flex items-center space-x-2.5">
            <div className="flex items-center justify-center w-8 h-8 bg-primary-600 rounded-lg">
              <Brain className="h-4.5 w-4.5 text-white" />
            </div>
            <span className="text-lg font-semibold text-secondary-900">
              RAG Filing Analyst
            </span>
          </div>

          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="text-secondary-500 hover:text-secondary-700 text-sm font-medium transition-colors flex items-center space-x-1.5"
          >
            <FileText className="h-4 w-4" />
            <span>API Docs</span>
          </a>
        </div>
      </div>
    </header>
  );
}

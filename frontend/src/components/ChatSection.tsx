import React, { useState } from 'react';
import { api } from '../api/client';
import type { QueryResponse } from '../types/api';
import { SourceItem } from './SourceItem';

export const ChatSection: React.FC = () => {
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState<{ question: string; answer: QueryResponse }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const result: QueryResponse = await api.queryKnowledgeBase(query);
      setHistory(prev => [{ question: query, answer: result }, ...prev]);
      setQuery('');
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred while querying.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full">
      <div className="flex-1 overflow-y-auto mb-6 space-y-6 pr-2">
        {history.length === 0 && (
          <div className="text-center py-20 text-gray-400">
            <p className="text-xl font-medium">Ask a question to begin</p>
            <p className="text-sm">Upload documents first to build your knowledge base.</p>
          </div>
        )}
        {history.map((turn, idx) => (
          <div key={idx} className="flex flex-col gap-3">
            <div className="flex justify-end">
              <div className="bg-blue-600 text-white px-4 py-2 rounded-2xl rounded-tr-none max-w-[80%]">
                {turn.question}
              </div>
            </div>
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 p-4 rounded-2xl rounded-tl-none max-w-[90%] shadow-sm">
                <p className="text-gray-800 leading-relaxed mb-4">{turn.answer.answer}</p>
                {turn.answer.sources.length > 0 && (
                  <div className="border-t pt-3">
                    <p className="text-xs font-semibold text-gray-400 uppercase mb-2">Sources</p>
                    <div className="space-y-2">
                      {turn.answer.sources.map((src, sIdx) => (
                        <SourceItem key={sIdx} detail={src} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSend} className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask your knowledge base..."
          className="w-full p-4 pr-24 rounded-xl border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all shadow-sm"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="absolute right-2 top-2 bottom-2 px-4 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 transition-colors"
        >
          {loading ? '...' : 'Send'}
        </button>
      </form>
      {error && (
        <div className="mt-2 p-2 text-center text-red-600 text-sm bg-red-50 rounded-lg border border-red-100">
          {error}
        </div>
      )}
    </div>
  );
};

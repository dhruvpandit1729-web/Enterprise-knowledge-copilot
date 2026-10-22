import React, { useState } from 'react';
import { api } from '../api/client';
import type { UploadResponse } from '../types/api';

export const UploadSection: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<{ type: 'idle' | 'loading' | 'success' | 'error'; message: string }>({
    type: 'idle',
    message: '',
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setStatus({ type: 'idle', message: '' });
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setStatus({ type: 'loading', message: 'Uploading and indexing...' });
    try {
      const result: UploadResponse = await api.uploadDocument(file);
      setStatus({
        type: 'success',
        message: `Successfully indexed ${result.filename} (${result.chunks_indexed} chunks).`,
      });
    } catch (error: any) {
      const detail = error.response?.data?.detail || error.message || 'An unexpected error occurred';
      setStatus({ type: 'error', message: detail });
    }
  };

  return (
    <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-200 mb-8">
      <h2 className="text-lg font-semibold mb-4 text-gray-800">Upload Knowledge Base</h2>
      <div className="flex flex-col sm:flex-row gap-4 items-end">
        <div className="flex-1">
          <input
            type="file"
            onChange={handleFileChange}
            accept=".pdf,.docx,.txt,.md"
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />
        </div>
        <button
          onClick={handleUpload}
          disabled={!file || status.type === 'loading'}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 transition-colors"
        >
          {status.type === 'loading' ? 'Processing...' : 'Upload'}
        </button>
      </div>
      {status.message && (
        <div className={`mt-4 p-3 rounded-lg text-sm ${
          status.type === 'success' ? 'bg-green-50 text-green-700' :
          status.type === 'error' ? 'bg-red-50 text-red-700' : 'bg-blue-50 text-blue-700'
        }`}>
          {status.message}
        </div>
      )}
    </div>
  );
};

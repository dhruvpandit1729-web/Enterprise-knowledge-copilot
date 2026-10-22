import React from 'react';
import type { SourceDetail } from '../types/api';

interface SourceItemProps {
  detail: SourceDetail;
}

export const SourceItem: React.FC<SourceItemProps> = ({ detail }) => {
  return (
    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-sm mb-2">
      <div className="flex justify-between items-center mb-1">
        <span className="font-semibold text-gray-700">{detail.source}</span>
        {detail.page_number && (
          <span className="text-xs text-gray-500">Page {detail.page_number}</span>
        )}
      </div>
      {detail.heading && (
        <div className="text-xs font-medium text-blue-600 mb-1">{detail.heading}</div>
      )}
      <p className="text-gray-600 leading-relaxed">{detail.text}</p>
    </div>
  );
};

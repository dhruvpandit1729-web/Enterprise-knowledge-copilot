import React from 'react';
import { UploadSection } from './components/UploadSection';
import { ChatSection } from './components/ChatSection';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans p-4 sm:p-8">
      <div className="max-w-5xl mx-auto">
        <header className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Knowledge Copilot</h1>
            <p className="text-gray-500">Enterprise Document Intelligence</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-gray-600">Backend Online</span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1">
            <UploadSection />
          </div>
          <div className="lg:col-span-2 h-[calc(100vh-200px)] flex flex-col">
            <ChatSection />
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;

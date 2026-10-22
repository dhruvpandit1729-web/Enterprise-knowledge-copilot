import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  async healthCheck() {
    const response = await apiClient.get('/health');
    return response.data;
  },

  async uploadDocument(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post('/documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async queryKnowledgeBase(question: string, limit: number = 5) {
    const response = await apiClient.post('/query', {
      question,
      limit,
    });
    return response.data;
  },
};

export default apiClient;

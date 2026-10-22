export interface SourceDetail {
  source: string;
  text: string;
  page_number?: number;
  heading?: string;
}

export interface QueryResponse {
  answer: string;
  sources: SourceDetail[];
  metadata: {
    model: string;
    chunks_retrieved: number;
  };
}

export interface UploadResponse {
  status: string;
  filename: string;
  chunks_indexed: number;
}
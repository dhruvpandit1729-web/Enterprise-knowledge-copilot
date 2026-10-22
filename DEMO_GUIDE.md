# Demo Guide: Enterprise Knowledge Copilot

This guide provides a step-by-step walkthrough to demonstrate the capabilities of the Enterprise Knowledge Copilot on a local machine.

## 🚀 Quick Start

### 1. Launch the System
1. **Start the Backend**:
   ```powershell
   .\.venv\Scripts\python app/main.py
   ```
2. **Start the Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
3. **Open the Browser**: Navigate to `http://localhost:5173`.

---

## 📖 Demonstration Workflow

### Step 1: Verify System Health
Check the top-right corner of the UI. The **"Backend Online"** indicator should be visible. Alternatively, visit `http://localhost:8000/health` to see the `{"status": "ok"}` response.

### Step 2: Knowledge Ingestion
1. Locate the **"Upload Knowledge Base"** section in the left sidebar.
2. Select a sample document (e.g., `data/test_docs/multi_page_document.pdf`).
3. Click **Upload**.
4. **Verify**: You should see a success message: *"Successfully indexed multi_page_document.pdf (N chunks)."*

### Step 3: Grounded Querying
1. In the chat input, type a specific question based on the uploaded document:
   - *Example*: `"When is the final project review scheduled?"`
2. Press **Send**.
3. **Inspect the Result**:
   - **The Answer**: You should see: *"The final project review is scheduled for December 15, 2026."*
   - **The Citations**: Below the answer, a "Sources" section will appear.
   - **Metadata**: Verify that the source is `multi_page_document.pdf` and specifies **Page 3**.

### Step 4: Testing Abstention
1. Ask a question completely unrelated to any uploaded documents:
   - *Example*: `"What is the secret recipe for the perfect chocolate cake?"*
2. **Verify**: The system should respond with: *"The available documents do not provide enough information."*

---

## 🛠 Troubleshooting the Demo
- **Ollama not responding**: Ensure Ollama is running in the background and you have pulled the model: `ollama pull qwen2.5:7b`.
- **File Upload Error**: Ensure the file is under 10MB and is one of the supported formats (.pdf, .docx, .txt, .md).
- **CORS Issues**: Ensure you started the backend using `app/main.py` rather than a raw uvicorn command without the middleware.

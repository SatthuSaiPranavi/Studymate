# 📘 StudyMate — AI-Powered PDF Q&A System

StudyMate is an AI-powered academic assistant that enables students to interact with their study materials (textbooks, lecture notes, research papers) in a **conversational Q&A format**. Instead of manually searching large PDFs, users can upload documents and ask questions in natural language. StudyMate retrieves relevant content and generates **grounded answers with references**.

---

## 🚀 Features

- Multi-PDF upload & preprocessing (PyMuPDF)
- Text chunking with overlap for better context
- Semantic embeddings with **SentenceTransformers**
- **FAISS** vector search for fast retrieval
- Answer generation using **IBM Watsonx LLM (Mixtral-8x7B-Instruct)**
- **Streamlit UI**: drag-and-drop PDFs, ask questions, view references
- Session history & downloadable Q&A transcripts
- Environment variables for secure API key handling (`.env`)

---

## 🏗️ Tech Stack

- **PDF Parsing**: [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Embeddings**: [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **Vector Index**: [FAISS](https://github.com/facebookresearch/faiss)
- **LLM Backend**: [IBM Watsonx.ai](https://www.ibm.com/cloud/watsonx) (`mistralai/mixtral-8x7b-instruct-v01`)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Config**: [python-dotenv](https://pypi.org/project/python-dotenv/)

---

## 📂 Project Structure

studymate/
├── app.py # Streamlit frontend
├── extractor.py # PDF parsing
├── chunker.py # Text chunking logic
├── embed_index.py # Embedding + FAISS index
├── retriever.py # Query retrieval logic
├── llm_client.py # Watsonx integration
├── utils.py # Helpers (env loader, logging)
├── requirements.txt # Dependencies
├── .env.example # Env variable template
└── README.md # Project docs


---

## ⚙️ Setup & Installation

### 1. Clone repo
```bash
git clone https://github.com/your-username/studymate.git
cd studymate

2. Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux

3. Install dependencies
pip install -r requirements.txt

4. Setup environment variables

Copy .env.example → .env and update with your credentials:

IBM_API_KEY=your_ibm_api_key_here
IBM_PROJECT_ID=your_project_id_here
IBM_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=mistralai/mixtral-8x7b-instruct-v01
MAX_NEW_TOKENS=300
TEMPERATURE=0.5

5. Run the app
streamlit run app.py

output:<img width="1600" height="862" alt="image" src="https://github.com/user-attachments/assets/c958325e-4c9d-4992-a189-f469c999dc53" />

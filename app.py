import streamlit as st
from extractor import extract_text_from_pdf
from chunker import chunk_text
from embed_index import embed_chunks, build_faiss_index, EMBED_MODEL
from retriever import retrieve_top_k
from llm_client import build_prompt, ask_watsonx
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('IBM_API_KEY')
PROJECT_ID = os.getenv('IBM_PROJECT_ID')
IBM_URL = os.getenv('IBM_URL')
MODEL_ID = os.getenv('IBM_MODEL_ID', 'mistralai/mistral-large')

st.set_page_config(page_title='StudyMate', layout='wide')
st.title('📘 StudyMate — AI PDF Q&A')

uploaded_files = st.file_uploader('Upload PDFs', accept_multiple_files=True, type=['pdf'])

if 'history' not in st.session_state:
    st.session_state['history'] = []

if uploaded_files and st.button('Process PDFs'):
    all_chunks = []
    for f in uploaded_files:
        text = extract_text_from_pdf(f)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    embed_model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_chunks(all_chunks, model_name=EMBED_MODEL)
    index = build_faiss_index(embeddings)
    st.session_state['chunks'] = all_chunks
    st.session_state['index'] = index
    st.session_state['embed_model'] = embed_model
    st.success('PDFs processed and indexed.')

question = st.text_input('Ask a question:')
if st.button('Submit') and question.strip():
    chunks = st.session_state.get('chunks', [])
    index = st.session_state.get('index', None)
    embed_model = st.session_state.get('embed_model', None)
    if not index:
        st.error('No index found — please upload and process PDFs first.')
    else:
        results = retrieve_top_k(question, embed_model, index, chunks, k=3)
        retrieved_texts = [r[0] for r in results]
        prompt = build_prompt(question, retrieved_texts)
        response = ask_watsonx(prompt, API_KEY, IBM_URL, PROJECT_ID, model_id=MODEL_ID)
        st.markdown('### Answer')
        st.write(response)
        with st.expander('Referenced Chunks'):
            for ct, score in results:
                st.write(f"(score: {score:.4f})\n{ct}\n---")
        st.session_state['history'].append({'q': question, 'a': response})
        if st.button('Download Q&A History'):
            history_text = "\n".join([f"Q: {h['q']}\nA: {h['a']}\n" for h in st.session_state['history']])
            st.download_button("Download", history_text, "history.txt")

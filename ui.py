import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        self.documents = documents
        texts = [doc['content'] for doc in documents]
        self.embeddings = self.vectorizer.fit_transform(texts)

    def get_relevant_documents(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        return [self.documents[i] for i in top_indices]

def process_pdfs(files):
    documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file.read())
            temp_file.flush()
            
            try:
                reader = PdfReader(temp_file.name)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append({"page": i + 1, "content": text})
                st.success(f"Processed: {file.name}")
            except Exception as e:
                st.error(f"Error with {file.name}: {str(e)}")
            finally:
                os.unlink(temp_file.name)
    return documents

def main():
    st.title("📚 PDF Question Answering System")
    
    # Initialize session state
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    # Sidebar for setup
    with st.sidebar:
        st.header("Setup")
        st.write("""
            1. Upload PDF files
            2. Click 'Process Documents'
            3. Ask questions about your documents
        """)
        
        files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
        
        if files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                documents = process_pdfs(files)
                
                if not documents:
                    st.error("No documents were successfully processed.")
                    return
                
                # Setup retriever
                retriever = SimpleRetriever()
                retriever.add_documents(documents)
                
                st.session_state.retriever = retriever
                st.success("✅ System is ready!")
    
    # Main area for Q&A
    if st.session_state.retriever:
        st.write("### Ask a Question")
        question = st.text_input("Enter your question about the documents:")
        
        if st.button("Get Answer") and question:
            with st.spinner("Finding answer..."):
                relevant_docs = st.session_state.retriever.get_relevant_documents(question)
                
                st.markdown("### 📝 Answer")
                answer = " ".join([doc['content'] for doc in relevant_docs])
                st.write(answer if answer else "No relevant information found.")
                
                st.markdown("### 📚 Sources")
                for doc in relevant_docs:
                    with st.expander(f"📄 Page {doc['page']}"):
                        st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px;'>{doc['content']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()

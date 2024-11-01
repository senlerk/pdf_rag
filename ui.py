import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import traceback

# Set page configuration
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)

class SimpleVectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = []
        self.embeddings = None

    def add_documents(self, documents):
        self.docs = documents
        texts = [doc.page_content for doc in documents]
        self.embeddings = self.model.encode(texts)

    def similarity_search(self, query, k=3):
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.docs[i] for i in top_k_indices]

    def as_retriever(self, search_kwargs=None):
        k = search_kwargs.get('k', 3) if search_kwargs else 3
        return lambda x: self.similarity_search(x, k=k)

def process_pdfs(uploaded_files):
    """Process uploaded PDF files"""
    documents = []
    with st.spinner("Processing PDF files..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_file_path)
                documents.extend(loader.load())
                st.success(f"✅ Processed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    return documents

def setup_qa_system(documents):
    """Set up the QA system"""
    with st.spinner("Setting up the QA system..."):
        try:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)

            # Create vector store
            vectorstore = SimpleVectorStore()
            with st.spinner("Creating document embeddings..."):
                vectorstore.add_documents(splits)

            # Setup QA chain
            template = """
            Use the following context to answer the question. If you don't know the answer or can't find it in the context, just say "I don't have enough information to answer this question."

            Context: {context}
            
            Question: {question}
            
            Answer: """

            QA_PROMPT = PromptTemplate(
                template=template,
                input_variables=['context', 'question']
            )

            llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                temperature=0
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': QA_PROMPT}
            )
            
            return qa_chain

        except Exception as e:
            st.error(f"Error setting up QA system: {str(e)}")
            st.error(f"Detailed error: {traceback.format_exc()}")
            raise e

def main():
    st.title("📚 PDF Question Answering System")
    
    st.markdown("""
    ### Welcome to the PDF Q&A Assistant! 
    
    This system allows you to:
    1. Upload PDF documents 📄
    2. Ask questions about their content ❓
    3. Get accurate answers with source references 📝
    """)
    
    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    
    # Sidebar
    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type=['pdf'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Process PDFs"):
                    documents = process_pdfs(uploaded_files)
                    if documents:
                        st.session_state.qa_chain = setup_qa_system(documents)
                        st.success("✅ System ready!")
    
    # Main Q&A interface
    if st.session_state.qa_chain:
        st.header("Ask Questions")
        question = st.text_input("Enter your question:")
        
        if st.button("Get Answer"):
            if question:
                try:
                    with st.spinner("Searching..."):
                        result = st.session_state.qa_chain({"query": question})
                        
                    st.markdown("### Answer")
                    st.write(result["result"])
                    
                    st.markdown("### Sources")
                    for doc in result["source_documents"]:
                        with st.expander(f"Page {doc.metadata.get('page', 'unknown')}"):
                            st.write(doc.page_content)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and LangChain")

if __name__ == "__main__":
    main()
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import torch
import pypdf

# Download required NLTK data
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .stAlert {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .source-text {
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

class SimpleVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.docs = []
        self.doc_embeddings = None

    def add_documents(self, documents):
        self.docs = documents
        texts = [doc.page_content for doc in documents]
        self.doc_embeddings = self.embeddings.embed_documents(texts)

    def similarity_search_with_score(self, query, k=3):
        query_embedding = self.embeddings.embed_query(query)
        similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(self.docs[i], similarities[i]) for i in top_k_indices]

    def as_retriever(self, search_kwargs=None):
        def retrieve(query):
            results = self.similarity_search_with_score(query, k=search_kwargs.get('k', 3))
            return [doc for doc, _ in results]
        return retrieve

def initialize_session_state():
    """Initialize session state variables"""
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'embeddings_model' not in st.session_state:
        st.session_state.embeddings_model = None

def verify_api_key(api_key):
    """Verify OpenAI API key"""
    llm = ChatOpenAI(
        api_key=api_key,
        model_name="gpt-4-turbo-preview",
        temperature=0
    )
    try:
        llm.invoke("test")
        return True
    except Exception as e:
        st.error(f"API Key Error: {str(e)}")
        return False

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
                os.unlink(tmp_file_path)
    
    if not documents:
        st.error("No documents were successfully processed.")
        return None
    
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

            with st.spinner("Loading embedding model (this might take a few minutes the first time)..."):
                # Create or reuse embeddings model
                if st.session_state.embeddings_model is None:
                    st.session_state.embeddings_model = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-mpnet-base-v2",
                        cache_folder="/tmp/huggingface"
                    )
                
                vectorstore = SimpleVectorStore(st.session_state.embeddings_model)
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
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            raise e

def main():
    initialize_session_state()
    
    st.title("📚 PDF Question Answering System")
    
    # Add a description with markdown formatting
    st.markdown("""
    ### Welcome to the PDF Q&A Assistant! 
    
    This system allows you to:
    1. Upload PDF documents 📄
    2. Ask questions about their content ❓
    3. Get accurate answers with source references 📝
    
    To get started:
    1. Enter your OpenAI API key in the sidebar
    2. Upload one or more PDF files
    3. Click "Process PDFs" to analyze the documents
    4. Start asking questions!
    """)
    
    # Sidebar for API key and file upload
    with st.sidebar:
        st.header("Setup")
        with st.expander("ℹ️ API Key Instructions", expanded=False):
            st.markdown("""
            1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
            2. Create a new secret key
            3. Copy and paste it here
            4. Make sure you have GPT-4 access enabled
            """)
        
        api_key = st.text_input("Enter OpenAI API Key", type="password", help="Enter your OpenAI API key with GPT-4 access")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            if not verify_api_key(api_key):
                st.error("❌ Invalid API key or no GPT-4 access")
                return
            
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Select one or more PDF files to analyze"
            )
            
            if uploaded_files:
                if set(f.name for f in uploaded_files) != set(f.name for f in st.session_state.uploaded_files):
                    st.session_state.uploaded_files = uploaded_files
                    st.session_state.processing_complete = False
                    
                if not st.session_state.processing_complete:
                    if st.button("Process PDFs", help="Click to analyze the uploaded documents"):
                        try:
                            documents = process_pdfs(uploaded_files)
                            if documents:
                                st.session_state.qa_chain = setup_qa_system(documents)
                                st.session_state.processing_complete = True
                                st.success("✅ System ready for questions!")
                                st.balloons()
                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")
    
    # Main area for Q&A
    if st.session_state.processing_complete:
        st.header("Ask Questions")
        
        with st.expander("💡 Tips for better answers", expanded=False):
            st.markdown("""
            - Be specific in your questions
            - Ask about one topic at a time
            - If you don't get a satisfactory answer, try rephrasing the question
            - Questions starting with What, How, Why, When often work well
            """)
        
        question = st.text_input("Enter your question:", help="Type your question about the uploaded documents")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", help="Click to get your answer")
        
        if search_button:
            if question:
                try:
                    with st.spinner("Searching through documents..."):
                        result = st.session_state.qa_chain({"query": question})
                        
                    st.markdown("### 📝 Answer")
                    st.write(result["result"])
                    
                    st.markdown("### 📚 Sources")
                    for doc in result["source_documents"]:
                        with st.expander(f"📄 Page {doc.metadata.get('page', 'unknown')}"):
                            st.markdown(f"<div class='source-text'>{doc.page_content}</div>",
                                      unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    with st.expander("🔍 Debug Information"):
                        st.error(f"Detailed error: {traceback.format_exc()}")
            else:
                st.warning("Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and LangChain")

if __name__ == "__main__":
    main()
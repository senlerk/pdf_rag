import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Changed from FAISS to Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

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
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

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
    
    return documents

def setup_qa_system(documents):
    """Set up the QA system"""
    with st.spinner("Setting up the QA system..."):
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store using Chroma instead of FAISS
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

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

def main():
    initialize_session_state()
    
    st.title("📚 PDF Question Answering System")
    st.markdown("""
    Upload your PDF documents and ask questions about their content. 
    The system will provide answers based on the information found in the documents.
    """)
    
    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            if not verify_api_key(api_key):
                st.error("❌ Invalid API key or no GPT-4 access")
                return
            
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type=['pdf'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if set(f.name for f in uploaded_files) != set(f.name for f in st.session_state.uploaded_files):
                    st.session_state.uploaded_files = uploaded_files
                    st.session_state.processing_complete = False
                    
                if not st.session_state.processing_complete:
                    if st.button("Process PDFs"):
                        documents = process_pdfs(uploaded_files)
                        if documents:
                            st.session_state.qa_chain = setup_qa_system(documents)
                            st.session_state.processing_complete = True
                            st.success("✅ System ready for questions!")
                            st.balloons()
    
    if st.session_state.processing_complete:
        st.header("Ask Questions")
        question = st.text_input("Enter your question:")
        
        if st.button("Get Answer"):
            if question:
                try:
                    with st.spinner("Searching for answer..."):
                        result = st.session_state.qa_chain({"query": question})
                        
                    st.markdown("### Answer")
                    st.write(result["result"])
                    
                    st.markdown("### Sources")
                    for doc in result["source_documents"]:
                        with st.expander(f"Page {doc.metadata.get('page', 'unknown')}"):
                            st.markdown(f"<div class='source-text'>{doc.page_content}</div>",
                                      unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and LangChain")

if __name__ == "__main__":
    main()
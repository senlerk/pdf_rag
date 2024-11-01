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
import pypdf  # Added this import

# Set page configuration
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)

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

            # Create vector store using our simple implementation
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            
            vectorstore = SimpleVectorStore(embeddings)
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
            raise e

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
                        try:
                            documents = process_pdfs(uploaded_files)
                            if documents:
                                st.session_state.qa_chain = setup_qa_system(documents)
                                st.session_state.processing_complete = True
                                st.success("✅ System ready for questions!")
                                st.balloons()
                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")
    
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
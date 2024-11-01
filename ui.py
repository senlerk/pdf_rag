import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set page configuration
st.set_page_config(page_title="PDF Q&A System", layout="wide")

class SimpleRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        self.documents = documents
        texts = [doc.page_content for doc in documents]
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
            temp_file.write(file.getvalue())
            temp_file.flush()
            
            try:
                loader = PyPDFLoader(temp_file.name)
                documents.extend(loader.load())
                st.success(f"Processed: {file.name}")
            except Exception as e:
                st.error(f"Error with {file.name}: {str(e)}")
            finally:
                os.unlink(temp_file.name)
    return documents

def main():
    st.title("📚 PDF Question Answering System")
    
    # Initialize session state
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    
    # Sidebar for setup
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
        
        if api_key and files:
            if st.button("Process Documents"):
                try:
                    os.environ["OPENAI_API_KEY"] = api_key
                    
                    with st.spinner("Processing documents..."):
                        # Process documents
                        documents = process_pdfs(files)
                        
                        # Split documents
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        splits = splitter.split_documents(documents)
                        
                        # Setup retriever
                        retriever = SimpleRetriever()
                        retriever.add_documents(splits)
                        
                        # Setup LLM
                        llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview")
                        
                        # Create chain
                        template = """Use these pieces of context to answer the question.
                        If you don't know the answer, just say "I don't have enough information."
                        
                        Context: {context}
                        Question: {question}
                        
                        Answer: """
                        
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever.get_relevant_documents,
                            chain_type_kwargs={
                                "prompt": PromptTemplate(
                                    template=template,
                                    input_variables=["context", "question"]
                                )
                            }
                        )
                        
                        st.session_state.qa_system = qa_chain
                        st.success("✅ System is ready!")
                except Exception as e:
                    st.error(f"Error setting up system: {str(e)}")

    # Main area
    if st.session_state.qa_system:
        st.write("### Ask a Question")
        question = st.text_input("Enter your question about the documents:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Finding answer..."):
                    try:
                        response = st.session_state.qa_system({"query": question})
                        st.write("### Answer")
                        st.write(response["result"])
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
            else:
                st.warning("Please enter a question.")

    # Footer
    st.markdown("---")
    st.markdown("Made with Streamlit")

if __name__ == "__main__":
    main()
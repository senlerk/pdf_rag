import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="PDF Q&A System", layout="wide")

class BasicEmbeddings:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base')
        self.model = AutoModel.from_pretrained('microsoft/mpnet-base')
        
    def get_embeddings(self, texts):
        # Tokenize and get model output
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        outputs = self.model(**inputs)
        
        # Use mean pooling
        attention_mask = inputs['attention_mask']
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(torch.sum(mask, 1), min=1e-9)
        mean_pooled = summed / counts
        
        # Normalize
        return torch.nn.functional.normalize(mean_pooled, p=2, dim=1).detach().numpy()

class SimpleRetriever:
    def __init__(self):
        self.embedder = BasicEmbeddings()
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.embeddings = self.embedder.get_embeddings(texts)

    def get_relevant_documents(self, query):
        query_embedding = self.embedder.get_embeddings([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]  # Get top 3
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
                    template = """Use the following pieces of context to answer the question. 
                    If you don't know the answer, just say "I don't know."
                    
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
                    st.success("Ready!")

    # Main area
    if st.session_state.qa_system:
        question = st.text_input("Ask a question about your documents:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Searching..."):
                    try:
                        response = st.session_state.qa_system({"query": question})
                        st.write("### Answer")
                        st.write(response["result"])
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
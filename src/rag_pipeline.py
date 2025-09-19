from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os

def get_embedding_function(model_name):
    """
    Initializes the Sentence Transformer embedding function.
    """
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformerEmbeddings(model_name=model_name)

def create_vector_db(chunks, embedding_function, persist_path):
    """
    Creates or loads a Chroma vector database from document chunks.
    """
    if os.path.exists(persist_path):
        print(f"Loading existing vector DB from: {persist_path}")
        vector_db = Chroma(
            persist_directory=persist_path,
            embedding_function=embedding_function
        )
    else:
        print("Creating new vector DB...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=persist_path
        )
        print(f"Vector DB created and persisted to: {persist_path}")
    
    return vector_db

def load_llm(model_repo, model_file, n_gpu_layers, n_batch, n_ctx, max_tokens):
    """
    Downloads the GGUF model from Hugging Face and initializes Llama.
    """
    print("Downloading LLM model...")
    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_file
    )
    
    print("Initializing Llama model...")
    lcpp_llm = Llama(
        model_path=model_path,
        n_threads=os.cpu_count() or 4,  # Use all available cores or default to 4
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        temperature=0.0,  # Set temperature to 0 for deterministic answers
        top_p=0.1,
        verbose=False # Set to True for detailed Llama logging
    )
    print("LLM loaded successfully.")
    return lcpp_llm

def create_qa_chain(llm, vector_db):
    """
    Creates the RetrievalQA chain with the custom prompt template from the notebook.
    """
    print("Creating QA Chain...")
    
    # Define the custom prompt template
    template = """<s>[INST] You are a helpful business analyst assistant. 
    Use the following context to answer the question. Context: {context}. 
    Based on the context provided, answer the following question: {question}. 
    Provide only the relevant answer from the context, do not add any external information.
    If the answer is not in the context, simply state "I am sorry, I cannot find this information in the provided report."
    [/INST]</s>"""
    
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    print("QA Chain created successfully.")
    return qa_chain
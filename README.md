# RAG-based Semantic Search for Business Report Analysis

This project uses a Retrieval-Augmented Generation (RAG) pipeline to perform semantic search and answer questions on the Harvard Business Review report, "How Apple is Organized for Innovation."

It leverages a local GGUF model (Mistral 7B Instruct) running via `llama-cpp-python` and the LangChain framework to create a question-answering system based *only* on the contents of the provided document.

---

## Project Structure
│
├── .gitignore                  # Ignore cache files, virtual environments, and local DB storage
├── LICENSE                     # Open-source license (MIT)
├── README.md                   # Project overview, setup guide, and usage instructions
├── requirements.txt            # Python dependencies for running the pipeline
│
├── data/
│   └── apple_hbr.pdf           # Input PDF (business report); must be placed here before running
│
├── notebooks/
│   ├── MLS_Apple_HBR.ipynb     # Main demo notebook: RAG Q&A workflow on Apple HBR report
│   └── auto_pipeline_run.ipynb # End-to-end automated pipeline execution notebook
│
└── src/
    ├── config.py               # Centralized configuration (paths, models, hyperparameters)
    ├── data_loader.py          # Handles PDF loading and chunking into text segments
    └── rag_pipeline.py         # Builds embeddings, vector DB (FAISS), loads LLM, and creates QA chain

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_GITHUB_REPO_URL]
    cd RAG-Business-Report-Project
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    This project requires specific libraries. For GPU support (NVIDIA), you must have the CUDA Toolkit installed.

    ```bash
    # Install Python packages
    pip install -r requirements.txt
    
    # Install llama-cpp-python with GPU (CUBLAS) support
    # (This command is from the notebook and assumes a CUDA environment)
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir
    
    # If you are on CPU ONLY, install this instead:
    # CMAKE_ARGS="-DLLAMA_CUBLAS=off" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir
    ```

4.  **Add Data:**
    Place the target PDF report inside the `/data/` folder and ensure it is named `apple_hbr.pdf`.

5.  **Run the Notebook:**
    Open and run the cells in `MLS_Apple_HBR (1).ipynb` sequentially. The notebook will:
    * Import configurations from `src/config.py`.
    * Call `src/data_loader.py` to load and chunk the PDF.
    * Call `src/rag_pipeline.py` to:
        1.  Load the embedding model.
        2.  Create and persist a Chroma vector database in the `./db_chroma/` folder.
        3.  Download the Mistral-7B-Instruct-v0.2 GGUF model.
        4.  Initialize the Llama LLM and the final RetrievalQA chain.
    * The final cells demonstrate how to ask questions and receive answers sourced from the document.

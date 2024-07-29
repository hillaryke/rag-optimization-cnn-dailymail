# RAG System Optimization for cnn Daily Mail

## Objective

The objective of this project is to:

- Develop a Retrieval-Augmented Generation (RAG) system for the CNN/Daily Mail
  dataset.
- Benchmark the performance of the RAG system using RAGAS.
- Implement an optimization technique to improve the performance of the RAG
  system.

## Getting Started

### Prerequisites

- Python 3.9+
- poetry (refer [here](https://python-poetry.org/docs/#installation) for
  installation instructions)

### Installation

1. Clone the repository

```bash
git clone git@github.com:hillaryke/contract-qa-high-precision-rag.git
```

2. Add OpenAI API key to `.env` file

- Run the command below add your OpenAI API key:

```bash
echo "OPENAI_API_KEY=<your_openai_api_key>" > .env
```

- Replace `<your_openai_api_key>` with your OpenAI API key.

3. Install dependencies

```bash
poetry install
```

## Approach
I followed the following stesp to develop the RAG system and later perform optimization.

1. Project setup
2. Data preparation and loading
3. RAG system setup.
4. Evaluation pipeline setup using RAGAS.
5. Run and analyze baseline benchmark evaluation.
5. Identify areas of improvement.
7. Identify optimization techniques.
6. Implement optimization techniques.

### Project setup
I created a new project using poetry and added the necessary dependencies i.e Lanchain tools and RAGAS.

### Data preparation and loading

I used the CNN/Daily Mail dataset for this project. The dataset is available on the Hugging Face datasets library. I loaded the dataset using the `datasets` library and extracted the necessary fields for the RAG system.

```dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]")```

The line above loads the first 1000 examples from the validation split of the CNN/Daily Mail dataset.
The function to do this can found under `src/rag_pipeline/load_docs.py`

### RAG system setup
#### Basic Rag system
Having some experience with using ChromaDB vectorstore, I decided to use it for the initial setup of the RAG system.

I used the steps to setup my basic RAG system as follows:
1. Load documents: I loaded the dataset from csv file, I then retrieved the `article` column only for use as page_content to get my documents.
2. Split documents: Using langchain `RecursiveChararacterTestSplitter`, I split the documents into small chunks.
3. Create vectorstore: I used `langchain_chroma` to create a vectorstore from the split documents.
4. Setup LLM: I used OpenAI's gpt-3.5-turbo for testing the setup. I would then upgrade to gpt-4o when ready.
5. Create RAG chain that can be used to retrieve documents and generate answers. The RAG chain was simple using [`RetrievalQA`](https://docs.smith.langchain.com/old/cookbook/hub-examples/retrieval-qa-chain) from langchain.

#### Advancing the RAG system with best practices
I followed these steps to setup the RAG system and make it reusable and scalable:
1. Created a class `RAGSystem` that would be used to setup the RAG system. The class can be found under `src/rag_pipeline/rag_system.py`
2. Added the methods and classes i.e to load documents, split documents, create vectorstore, setup LLM, create RAG chain and more.
3. Usage: I could import the class and initialize as follows:
    ```
    from src.rag_pipeline.rag_system import RAGSystem

    rag_system = RAGSystem(
      model_name = "gpt-4o",
      embeddings = embeddings,
      # Here you can add more parameters to customize the RAG system
    )

    rag_system.initialize()
    ```

#### Integrating pgvector for vectordatabase
I decided to integrate pgvector vectorstore for improved performance.
I followed the steps below to integrate pgvector:
1. Setup pgvector database:
    - Install the necessary dependencies using poetry for pgvector including `langchain-pgvector` and `pgvector`.
    - Using docker, I installed pgvector database which uses postgresql as the database.
    - I created a docker-compose file to install the database. The file can be found under `docker-compose.yml` containing the pgvector service and the database service.
    - Create a script to create `vector` extension and create embeddings table. The script is under `scripts/init.sql`. However, when using langchain-pgvector, the script is not necessary as the library will create the table and extension for us.
    - I started the database using the command `docker compose up -d`.
    - I wrote a make target to save this command. The target can be found under `Makefile` as `up`. Other commands can be found under the `Makefile` as well. The `Makefile` allows me to easily document and run commands critical to the project.

2. Add pgvector vectorstore to the RAG system




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Contact

For any inquiries, please contact through email
[Hillary Kipkemoi](mailto:hillary6k@gmail.com)

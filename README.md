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

I followed the following stesp to develop the RAG system and later perform
optimization.

1. Project setup
2. Data preparation and loading
3. RAG system setup.
4. Evaluation pipeline setup using RAGAS.
5. Run and analyze baseline benchmark evaluation.
6. Identify areas of improvement.
7. Identify optimization techniques.
8. Implement optimization techniques.

### Project setup

I created a new project using poetry and added the necessary dependencies i.e
Lanchain tools and RAGAS.

### Data preparation and loading

I used the CNN/Daily Mail dataset for this project. The dataset is available on
the Hugging Face datasets library. I loaded the dataset using the `datasets`
library and extracted the necessary fields for the RAG system.

`dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]")`

The line above loads the first 1000 examples from the validation split of the
CNN/Daily Mail dataset. The function to do this can found under
`src/rag_pipeline/load_docs.py`

## RAG system setup

### Basic Rag system

Having some experience with using ChromaDB vectorstore, I decided to use it for
the initial setup of the RAG system.

I used the steps to setup my basic RAG system as follows:

1. Load documents: I loaded the dataset from csv file, I then retrieved the
   `article` column only for use as page_content to get my documents.
2. Split documents: Using langchain `RecursiveChararacterTestSplitter`, I split
   the documents into small chunks.
3. Create vectorstore: I used `langchain_chroma` to create a vectorstore from
   the split documents.
4. Setup LLM: I used OpenAI's gpt-3.5-turbo for testing the setup. I would then
   upgrade to gpt-4o when ready.
5. Create RAG chain that can be used to retrieve documents and generate answers.
   The RAG chain was simple using
   [`RetrievalQA`](https://docs.smith.langchain.com/old/cookbook/hub-examples/retrieval-qa-chain)
   from langchain.

### Advancing the RAG system with best practices

I followed these steps to setup the RAG system and make it reusable and
scalable:

1. Created a class `RAGSystem` that would be used to setup the RAG system. The
   class can be found under `src/rag_pipeline/rag_system.py`
2. Added the methods and classes i.e to load documents, split documents, create
   vectorstore, setup LLM, create RAG chain and more.
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

### Create custom rag_chain:

I created a custom rag_chain that would be used to retrieve documents and
generate to allow customazibality over the RetrievalQA chain. The custom
rag_chain can be found under `src/rag_pipeline/rag_utils.py`

These are the steps I followed to create the custom rag_chain:

- Defining Helper Functions: Two helper functions are defined: `format_docs`,
  which formats a list of documents into a concatenated string, and
  `ragas_output_parser`, which extracts page content from a list of documents.

- **Custom Prompt Templates for generator llm**: I created custom prompt
  template `GENERATOR_TEMPLATE` in the settings (`misc/settings.py`). This
  template is then combined with a language model (`llm`) and a string output
  parser to form the generator component of the RAG chain.

- **Creating Context Retriever**: A `RunnableParallel` Langchain object named
  `context_retriever` is set up to handle the retrieval of relevant documents.
  It combines the retriever with the `format_docs` function to fetch and format
  the context, while passing through the question as-is.

- **Filtering Dataset**: A `RunnableLambda` Langchain object named
  `filter_langsmith_dataset` is created to filter the input, ensuring that only
  the question is processed if the input is a dictionary. Note this function was
  initially used for RAGAS+LangSmith evaluation, however it works well for any
  dataset.

- **Constructing the RAG Chain**: The final RAG chain is constructed as a
  `RunnableParallel` Langchain object. It does the following: - processes the
  question through the filter - retrieves and formats the context, - generates
  an answer using the generator, and - extracts contexts using the
  `ragas_output_parser`.

- **Returning the RAG Chain**: The function returns the constructed
  RunnableParallel object, representing the complete RAG chain setup for
  LangSmith integration.

## Integrating pgvector for vectordatabase

I decided to integrate pgvector vectorstore for improved performance. I followed
the steps below to integrate pgvector:

1. Setup pgvector database:

   - Install the necessary dependencies using poetry for pgvector including
     `langchain-pgvector` and `pgvector`.
   - Using docker, I installed pgvector database which uses postgresql as the
     database.
   - I created a docker-compose file to install the database. The file can be
     found under `docker-compose.yml` containing the pgvector service and the
     database service.
   - Create a script to create `vector` extension and create embeddings table.
     The script is under `scripts/init.sql`. However, when using
     langchain-pgvector, the script is not necessary as the library will create
     the table and extension for us.
   - I started the database using the command `docker compose up -d`.
   - I wrote a make target to save this command. The target can be found under
     `Makefile` as `up`. Other commands can be found under the `Makefile` as
     well. The `Makefile` allows me to easily document and run commands critical
     to the project.

2. Add pgvector vectorstore to the RAG system. Implementation and example usage
   from langchain docs found
   [here](https://python.langchain.com/v0.2/docs/integrations/vectorstores/pgvector/).

   Since I had chroma vectorstore setup. It was easy to replace it with pgvector
   when using langchain. Both can be initialized in a similar manner. Let's look
   at the examples.

   ```Python
     chroma_vectorstore = Chroma(
         client=persistent_client,
         collection_name="collection_name",
         embedding_function=embedding_function,
     )
   ```

   Pgvector:

   ```Python
     pgvector_vectorstore = PGVector(
       embeddings=embeddings,
       collection_name=collection_name,
       connection=connection,
       use_jsonb=True,
     )
   ```

   - To make my pgvector complete, I added Connection string to the `.env` file.
     The connection string is used to connect to the pgvector database. This
     connection string uses the details from the `docker-compose.yml` file under
     the `pgvector` service `environments` section.

3. I then added the pgvector vectorstore to the RAG system. The vectorstore can
   be found under the rag_system.py file.

## Generating evaluation Q&A pairs for RAGAS evaluation

To come up with q&a pairs that are diverse in complexity, I used RAGAS function
to generate q&a pairs. I followed the steps below to generate the q&a pairs:

1. Setting up LLMs:
   - OpenAI's gpt-3.5-turbo was initialized as the generator llm.
   - OpenAI's gpt-4o was initialized as the critic llm.
2. Determine the distributions in complexity of the q&a pairs.
   - Ragas provides three distributions that can be tweaked i.e `simple`,
     `multi-context`, and 'reasoning`.
   - I used the distributions:
     `simple: 0.5, multi-context: 0.25, and reasoning: 0.25`
3. Generate the q&a pairs.
   - To generate the evaluation set, I decided to use the
     `generate_with_langchain_docs` function from ragas, the `distributions`,
     `llms`, and `20` samples with `20` documents.

## Evaluation pipeline setup using RAGAS

I used [RAGAS](https://docs.ragas.io/en/latest/getstarted/evaluation.html) to
evaluate the RAG system.

At some point, I opted to use Langsmith to trace my evaluations and to store the
results. However, with Langsmith, many things were not clear. The documentation
on ragas website was empty. I therefore opted to build my own RAGAS pipeline and
save the results in csv files.

### Choice of RAGAS metrics for evaluation
I prioritized the following metrics for evaluation in that order:
1. Answer Correctness: How accurate the answer is compared to the ground truth.
2. Faithfulness: How well the answer aligns with the facts in the given context.
3. Answer Relevancy: How well the answer addresses the question asked.
4. Context Precision: How relevant the retrieved information is to the question.

More metrics and their explanations can be found on ragas documentation [here](https://docs.ragas.io/en/stable/concepts/metrics/index.html).


### Steps followed to setup the evaluation pipeline
1. Installed RAGAS using poetry.
2. I started with the simple setup from the RAGAS documentation
   [here](https://docs.ragas.io/en/latest/getstarted/evaluation.html).
3. Setup the utility functions to load the dataset and for evaluation using
   RAGAS+Langsmith, to upload csv to Langsmith. They can be found under
   `src/ragas_pipeline/ragas_utils.py`.
4. **Getting contexts and answers**: I then created a function to get the
   contexts and answers for the questions in the evaluation q&a pairs.

   - The function can be found under `src/ragas/ragas_pipeline.py`
   - It receieves the evaluation q&a pairs and the rag_chain and uses the
     rag_chain to get the contexts and answers.

5. **Evaluation pipeline**: I then created a function to run the evaluation
   pipeline. The function can be found under `src/ragas/ragas_pipeline.py`. This
   is what the function does:
   - You begin by defining key metrics like answer correctness, faithfulness,
     answer relevancy, and context precision, and then load the evaluation data.
   - Choosing Evaluation Method: You decide whether to evaluate using LangSmith
     or locally on your machine.
   - Using Langsmith: If using LangSmith, we ensure the dataset name is provided
     alongside the experiment name, upload the dataset if needed, and then
     evaluate the RAG chain on LangSmith, which will show the results on the
     LangSmith dashboard.
   - Evaluating Locally: If evaluating locally, the function
     `get_contexts_and_answers` is used which uses rag_chain as mentioned in the
     last step. It then evaluates process using ragas against the predefined
     metrics.
   - Converting and saving results: After evaluation, the results are converted
     into a pandas DataFrame. If saving locally, a directory is created if
     needed and the results saved as a CSV file.

## How to run a benchmark on the RAG system using RAGAS

Running the evaluation pipeline using ragas is fairly simple. Assuming we have
initialized the RAG system in this manner as seen in the section on RAG system
setup above:

```Python
  from src.rag_pipeline.rag_system import RAGSystem

  rag_system = RAGSystem(
    model_name = "gpt-4o",
    embeddings = embeddings,
    # Here we can add more parameters to customize the RAG system
  )

  rag_system.initialize()
```

We can then run the evaluation pipeline as follows, providing the `rag_chain`
initialized in the instance of RAGsystem above:

```Python
  from src.ragas.ragas_pipeline import run_ragas_evaluation

  rag_results = run_ragas_evaluation(
    rag_chain=rag_system.rag_chain,
    save_results=True,
    experiment_name="embedding_model_bge_large"
  )
```

The function will run the evaluation pipeline and save the results in a csv file
with the `experiment_name` being used to name the csv results file.

## The results of the baseline benchmark evaluation

The baseline benchmark evaluation was run using the RAG system with the following configurations:
- Model: GPT-4o
- Embeddings: OpenAIEmbeddings (text-embeddings-ada-002)
- Vectorstore: pgvector
- Chunking strategy: RecursiveCharacterTestSplitter, chunk_size=1000, overlap=200
- Ragchain - RetrievalQA with the default prompt

### Summary statistics
Since the metrics were all of type float64, I could carry out numerical calculations. i calculated the summary statistics i.e mean, standard deviation and creating visualizations to understand the performance of the RAG system.

Below is a boxplot of the summary statistics:

![baseline-benchmark-results](screenshots/results/baseline_benchmark_visualization.png)

Key observations from the summary statistics and boxplots:

- **Answer Correctness**: The average answer correctness is `0.689`, suggesting that the system generates reasonably accurate answers most of the time. However, there's a wide range `(0.23 to 1)`, indicating that the accuracy can vary significantly depending on the question. The standard deviation of 0.189 also supports this observation.

- **Faithfulness**: The system excels in faithfulness, with a high average score of `0.863` and `75%` of the values at the maximum of 1. This indicates that the generated answers are generally consistent with the provided context.

- **Answer Relevancy**: The average answer relevancy is 0.847, suggesting that the answers are mostly relevant to the questions. However, there are a few instances where the relevancy is 0, indicating that the system might sometimes generate irrelevant responses. The standard deviation of 0.292 also indicates a relatively wide range of relevancy scores.

- **Context Precision**: The system performs exceptionally well in context precision, with an average score of 0.98 and most values concentrated near 1. This suggests that the system is highly effective at retrieving relevant context for answering questions.



## Optimization techniques

### Using open source model for CrossEncoderReranking.

The embeddings I used were from the `sentence-transformers` library. I used
embeddings the model `sentence-transformers/msmarco-distilbert-dot-v5`

The drawbacks:

- The reranker was quite slow on average it used 22 seconds in retrieval as seen
  in the
  Langsmith![cross-encoder-reranking-opensource-model-langsmith-traces](screenshots/langsmith-tracing-opensource-rerankerScreenshot%20from%202024-07-30%2006-34-14.png)
- During the evaluation with ragas, the entire process took 15 minutes.

- My assumptions were that since the model is from huggingface and is running locally, therefore the it would use cpu to carry out the operations making it slow. I believe this can be improved by hosting the and using gpu.
- Another reason to support this is that I used, bge-raranker-base which is 1.1GB in size.
When I throttled the CPU this got slower, upto 110 seconds.

### 
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Contact

For any inquiries, please contact through email
[Hillary Kipkemoi](mailto:hillary6k@gmail.com)

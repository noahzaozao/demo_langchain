# demo_langchain

## Overview

This repository provides practical examples of using **LangChain** with the **Tongyi** language model. **LangChain** is a framework designed to simplify the development of applications powered by large language models (LLMs). It enables developers to easily connect language models to different data sources, interact with external systems, and build advanced applications.

The repository contains easy-to-follow examples demonstrating how to integrate **LangChain** with **Tongyi**, enabling developers to leverage LLM capabilities in their applications. The focus is on vector search, embeddings, and Retrieval-Augmented Generation (RAG).

## Key Features
- **Practical Examples**: Simple and clear examples of using **LangChain** with **Tongyi**.
- **Data Integration**: Demonstrates how to connect and work with data from various sources using **LangChain**.
- **Customization**: Explains how to customize **LangChain** to meet specific use cases.
- **RAG Support**: Shows how to integrate Retrieval-Augmented Generation (RAG) to enhance query responses with external data.

## Key Concepts

### Vector Search and Embeddings
**Vector search** is a technique used to find relevant information by understanding the meaning of a query, rather than just matching keywords. **Embeddings** are numerical representations of text (e.g., words, sentences, or paragraphs) that capture the meaning of the text. These embeddings are then used for comparison to find similar content.

In **LangChain**, vector search is used to search through large datasets, retrieving the most relevant information based on the meaning of the query. Embeddings are generated from text to convert it into vectors, which can then be compared to other vectors in the database.

### Retrieval-Augmented Generation (RAG)
RAG is a method that combines information retrieval and text generation. It allows the system to search for relevant information from external sources and use that data to generate more accurate and contextually relevant responses. 

The workflow for RAG involves:
1. **User Query**: A question or search request from the user.
2. **Retrieving Relevant Information**: The query is transformed into an embedding, which is used to search for similar content in an external data source.
3. **Text Generation**: The retrieved data is fed into a language model like **Tongyi**, which generates a response using both the model’s knowledge and the retrieved information.
4. **Return Response**: The model returns a response based on the search results and the query.

RAG enhances the model’s performance by combining the power of retrieval and generation, making it capable of handling more complex queries and providing more accurate results.

## Workflow Overview

1. **User Input**: The process begins when the user provides a query.
2. **Embeddings Creation**: The query is transformed into an embedding (a vector) to capture the meaning of the text.
3. **Vector Search**: The system performs a vector search to find the most relevant content based on the query’s meaning.
4. **Retrieve and Process Results**: The system retrieves the relevant information and may use **Tongyi** or another LLM to process the data and generate a response.
5. **Return Response**: The final response is generated and returned to the user.

This process allows applications to understand the meaning behind the user's query, retrieve relevant information, and provide more contextually accurate answers.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/noahzaozao/demo_langchain.git
    cd demo_langchain
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your **Tongyi** API key and any necessary configurations for external data sources.

4. Run the examples:
    ```bash
    python example_tongyi.py
    ```

## Examples

- **Example 1**: Basic usage with LangChain and **Tongyi**.
    - Demonstrates how to integrate **Tongyi** with **LangChain** for simple text generation and basic queries.
    
- **Example 2**: Chat history with LangChain and **Tongyi**.
    - Shows how to maintain chat history and use it in conversations with **Tongyi** to simulate continuous dialogue.

- **Example 3**: RAG with LangChain and **Tongyi**.
    - Demonstrates the **Retrieval-Augmented Generation (RAG)** approach by integrating **Tongyi** with external data sources for more accurate and contextually relevant responses.

- **Example 4**: RAG and Chat history with LangChain and **Tongyi**.
    - Combines **RAG** with chat history for generating responses based on past interactions while retrieving external information to improve accuracy.

- **Example 5**: Basic usage with LangChain and **Ollama**.
    - Shows how to integrate **Ollama** with **LangChain** for basic text generation tasks.

- **Example 6**: Chat history with LangChain and **Ollama**.
    - Demonstrates using **Ollama** with **LangChain** while maintaining chat history for simulating ongoing conversations.

- **Example 7**: RAG with LangChain and **Ollama**.
    - Implements **RAG** with **Ollama**, showing how to augment the model's responses by retrieving relevant data from external sources.

- **Example 8**: RAG and Chat history with LangChain and **Ollama**.
    - Combines **RAG** and chat history with **Ollama** to generate responses that take both past interactions and external data into account.

- **Example 9**: HTTP service with LangChain and **Tongyi**.
    - Implements an HTTP service using **LangChain** and **Tongyi**, allowing users to interact with the model through a web interface.

- **Example 10**: HTTP service with LangChain and **Ollama**.
    - Similar to Example 9, but using **Ollama** for generating responses in a web service environment powered by **LangChain**.



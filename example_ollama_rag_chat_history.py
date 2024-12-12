import os
import bs4
import config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

# Set up the directory where the vector store data will persist
persist_dir = 'ollama_chroma_data_dir'

# Initialize a vector store for embeddings using Ollama's Qwen2:7b model
vector_store = Chroma(
    embedding_function=OllamaEmbeddings(model="qwen2:7b"),
    persist_directory=persist_dir
)

# Create a retriever from the vector store for retrieving relevant documents
retriever = vector_store.as_retriever()

# Define the system-level prompt for the assistant to use retrieved context in answers
system_prompt = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't find the answer, say that you
don't know. Use three sentences maximum and keep the answer concise. \n

{context}
"""

# Create a chat prompt template for the assistant
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder('chat_history'),  # Placeholder for chat history
    ('human', '{input}')
])

# Initialize the Ollama chat model
model = ChatOllama(
    model='qwen2:7b'
)

# Create a chain to process documents using the model and prompt
chain1 = create_stuff_documents_chain(model, prompt)

# Define a prompt to contextualize user questions based on chat history
contextualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. DO NOT answer the question,
just reformulate it if needed and otherwise return it as is.
"""

# Create a prompt template for handling historical context in questions
retriever_history_temp = ChatPromptTemplate.from_messages([
    ('system', contextualize_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

# Create a chain to integrate historical context in retrieval
history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

# Store chat histories per session
store = {}

# Function to get chat history for a given session ID
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create a retrieval chain combining historical context and document-based responses
chain = create_retrieval_chain(history_chain, chain1)

# Combine retrieval chain with message history for session-based interactions
result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

config = {'configurable': {'session_id': 'nw123'}}

# Test the chain with a question and specific session ID
resp1 = result_chain.invoke(
    {'input': 'What is Task Decomposition?',},  # User input
    config  # Session-specific config
)

print(resp1['answer'])  # Print the response to the first question

# Test the chain with a follow-up question in the same session
resp2 = result_chain.invoke(
    {'input': 'What are common ways of doing it?',},  # User input
    config  # Same session ID
)

print(resp2['answer'])  # Print the response to the follow-up question

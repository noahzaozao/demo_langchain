import os
import bs4
import config
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

# Directory to store Chroma vector data
persist_dir = 'chroma_data_dir'

# Initialize the vector store with DashScope embeddings and specify the persistence directory
vector_store = Chroma(
    embedding_function=DashScopeEmbeddings(),
    persist_directory=persist_dir
)

# Set up a retriever to fetch relevant documents based on queries
retriever = vector_store.as_retriever()

# Define the system prompt for question-answering
system_prompt = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't find the answer, say that you
don't know. Use three sentences maximum and keep the answer concise. \n

{context}
"""

# Create a chat prompt template for processing user input
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),  # System-level instructions for the assistant
    MessagesPlaceholder('chat_history'),  # Placeholder for chat history
    ('human', '{input}')  # Placeholder for user input
])

# Initialize the language model (Tongyi) with a temperature setting
model = Tongyi(temperature=1)

# Create a chain for combining document retrieval and answering
chain1 = create_stuff_documents_chain(model, prompt)

# Define a prompt for contextualizing user questions based on chat history
contexttualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. DO NOT answer the question,
just reformulate it if needed and otherwise return it as is.
"""

# Create a prompt template for the history-aware retriever
retriever_history_temp = ChatPromptTemplate.from_messages([
    ('system', contexttualize_q_system_prompt),  # System-level instructions for question reformulation
    MessagesPlaceholder('chat_history'),  # Placeholder for chat history
    ('human', '{input}')  # Placeholder for user input
])

# Create a history-aware retriever chain that combines retrieval with history handling
history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

# Dictionary to store session-specific chat histories
store = {}

# Function to fetch or initialize the chat history for a given session
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # Initialize chat history for new sessions
    return store[session_id]

# Combine the history-aware retriever and answering chain into a retrieval chain
chain = create_retrieval_chain(history_chain, chain1)

# Wrap the chain with a handler for maintaining chat history
result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Function to manage session-specific histories
    input_messages_key='input',  # Key for user input in the chain
    history_messages_key='chat_history',  # Key for chat history in the chain
    output_messages_key='answer'  # Key for storing the final response
)

config = {'configurable': {'session_id': 'nw123566'}}

# Example query 1: Ask about task decomposition
resp1 = result_chain.invoke(
    {'input': 'What is Task Decomposition?',},  # User input
    config  # Session-specific configuration
)

# Print the response for the first query
print(resp1['answer'])

# Example query 2: Ask about common methods of task decomposition
resp2 = result_chain.invoke(
    {'input': 'What are common ways of doing it?',},  # User input
    config  # Session-specific configuration
)

# Print the response for the second query
print(resp2['answer'])

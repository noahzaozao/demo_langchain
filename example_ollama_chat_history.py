import os
import config
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder

# Define a dictionary to act as a store for session histories.
store = {}

# Function to retrieve or initialize session history for a given session ID.
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Initialize the ChatOllama model.
model = ChatOllama(
    # Specify the model to use (commented out example: "llama3.2").
    model="qwen2:7b"
)

# Define a prompt template for the conversation.
prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Please use {language} to answer the question.'),  # System message to set the assistant's role.
    MessagesPlaceholder(variable_name='my_msg'),  # Placeholder for user messages.
])

# Combine the prompt template and model into a chain.
chain = prompt_template | model

# Create a runnable with message history handling.
do_message = RunnableWithMessageHistory(
    chain,  # Chain containing the prompt and model.
    get_session_history,  # Function to retrieve session history.
    input_messages_key='my_msg'  # Key to identify user messages in input.
)

# Configuration dictionary with session details.
config = {'configurable': {'session_id': 'nw123'}}

# Step 1: Invoke the chain with a new message and language setting.
resp = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='Hi, my name is Noah.')],  # Human message content.
        'language': 'English',  # Specify the language for the response.
    },
    config  # Pass the configuration including session ID.
)
print(resp.content)  # Print the response content.

# Step 2: Invoke the chain again to test session memory.
resp2 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='What is my name?')],  # Human message content.
        'language': 'English',  # Language setting.
    },
    config  # Use the same session configuration.
)
print(resp2.content)  # Print the response content.

# Step 3: Stream a response using the `stream` method.
for res in do_message.stream(
    {
        'my_msg': [HumanMessage(content='Tell me a joke.')],  # Human message requesting a joke.
        'language': 'English',  # Language for the response.
    },
    config  # Same session configuration.
):
    print(res.content, end='-')  # Print each streamed response part.

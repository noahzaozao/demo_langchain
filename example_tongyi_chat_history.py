import os
import config
from langchain_community.llms import Tongyi  # Import the Tongyi LLM from the LangChain community.
from langchain_community.chat_message_histories import ChatMessageHistory  # To maintain chat history.
from langchain.prompts import ChatPromptTemplate  # For creating prompt templates.
from langchain_core.runnables.history import RunnableWithMessageHistory  # To manage message history with the chain.
from langchain_core.messages import HumanMessage  # Represents a message from a human.
from langchain_core.prompts import MessagesPlaceholder  # Placeholder for dynamic content in prompts.

# In-memory store for session-based message history.
store = {}

# Function to retrieve or initialize chat history for a given session ID.
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # Initialize if session ID not in store.
    return store[session_id]

# Initialize the Tongyi model with a specific temperature for response creativity.
model = Tongyi(temperature=1)

# Define the prompt template for the chatbot.
prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Please use {language} to answer the question.'),  # System message to set assistant behavior.
    MessagesPlaceholder(variable_name='my_msg'),  # Placeholder for dynamically adding human messages.
])

# Create a processing chain by combining the prompt template with the model.
chain = prompt_template | model

# Combine the processing chain with session-based message history.
do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Function to retrieve message history.
    input_messages_key='my_msg'  # Key to identify human messages in input.
)

# Configuration settings for the session (e.g., session ID).
config = {'configurable': {'session_id': 'nw123'}}

# Step 1: First interaction with the chatbot.
resp = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='Hi, my name is Noah.')],  # Human message for the assistant.
        'language': 'English',  # Language to use in the response.
    },
    config  # Session configuration.
)
print(resp)  # Print the response from the assistant.

# Step 2: Follow-up question in the same session.
resp2 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='What is my name?')],  # Question based on the previous message.
        'language': 'English',
    },
    config
)
print(resp2)  # Print the response from the assistant.

# Step 3: Streaming responses for a different query.
for res in do_message.stream(
    {
        'my_msg': [HumanMessage(content='Tell me a joke.')],  # Human message asking for a joke.
        'language': 'English',
    },
    config
):
    print(res, end='-')  # Print each part of the streamed response without a newline.

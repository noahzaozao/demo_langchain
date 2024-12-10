import os  # Importing the os module for potential environment variable management
import config  # Importing a config module, which could contain environment settings or credentials
from langchain.prompts import PromptTemplate  # Importing PromptTemplate from langchain for creating structured prompts
from langchain_ollama import ChatOllama  # Importing ChatOllama to use the Ollama model for text generation
from fastapi import FastAPI
from langserve import add_routes

# The GitHub issue reference is likely included for context or debugging purposes
# https://github.com/langchain-ai/langchain/issues/28607

# Initialize the ChatOllama model with a specific configuration
model = ChatOllama(
    # Uncomment this line to use the "llama3.2" model if needed
    # model="llama3.2",
    model="qwen2:7b"  # Specifies the "qwen2:7b" model
)

# Define a template for constructing prompts
template = """
You are a helpful assistant.
Please use {language} to answer 
the following question: {question}
"""

# Create a PromptTemplate object with the defined template and input variables
prompt_template = PromptTemplate(
    template=template,
    input_variables=['language', 'question']
)

# Use the pipe operator to chain the prompt template with the model
chain = prompt_template | model

# Initialize a FastAPI application instance.
app = FastAPI()

# Add routes to the FastAPI app.
# This creates an endpoint (`/chain_demo`) to expose the processing chain as a web service.
add_routes(
    app,
    chain,
    path='/chain_demo'
)

# The entry point of the application.
# Use Uvicorn to run the FastAPI application on the specified host and port.
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

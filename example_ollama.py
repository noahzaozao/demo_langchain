import os  # Importing the os module for potential environment variable management
import config  # Importing a config module, which could contain environment settings or credentials
from langchain.prompts import PromptTemplate  # Importing PromptTemplate from langchain for creating structured prompts
from langchain_ollama import ChatOllama  # Importing ChatOllama to use the Ollama model for text generation

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

# Define the language variable to specify the language for the response
language = 'English'
# Uncomment to use other languages like French or Chinese
# language = 'French'
# language = 'Chinese'

# Define the question to be answered by the model
question = 'Who are you?'
# Uncomment to use other questions
# question = 'How to make scrambled eggs with tomatoes?'
# question = 'How to use Python Django?'

# Use the chain to invoke the model with the specified inputs
response = chain.invoke({
    'language': language,
    'question': question
})

# Print the results, including the language, question, and response content
print(f'language: {language} \nquestion: {question} \n{response.content}')

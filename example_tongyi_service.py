import os
import config
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from fastapi import FastAPI
from langserve import add_routes

# Initialize the Tongyi language model with specified parameters.
# `temperature` controls the randomness of the output.
# A higher value for `temperature` results in more diverse and creative outputs.
model = Tongyi(
    temperature=1,  # Set the randomness of the output to a high level.
    # model="qwen-plus"  # Uncomment this line to use the default "qwen-plus" model.
    # model='qwen-max'  # Uncomment this line to use the "qwen-max" model for extended capabilities.
)

# Define the prompt template to structure the input for the model.
# This template allows injecting specific variables (`language` and `question`) dynamically.
template = """
You are a helpful assistant.
Please use {language} to answer 
the following question: {question}
"""

# Create a PromptTemplate instance with the defined template.
# The `input_variables` parameter specifies the placeholders to be replaced dynamically.
prompt_template = PromptTemplate(
    template=template,
    input_variables=['language', 'question']
)

# Define a processing chain.
# The chain takes the prompt template's output and passes it as input to the Tongyi model.
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
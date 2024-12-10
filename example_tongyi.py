import os
import config
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi

# Initialize the Tongyi language model with specified parameters.
model = Tongyi(
    temperature=1,  # Control the randomness of the output; higher values increase randomness.
    # model="qwen-plus"  # Uncomment to use the default "qwen-plus" model.
    # model='qwen-max'  # Uncomment to use the "qwen-max" model for extended capabilities.
)

# Define the prompt template to structure the input for the model.
template = """
You are a helpful assistant.
Please use {language} to answer 
the following question: {question}
"""

# Create a prompt template instance, specifying the input variables to be replaced.
prompt_template = PromptTemplate(
    template=template,
    input_variables=['language', 'question']
)

# Define a processing chain where the prompt template output is passed to the model.
chain = prompt_template | model

# Set the desired language for the response.
language = 'English'
# language = 'French'  # Uncomment for a French response.
# language = 'Chinese'  # Uncomment for a Chinese response.

# Define the question to be answered by the model.
question = 'Who are you?'
# question = 'How to make scrambled eggs with tomatoes?'  # Uncomment for a cooking question.
# question = 'How to use Python Django?'  # Uncomment for a Django-related question.

# Execute the chain by passing the input variables.
response = chain.invoke({
    'language': language,
    'question': question
})

# Print the input language, question, and model's response.
print(f'language: {language} \nquestion: {question} \n{response}')

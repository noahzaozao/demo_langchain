import os
import config_bianxie
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# https://api.bianxie.ai/log

model = ChatOpenAI(model="gpt-3.5-turbo")
template = """
You are a helpful assistant.
Please use {language} to answer 
the following question: {question}
"""
prompt_template = PromptTemplate(
    template=template,
    input_variables=['language', 'question']
)
chain = prompt_template | model

language = 'English'
# language = 'French'  # Uncomment for a French response.
# language = 'Chinese'  # Uncomment for a Chinese response.

# Define the question to be answered by the model.
question = 'Who are you?'
# question = 'How to make scrambled eggs with tomatoes?'  # Uncomment for a cooking question.
# question = 'How to use Python Django?'  # Uncomment for a Django-related question.

response = chain.invoke({
    'language': language,
    'question': question
})

print(f'language: {language} \nquestion: {question} \n{response}')

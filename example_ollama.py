import os
import config
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama


# https://github.com/langchain-ai/langchain/issues/28607


model = ChatOllama(
    # model="llama3.2",
    model="qwen2:7b"
)

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
# # language = 'French'
# # language = 'Chinese'

question = 'Who are you?'
# # question = 'How to make scrambled eggs with tomatoes?'
# # question = 'How to use Python Django?'

response = chain.invoke({
    'language': language,
    'question': question
})

print(f'language: {language} \nquestion: {question} \n{response.content}')

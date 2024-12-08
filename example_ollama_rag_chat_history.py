import os
import bs4
import config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_chroma.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder


# https://github.com/langchain-ai/langchain/issues/28607


persist_dir = 'ollama_chroma_data_dir'
vector_store = Chroma(
    embedding_function=OllamaEmbeddings(model="qwen2:7b"),
    persist_directory=persist_dir
)
retriever = vector_store.as_retriever()
system_prompt = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't find the answer, say that you
don't know. Use three sentences maximum and keep the answer concise. \n

{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder('chat_history'),  # chat history
    ('human', '{input}')
])

model = ChatOllama(
    model='qwen2:7b'
)

chain1 = create_stuff_documents_chain(model, prompt)

contexttualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. DO NOT answer the question,
just reformulate it if needed and otherwise return it as is.
"""

retriever_history_temp = ChatPromptTemplate.from_messages([
    ('system', contexttualize_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain = create_retrieval_chain(history_chain, chain1)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

resp1 = result_chain.invoke(
    {'input': 'What is Task Decomposition?',},
    config={'configurable': {'session_id': 'nw123566'}}
)

print(resp1['answer'])


resp2 = result_chain.invoke(
    {'input': 'What are common ways of doing it?',},
    config={'configurable': {'session_id': 'nw123566'}}
)

print(resp2['answer'])

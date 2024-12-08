import os
import bs4
import config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# https://github.com/langchain-ai/langchain/issues/28607


def create_vector_store():
    loader = WebBaseLoader(
        web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent'],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=(
                'post-header', 'post-title', 'post-content'
            ))
        )
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=4)
    splits = splitter.split_documents(docs)
    persist_dir = 'ollama_chroma_data_dir'
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="qwen2:7b"),
        persist_directory=persist_dir
    )


def load_vector_store():
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
        ('human', '{input}')
    ])

    model = ChatOllama(
        model='qwen2:7b'
    )

    chain1 = create_stuff_documents_chain(model, prompt)
    chain2 = create_retrieval_chain(retriever, chain1)

    resp = chain2.invoke({
        'input': 'What is Task Decomposition?'
    })
    print(resp)


# create_vector_store()
load_vector_store()

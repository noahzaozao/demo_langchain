import os
import config
from langchain_community.llms import Tongyi
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder


store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


model = Tongyi(temperature=1)

prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Please use {language} to answer the question.'),
    MessagesPlaceholder(variable_name='my_msg'),    
])

chain = prompt_template | model

do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'
)

config = {'configurable': {'session_id': 'nw123'}}

# Step 1
resp = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='Hi, my name is Noah.')],
        'language': 'English',
    },
    config
)
print(resp)

# Step 2
resp2 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='What is my name?')],
        'language': 'English',
    },
    config
)
print(resp2)


# Step 3
for res in do_message.stream(
    {
        'my_msg': [HumanMessage(content='Tell me a joke.')],
        'language': 'English',
    },
    config
):
    print(res, end='-')

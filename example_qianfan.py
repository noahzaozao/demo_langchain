import os
import config
from langchain_community.llms import QianfanLLMEndpoint
from langchain_core.messages import HumanMessage, SystemMessage

model = QianfanLLMEndpoint()

res = model.invoke("write a funny joke")
print(res)
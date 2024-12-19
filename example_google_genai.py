import os
import config_google
from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
res = llm.invoke("Write me a ballad about LangChain")
print(res)
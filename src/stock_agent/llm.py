import os

from langchain_openai import ChatOpenAI
from .config import settings

print("MODEL NAME:",settings.model_name)

llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.nine_router_api_key,
    base_url=settings.nine_router_base_url,
    temperature=0,
)
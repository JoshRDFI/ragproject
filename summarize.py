from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage
import os

llm = ChatOllama(model=os.getenv("MODEL_NAME"))

def summarize_chat(history: list[str]) -> str:
    # join history into a single string
    history_str = "\n".join(history)
    prompt = f"Summarize the following chat history in 3-5 sentences:\n\n{history_str}"
    summary = llm([HumanMessage(content=prompt)])
    return summary.content

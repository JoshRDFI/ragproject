from langchain_ollama import OllamaLLM as Ollama
import os
from datetime import datetime

llm = Ollama(model=os.getenv("MODEL_NAME"))

def summarize_chat(history: list[str]) -> str:
    """Summarize chat history with timestamp and topic extraction"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    history_text = "\n".join(history)

    prompt = f"""Please analyze the following conversation and provide:
1. A 3-5 sentence summary of the main discussion
2. Key topics discussed (list 2-3 main topics)
3. Any important decisions or conclusions reached

Conversation:
{history_text}

Format your response as:
TIMESTAMP: {timestamp}
TOPICS: [list the main topics separated by commas]
SUMMARY: [your 3-5 sentence summary]
CONCLUSIONS: [any key decisions or outcomes, or "None" if no specific conclusions]
"""

    response = llm.invoke(prompt)
    return response if isinstance(response, str) else str(response)

def extract_topic_from_input(user_input: str) -> str:
    """Extract the main topic from user input for better retrieval"""
    prompt = f"""Extract the main topic or subject from this user input in 1-3 words:

User input: {user_input}

Respond with only the topic/subject (e.g., "programming", "cooking recipe", "travel planning"):"""

    response = llm.invoke(prompt)
    return response.strip() if isinstance(response, str) else str(response).strip()
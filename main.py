from summarize import summarize_chat
from memory_store import create_vector_store, add_to_memory, retrieve_memory
import readline  # enables history in CLI
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOllama(model=os.getenv("MODEL_NAME"))

def build_prompt(context: str, recent: list[str]) -> str:
    return f"Context from memory:\n{context}\n\nRecent chat:\n" + "\n".join(recent)

def main():
    vector_store = create_vector_store()
    full_history = []
    recent_chat = []

    print("🧠 LLM with RAG memory. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        full_history.append(f"User: {user_input}")
        recent_chat.append(f"User: {user_input}")

        # Step 1: retrieve memory
        retrieved = retrieve_memory(vector_store, user_input)
        memory_context = "\n".join([d.page_content for d in retrieved]) if retrieved else "None"

        # Step 2: build prompt
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=build_prompt(memory_context, recent_chat))
        ]

        response = llm(messages).content
        print(f"AI: {response}")

        full_history.append(f"AI: {response}")
        recent_chat.append(f"AI: {response}")

        # Step 3: summarize occasionally
        if len(full_history) % 6 == 0:  # summarize every 3 turns
            summary = summarize_chat(full_history)
            add_to_memory(vector_store, summary)
            recent_chat = []  # reset buffer

if __name__ == "__main__":
    main()

import readline
import os
from dotenv import load_dotenv
from memory_store import get_or_create_vector_store, add_to_memory, hybrid_search
from summarize import summarize_chat
from langchain_memory import EnhancedRAGMemory

load_dotenv()

def build_prompt(vector_context: str, enhanced_context: str, recent: list[str]) -> str:
    prompt = """You are a helpful assistant.\n\n"""
    if vector_context:
        prompt += f"Relevant memory:\n{vector_context}\n\n"
    if enhanced_context:
        prompt += f"Conversation summary:\n{enhanced_context}\n\n"
    if recent:
        prompt += "Recent chat:\n" + "\n".join(recent) + "\n\n"
    prompt += "Answer the user's question based on the above."
    return prompt

def main():
    print("--- RAG Chatbot with Enhanced Memory ---")
    print("Type 'exit' to quit.\n")
    topic = input("Enter topic for this session (default: 'general'): ").strip() or "general"
    vector_store = get_or_create_vector_store(topic)
    memory_system = EnhancedRAGMemory()
    full_history = []
    recent_chat = []
    turn = 0
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Exiting chat.")
            break
        # Hybrid search from vector DB
        retrieved = hybrid_search(vector_store, user_input)
        vector_context = "\n".join([d.page_content for d in retrieved]) if retrieved else ""
        # Get enhanced memory context (summary)
        enhanced_context = memory_system.summary_memory.buffer_as_str if hasattr(memory_system.summary_memory, 'buffer_as_str') else ""
        # Build prompt
        prompt = build_prompt(vector_context, enhanced_context, recent_chat)
        # Get response from EnhancedRAGMemory (uses its own chain)
        response = memory_system.chat(user_input)
        print(f"Bot: {response}\n")
        # Update histories
        full_history.append(f"User: {user_input}\nAssistant: {response}")
        recent_chat.append(f"User: {user_input}\nAssistant: {response}")
        # Periodically summarize and store in vector DB
        turn += 1
        if turn % 3 == 0:
            summary = summarize_chat(full_history)
            add_to_memory(vector_store, summary, metadata={"topic": topic})
            recent_chat = []

if __name__ == "__main__":
    main()

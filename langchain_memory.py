# Enhanced LangChain Memory Integration Example
# This demonstrates how to scale the RAG chatbot with LangChain memory wrappers

from langchain.memory import ConversationSummaryBufferMemory, VectorStoreRetrieverMemory
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from memory_store import get_or_create_vector_store
import os
from dotenv import load_dotenv

load_dotenv()

class EnhancedRAGMemory:
    """Enhanced memory system combining LangChain memory with custom RAG"""
    
    def __init__(self, model_name: str = None):
        self.llm = Ollama(model=model_name or os.getenv("MODEL_NAME"))
        self.vector_store = get_or_create_vector_store()
        
        # LangChain's conversation summary buffer memory
        self.summary_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )
        
        # Vector store retriever memory
        self.vector_memory = VectorStoreRetrieverMemory(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory_key="vector_context"
        )
        
        # Create conversation chain with combined memory
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.summary_memory,
            verbose=True
        )
    
    def chat(self, user_input: str) -> str:
        """Enhanced chat with multiple memory systems"""
        # Get vector context
        vector_context = self.vector_memory.load_memory_variables({"input": user_input})
        
        # Combine with conversation memory
        response = self.conversation.predict(
            input=f"Context: {vector_context.get('vector_context', '')}\n\nUser: {user_input}"
        )
        
        # Save to vector memory
        self.vector_memory.save_context({"input": user_input}, {"output": response})
        
        return response

# Usage example:
# memory_system = EnhancedRAGMemory()
# response = memory_system.chat("Hello, how are you?")

if __name__ == "__main__":
    print("--- Enhanced RAG Memory Example ---")
    memory_system = EnhancedRAGMemory()
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Exiting chat.")
            break
        response = memory_system.chat(user_input)
        print(f"Bot: {response}\n")
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage

from memory_store import create_vector_store, add_to_memory, retrieve_memory
from summarize import summarize_chat
from main import get_response, build_prompt

load_dotenv()

st.set_page_config(page_title="Local RAG Chatbot", page_icon="🧠")
st.title("🧠 Local RAG Chatbot")

# Sidebar for system prompt
with st.sidebar:
    st.header("Configuration")
    system_prompt = st.text_area("System Prompt", "You are a helpful assistant.", height=150)

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = create_vector_store()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_history" not in st.session_state:
    st.session_state.full_history = []
if "recent_chat" not in st.session_state:
    st.session_state.recent_chat = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Retrieve memory
        retrieved = retrieve_memory(st.session_state.vector_store, prompt)
        memory_context = "\n".join([d.page_content for d in retrieved]) if retrieved else "None"

        # Append to histories
        st.session_state.full_history.append(f"User: {prompt}")
        st.session_state.recent_chat.append(f"User: {prompt}")

        # Build prompt and get response
        final_prompt = build_prompt(memory_context, st.session_state.recent_chat)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=final_prompt)
        ]
        
        response = get_response(messages)
        
        message_placeholder.markdown(response)
        
        # Append AI response to histories
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.full_history.append(f"AI: {response}")
        st.session_state.recent_chat.append(f"AI: {response}")

        # Summarize and add to memory if needed
        if len(st.session_state.full_history) > 0 and len(st.session_state.full_history) % 6 == 0:
            summary = summarize_chat(st.session_state.full_history)
            add_to_memory(st.session_state.vector_store, summary)
            st.session_state.recent_chat = []  # Reset recent chat
            st.sidebar.success(f"Just summarized and added to memory:\n\n{summary}")


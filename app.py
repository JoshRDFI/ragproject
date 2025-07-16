import streamlit as st
from memory_store import get_or_create_vector_store, add_to_memory, hybrid_search
from summarize import summarize_chat
from langchain_memory import EnhancedRAGMemory

st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖")
st.title("Local RAG Chatbot with Enhanced Memory")

if "topic" not in st.session_state:
    st.session_state.topic = "general"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_or_create_vector_store(st.session_state.topic)
if "memory_system" not in st.session_state:
    st.session_state.memory_system = EnhancedRAGMemory()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_history" not in st.session_state:
    st.session_state.full_history = []
if "recent_chat" not in st.session_state:
    st.session_state.recent_chat = []
if "turn" not in st.session_state:
    st.session_state.turn = 0

st.sidebar.title("Configuration")
topic = st.sidebar.text_input("Topic", st.session_state.topic)
if topic != st.session_state.topic:
    st.session_state.topic = topic
    st.session_state.vector_store = get_or_create_vector_store(topic)

system_prompt = st.sidebar.text_area("System Prompt", "You are a helpful assistant.", height=150)

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.recent_chat.append(f"User: {prompt}")

    # Hybrid search from vector DB
    retrieved = hybrid_search(st.session_state.vector_store, prompt)
    vector_context = "\n".join([d.page_content for d in retrieved]) if retrieved else ""
    # Get enhanced memory context (summary)
    enhanced_context = st.session_state.memory_system.summary_memory.buffer_as_str if hasattr(st.session_state.memory_system.summary_memory, 'buffer_as_str') else ""
    # Build prompt
    final_prompt = build_prompt(vector_context, enhanced_context, st.session_state.recent_chat)
    # Get response from EnhancedRAGMemory (uses its own chain)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = st.session_state.memory_system.chat(prompt)
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.full_history.append(f"User: {prompt}\nAssistant: {response}")
    st.session_state.recent_chat.append(f"Assistant: {response}")
    st.session_state.turn += 1
    # Periodically summarize and store in vector DB
    if st.session_state.turn % 3 == 0:
        summary = summarize_chat(st.session_state.full_history)
        add_to_memory(st.session_state.vector_store, summary, metadata={"topic": st.session_state.topic})
        st.session_state.recent_chat = []
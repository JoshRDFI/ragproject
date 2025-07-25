# Local RAG Chatbot with Ollama

This project demonstrates a simple Retrieval-Augmented Generation (RAG) chatbot that runs entirely on your local machine using Ollama, LangChain, and FAISS for vector storage. It maintains a long-term memory of the conversation by summarizing it periodically and retrieving relevant parts of the summary to provide context for new responses.

This chatbot can be run as a command-line application or as a web-based interface using Streamlit.

## Features

*   **Persistent Memory**: Conversations are saved and reloaded across sessions, ensuring the chatbot remembers past interactions.
*   **Structured Summaries**: Summaries of conversations include timestamps, topics, and key conclusions for better context and searchability.
*   **Multiple Vector Stores**: The chatbot can manage separate vector stores for different topics, allowing for more organized and efficient memory retrieval.
*   **Advanced Retrieval**: A hybrid search approach combining semantic and keyword search is implemented for more accurate retrieval of memories.
*   **Memory Compression**: Hierarchical summarization is used to compress older memories, keeping the long-term memory concise and relevant.
*   **LangChain Integration**: Built with LangChain for robust and scalable memory management and agent integration.
*   **Local First**: Runs entirely on your local machine with Ollama, ensuring privacy and control over your data.

## How It Works

The chatbot's memory and response generation process follows these steps:

1.  **User Input**: The application takes user input from either the command line or a web interface.
2.  **Memory Retrieval**: The user's input is used to query multiple FAISS vector stores using a hybrid search (semantic + keyword). The stores contain structured summaries of past parts of the conversation. The most relevant summaries are retrieved using a dedicated embedding model.
3.  **Prompt Construction**: A prompt is constructed for the language model. This prompt includes an optional system prompt, the retrieved summaries (long-term memory), and the most recent conversation turns (short-term memory).
4.  **Response Generation**: The constructed prompt is sent to a local language model served by Ollama (e.g., Qwen3, Llama 2, Mistral) to generate a response.
5.  **Memory Consolidation**: After every few turns (currently 3), the conversation history is summarized by the language model. The summary is enriched with a timestamp, extracted topics, and key conclusions.
6.  **Memory Storage**: The generated summary is converted into a vector embedding and stored in the appropriate FAISS vector store based on its topic for future retrieval. The vector stores are persisted to disk.
7.  **Chat History**: The full conversation history is tracked, while the recent chat history buffer is cleared after each summarization to manage context length.

## Setup and Running

### Prerequisites

*   [Python 3.8+](https://www.python.org/downloads/)
*   [Ollama](https://ollama.com/) installed and running.
*   **Note on GPU Acceleration**: For GPU acceleration with Ollama, it is recommended to run the application and the Python virtual environment under Linux. The Windows version of Ollama does not currently support GPU acceleration.
*   Two models pulled via Ollama:
    *   A chat model (e.g., `ollama pull qwen2.5`)
    *   An embedding model (e.g., `ollama pull nomic-embed-text`)

### Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JoshRDFI/ragproject
    cd <repository-folder -- where you saved it>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the models:**
    Create a `.env` file in the root of the project and add the names of both models you want to use:
    ```
    MODEL_NAME=qwen2.5
    EMBEDDING_MODEL_NAME=nomic-embed-text
    ```
    Replace `qwen2.5` with the name of any chat model you have available in Ollama, and `nomic-embed-text` with your preferred embedding model.

5.  **Run the chatbot:**

    You can run the chatbot in two ways:

    **A) As a web application (using Streamlit):**
    ```bash
    streamlit run app.py
    ```
    This will open a new tab in your browser with the chat interface. You can configure the system prompt from the sidebar.

    **B) As a command-line application:**
    ```bash
    python main.py
    ```

## Project Structure

```
.
├── .env                # Environment variables for configuration (you need to create this)
├── app.py              # Streamlit application entry point.
├── main.py             # Command-line application entry point.
├── memory_store.py     # Manages the FAISS vector store for conversation summaries.
├── summarize.py        # Contains the logic for summarizing the chat history.
├── langchain_memory.py # Demonstrates advanced memory management with LangChain.
├── requirements.txt    # Lists the Python dependencies for the project.
├── structure.txt       # Describes the project structure.
├── vector_stores/      # Directory to store the persisted FAISS vector stores.
└── README.md           # This file.
```

## Model Configuration

This project uses two separate Ollama models:

*   **Chat Model** (`MODEL_NAME`): Used for generating responses and summarizing chat history. Examples: `qwen2.5`, `llama2`, `mistral`.
*   **Embedding Model** (`EMBEDDING_MODEL_NAME`): Used for creating vector embeddings of chat summaries for retrieval. Recommended: `nomic-embed-text`.

Using separate models allows for optimal performance - chat models excel at text generation while dedicated embedding models are more efficient for similarity search and retrieval tasks.

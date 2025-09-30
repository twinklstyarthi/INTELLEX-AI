import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import tempfile
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

CHAT_HISTORY_DIR = "chat_history"
FAISS_INDEX_PATH = "vector_store" 

def load_css(file_name):
    """Loads a CSS file into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def save_chat_history():
    """Saves the current chat history to a JSON file."""
    if 'current_chat_id' not in st.session_state:
        return


    if not st.session_state.messages:
        
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{st.session_state.current_chat_id}.json")
        if not os.path.exists(file_path):
             with open(file_path, 'w') as f:
                json.dump([], f)
        return

    chat_id = st.session_state.current_chat_id
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    with open(file_path, 'w') as f:
        json.dump(st.session_state.messages, f, indent=4)


def load_chat_history(chat_id):
    """Loads a chat history from a JSON file."""
    
    st.session_state.uploader_key += 1
    
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            messages = json.load(f)
        
        st.session_state.messages = messages
        st.session_state.current_chat_id = chat_id
        st.session_state.chat_engine = None 
        st.rerun() 
    else:
        st.error("Chat history not found.")

def get_chat_display_name(chat_id):
    """Generates a display name for a chat session."""
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    try:
        with open(file_path, 'r') as f:
            messages = json.load(f)
        
        first_user_query = next((msg['content'] for msg in messages if msg['role'] == 'user'), None)
        
        if first_user_query:
            
            display_text = (first_user_query[:40] + '...') if len(first_user_query) > 40 else first_user_query
            date_obj = datetime.strptime(chat_id, "%Y-%m-%d_%H-%M-%S")
            date_str = date_obj.strftime("%b %d, %Y")
            
            return f"{display_text} - {date_str}"
        
    except (FileNotFoundError, json.JSONDecodeError, StopIteration):
        
        pass
    return chat_id

def get_sorted_chat_history():
    """Gets a sorted list of chat history files."""
    files = os.listdir(CHAT_HISTORY_DIR)
    
    json_files = [f for f in files if f.endswith('.json')]
    json_files.sort(key=lambda x: os.path.getmtime(os.path.join(CHAT_HISTORY_DIR, x)), reverse=True)
    return [os.path.splitext(f)[0] for f in json_files]

def new_chat():
    """Starts a new chat session."""
    st.session_state.messages = []
    st.session_state.chat_engine = None
    st.session_state.current_chat_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.uploader_key += 1
    save_chat_history() 
    st.rerun()


#Main Application Logic
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Multi-file Universal RAG Bot",
        page_icon="🤖",
        layout="wide"
    )
    load_dotenv()
    load_css("styles.css")
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)
    if not os.path.exists(FAISS_INDEX_PATH):
        os.makedirs(FAISS_INDEX_PATH)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
   

    #Sidebar UI
    with st.sidebar:
        st.header("Chat Controls")
        if st.button("➕ New Chat"):
            new_chat()
        
        st.header("Chat History")
        chat_history_files = get_sorted_chat_history()
        for chat_id in chat_history_files:
            display_name = get_chat_display_name(chat_id)
            if st.button(display_name, key=f"history_{chat_id}", use_container_width=True):
                load_chat_history(chat_id)
        

    #Main Chat Interface
    st.title("📄 Multi-file Universal RAG Bot")
    st.markdown("Upload your documents and ask questions about them.")

    #File Uploader
    with st.expander("📁 Upload and Process Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload one or more files (PDF, TXT, DOCX, etc.)",
            type=None, 
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_key}"
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a moment."):
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Load documents from the temporary directory
                    reader = SimpleDirectoryReader(input_dir=temp_dir)
                    documents = reader.load_data()
                    
                    if not documents:
                        st.warning("Could not read any content from the uploaded files. Please check the file formats.")
                        return
                    Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
                    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                    d = 384 
                    faiss_index = faiss.IndexFlatL2(d)
                    vector_store = FaissVectorStore(faiss_index=faiss_index)
                    index = VectorStoreIndex.from_documents(
                        documents, vector_store=vector_store
                    )
                    chat_engine = index.as_chat_engine(
                        chat_mode="condense_plus_context", 
                        verbose=True
                    )
                    
                    st.session_state.chat_engine = chat_engine
                    st.success("Documents processed successfully! You can now ask questions.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.chat_engine is None:
            st.warning("Please upload and process documents before asking questions.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_engine.chat(prompt)
                    response_text = str(response)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        save_chat_history()


if __name__ == "__main__":
    main()
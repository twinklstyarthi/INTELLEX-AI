import streamlit as st
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import base64

USER_DATA_DIR = "user_data"

def load_css(file_name):
    """Loads a CSS file into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def get_image_as_base64(path):
    """Encodes a local image file to a Base64 string."""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    st.session_state.setdefault("session_id", f"session_{uuid.uuid4()}")
    st.session_state.setdefault("chat_histories", {})
    st.session_state.setdefault("current_chat_id", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_engine", None)
    st.session_state.setdefault("index", None) 
    st.session_state.setdefault("uploader_key", 0)

def save_current_chat():
    """Saves the state of the currently active chat into the session's chat history."""
    if st.session_state.current_chat_id: 
        st.session_state.chat_histories[st.session_state.current_chat_id] = {
            "messages": list(st.session_state.messages),
            "chat_engine": st.session_state.chat_engine,
            "index": st.session_state.index,
        }

def new_chat():
    """Starts a new chat session."""
    save_current_chat()
    new_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.current_chat_id = new_id
    st.session_state.messages = []
    st.session_state.chat_engine = None
    st.session_state.index = None
    st.session_state.uploader_key += 1
    st.session_state.chat_histories[new_id] = {"messages": [], "chat_engine": None, "index": None}

def load_chat(chat_id):
    """Loads a previous chat session from the session state."""
    if chat_id == st.session_state.current_chat_id:
        return 
        
    save_current_chat()
    
    if chat_id in st.session_state.chat_histories:
        chat_data = st.session_state.chat_histories[chat_id]
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = chat_data.get("messages", [])
        st.session_state.chat_engine = chat_data.get("chat_engine", None)
        st.session_state.index = chat_data.get("index", None)
        st.session_state.uploader_key += 1
    else:
        st.error("Chat history not found in this session.")

def get_chat_display_name(chat_id):
    """Generates a display name for a chat session from session state."""
    if chat_id in st.session_state.chat_histories:
        messages = st.session_state.chat_histories[chat_id]["messages"]
        first_user_query = next((msg['content'] for msg in messages if msg['role'] == 'user'), "New Chat")
        
        display_text = (first_user_query[:40] + '...') if len(first_user_query) > 40 else first_user_query
        return display_text
    return "New Chat"

#Main Application Logic
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="INTELLEX AI",
        page_icon="assets/favicon.png", 
        layout="wide"
    )
    load_dotenv()
    load_css("styles.css")
    os.makedirs(USER_DATA_DIR, exist_ok=True)

    initialize_session_state()

    if not st.session_state.current_chat_id:
        new_chat()

    with st.sidebar:
        try:
            logo_b64 = get_image_as_base64("assets/logo.png")
            st.markdown(f"""
                <div class="sidebar-header">
                    <img src="data:image/png;base64,{logo_b64}" alt="Logo">
                    <span>Your personal document expert.</span>
                </div>
                """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.markdown(
                '<div class="sidebar-header"><span>Logo not found.</span></div>',
                unsafe_allow_html=True
            )
        
        st.header("Chat Controls")
        if st.button("➕ New Chat", use_container_width=True, on_click=new_chat):
            st.rerun()
        
        st.header("Chat History")
        chat_ids = sorted(st.session_state.chat_histories.keys(), reverse=True)
        for chat_id in chat_ids:
            display_name = get_chat_display_name(chat_id)
            button_type = "primary" if chat_id == st.session_state.current_chat_id else "secondary"
            if st.button(
                display_name,
                key=f"history_{chat_id}",
                on_click=load_chat,
                args=(chat_id,),
                use_container_width=True,
                type=button_type,
            ):
                st.rerun()

    st.title("📄 INTELLEX AI")
    st.markdown("Upload your documents and ask questions about them. You can add more documents at any time.")

    with st.expander("📁 Upload and Process Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload one or more files (PDF, TXT, DOCX, etc.)",
            type=None,
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_key}"
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a moment."):
                session_specific_dir = os.path.join(USER_DATA_DIR, st.session_state.session_id)
                os.makedirs(session_specific_dir, exist_ok=True)

                with tempfile.TemporaryDirectory(dir=session_specific_dir) as temp_dir:
                    for uploaded_file in uploaded_files:
                        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    reader = SimpleDirectoryReader(input_dir=temp_dir)
                    new_documents = reader.load_data()
                    
                    if not new_documents:
                        st.warning("Could not read any content. Please check file formats.")
                        return
                    
                    if st.session_state.index is None:
                        st.info("Creating a new knowledge base for this chat...")
                        Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
                        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                        
                        d = 384
                        faiss_index = faiss.IndexFlatL2(d)
                        vector_store = FaissVectorStore(faiss_index=faiss_index)
                        
                        index = VectorStoreIndex.from_documents(new_documents, vector_store=vector_store)
                        st.session_state.index = index
                        st.success("New knowledge base created successfully!")
                    else:
                        st.info("Adding new documents to the existing knowledge base...")
                        for doc in new_documents:
                           st.session_state.index.insert(doc)
                        st.success("New documents added successfully!")

                    chat_engine = st.session_state.index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)
                    st.session_state.chat_engine = chat_engine
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if st.session_state.chat_engine is None:
            st.warning("Please upload and process documents before asking questions.")
            st.rerun()
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_engine.chat(prompt)
                    response_text = str(response)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            save_current_chat()
            st.rerun()

if __name__ == "__main__":
    main()
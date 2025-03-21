import streamlit as st
import time
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import StorageContext

# Set page configuration
st.set_page_config(
    page_title="DocChat - RAG Powered Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more visually appealing
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #2b313e;
        color: #ffffff;
        border-bottom-right-radius: 0;
    }
    .chat-message.bot {
        background-color: #ffffff;
        border-bottom-left-radius: 0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .thinking-animation {
        display: inline-block;
        position: relative;
        width: 80px;
        height: 20px;
    }
    .thinking-animation div {
        position: absolute;
        top: 8px;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #2b313e;
        animation-timing-function: cubic-bezier(0, 1, 1, 0);
    }
    .thinking-animation div:nth-child(1) {
        left: 8px;
        animation: thinking1 0.6s infinite;
    }
    .thinking-animation div:nth-child(2) {
        left: 8px;
        animation: thinking2 0.6s infinite;
    }
    .thinking-animation div:nth-child(3) {
        left: 32px;
        animation: thinking2 0.6s infinite;
    }
    .thinking-animation div:nth-child(4) {
        left: 56px;
        animation: thinking3 0.6s infinite;
    }
    @keyframes thinking1 {
        0% {transform: scale(0);}
        100% {transform: scale(1);}
    }
    @keyframes thinking3 {
        0% {transform: scale(1);}
        100% {transform: scale(0);}
    }
    @keyframes thinking2 {
        0% {transform: translate(0, 0);}
        100% {transform: translate(24px, 0);}
    }
    .stButton>button {
        background-color: #4e8cff;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3a7be0;
    }
    .progress-container {
        width: 100%;
        background-color: #f1f1f1;
        border-radius: 10px;
        margin: 10px 0;
    }
    .progress-bar {
        height: 10px;
        background-color: #4CAF50;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    /* Remove form borders and spacing */
    .stForm {
        border: none !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None

# Constants matching your notebook
COLLECTION_NAME = "chat_with_docs"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "llama3.2:1b"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"

def connect_to_existing_index():
    """Connect to the existing Qdrant index rather than creating a new one"""
    progress_text = "Connecting to your knowledge base..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # Step 1: Initialize Qdrant client
        my_bar.progress(20, text=f"{progress_text} (20%)")
        client = qdrant_client.QdrantClient(
            host="localhost",
            port=6333
        )
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME not in collection_names:
            my_bar.empty()
            st.error(f"Collection '{COLLECTION_NAME}' not found in Qdrant.")
            st.info("Make sure you've run the notebook to index your documents first.")
            return None
        
        # Step 2: Load embedding model
        my_bar.progress(40, text=f"{progress_text} (40%)")
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=True
        )
        Settings.embed_model = embed_model
        
        # Step 3: Connect to vector store
        my_bar.progress(60, text=f"{progress_text} (60%)")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME
        )
        
        # Step 4: Create index from existing vector store
        my_bar.progress(80, text=f"{progress_text} (80%)")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        
        # Step 5: Load LLM
        llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
        Settings.llm = llm
        
        # Step 6: Create query engine with reranker
        rerank = SentenceTransformerRerank(
            model=RERANKER_MODEL, 
            top_n=5
        )
        
        # Define QA prompt template (same as in notebook)
        template = """Context information is below:
                      ---------------------
                      {context_str}
                      ---------------------
                      Given the context information above I want you to think
                      step by step to answer the query in a crisp manner,
                      incase you don't know the answer say 'I don't know! I was not able to find the answer in the documents provided'
                    
                      Query: {query_str}
                
                      Answer:"""
        
        qa_prompt_tmpl = PromptTemplate(template)
        
        query_engine = index.as_query_engine(
            similarity_top_k=15,
            node_postprocessors=[rerank]
        )
        
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )
        
        my_bar.progress(100, text="Successfully connected to your knowledge base!")
        time.sleep(0.5)
        my_bar.empty()
        
        return query_engine
        
    except Exception as e:
        my_bar.empty()
        st.error(f"Error connecting to Qdrant: {str(e)}")
        if "connection" in str(e).lower():
            st.info("Make sure Qdrant is running on localhost:6333. You can start it with Docker: docker run -p 6333:6333 qdrant/qdrant")
        return None

def display_chat_messages():
    """Display chat messages in a visually appealing way"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="avatar">👤</div>
                    <div class="message">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:  # Assistant
            with st.container():
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="avatar">🤖</div>
                    <div class="message">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

def thinking_animation():
    """Display a thinking animation while processing the query"""
    with st.container():
        st.markdown("""
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">
                <div>Thinking...</div>
                <div class="thinking-animation">
                    <div></div><div></div><div></div><div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def process_query(query):
    """Process the query and add response to chat history"""
    if not query or not query.strip():
        return
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display thinking animation
    thinking_placeholder = st.empty()
    with thinking_placeholder.container():
        thinking_animation()
    
    # Process the query
    try:
        response = st.session_state.query_engine.query(query)
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
    except Exception as e:
        # Add error message to chat
        st.session_state.messages.append({"role": "assistant", 
                                        "content": f"Sorry, I encountered an error: {str(e)}"})
    
    # Clear thinking animation
    thinking_placeholder.empty()

def main():
    # App header
    st.title("📚 DocChat - RAG Powered Document Assistant")
    st.subheader("Ask questions about your documents and get intelligent answers")
    
    # Sidebar for settings and info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        DocChat uses RAG (Retrieval Augmented Generation) to provide intelligent answers from your documents.
        
        **Features:**
        - 🔍 Semantic search with Qdrant
        - 🧠 Powered by Llama 3 (1B parameters)
        - 🔄 Cross-encoder reranking for better context relevance
        - 📄 PDF document support
        """)
        
        st.header("Settings")
        
        # System status
        st.subheader("System Status")
        if st.session_state.query_engine is None:
            st.warning("⚠️ Not connected to knowledge base")
            if st.button("Connect to Knowledge Base"):
                try:
                    st.session_state.query_engine = connect_to_existing_index()
                    if st.session_state.query_engine:
                        st.success("✅ Connected successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        else:
            st.success("✅ Connected to knowledge base")
            
            # Add a reset button
            if st.button("Disconnect"):
                st.session_state.query_engine = None
                st.rerun()
    
    # Main chat interface
    display_chat_messages()
    
    # Only show query input if connected to knowledge base
    if st.session_state.query_engine is not None:
        # Create a form for the query input
        with st.form(key="query_form", clear_on_submit=True):
            user_query = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What happened on June 14, 2017?",
                key="query_input"
            )
            submit_button = st.form_submit_button("Ask")
            
            if submit_button and user_query:
                process_query(user_query)
                st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your system setup and try again.")
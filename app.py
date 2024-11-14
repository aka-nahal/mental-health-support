import streamlit as st
import requests
from datetime import datetime
import time

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "ALIENTELLIGENCE/mindwell:latest"
TIMEOUT = 60  # Increased timeout to 60 seconds
MAX_RETRIES = 3

# Page setup
st.set_page_config(
    page_title="MindWell Chat",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Styling
st.markdown("""
<style>
    .stApp {
        background-color: #1a1b1e;
        color: #ffffff;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #7c5cff;
        margin-left: 2rem;
        border-top-right-radius: 0;
    }
    
    .assistant-message {
        background-color: #2d2d2d;
        margin-right: 2rem;
        border-top-left-radius: 0;
    }
    
    .chat-header {
        text-align: center;
        padding: 2rem 0;
        color: #7c5cff;
    }
    
    .warning-banner {
        background-color: rgba(255, 82, 82, 0.1);
        border: 1px solid rgba(255, 82, 82, 0.3);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .timestamp {
        font-size: 0.7rem;
        color: #8e8e8e;
        align-self: flex-end;
        margin-top: 0.2rem;
    }
    
    .error-message {
        background-color: rgba(255, 82, 82, 0.1);
        border-left: 4px solid #ff5252;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stTextInput input {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border-radius: 1.5rem !important;
        border: 1px solid #7c5cff !important;
        padding: 0.8rem 1.2rem !important;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

def check_ollama_status():
    """Check if Ollama server is running and model is loaded"""
    try:
        response = requests.get(f"http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

def send_message(message: str, retries=0) -> str:
    """Send a message to Ollama with retry logic"""
    if retries >= MAX_RETRIES:
        return "Error: Maximum retries reached. Please try again later."
    
    try:
        # Prepare context from previous messages
        context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in st.session_state.messages[-5:]
        ])
        
        # Prepare prompt
        prompt = f"Context:\n{context}\n\nUser: {message}\nAssistant:"
        
        # Send request
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("response", "I apologize, but I couldn't generate a response.")
        else:
            time.sleep(1)  # Wait before retry
            return send_message(message, retries + 1)
            
    except requests.exceptions.Timeout:
        time.sleep(2)  # Wait longer before retry
        return send_message(message, retries + 1)
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama server. Please ensure it is running."
    except Exception as e:
        return f"Error: {str(e)}"

def get_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().strftime("%I:%M %p")

# Check if Ollama is running
if not st.session_state.model_ready:
    if check_ollama_status():
        st.session_state.model_ready = True
    else:
        st.error("⚠️ Ollama server is not running. Please start the Ollama server first.")
        st.stop()

# Display header
st.markdown("""
    <div class="chat-header">
        <h1>🌱 MindWell</h1>
        <p>A safe space for support and understanding</p>
    </div>
""", unsafe_allow_html=True)

# Display warning banner
st.markdown("""
    <div class="warning-banner">
        ⚠️ <strong>Crisis Support Resources:</strong><br>
        • Emergency: 911 (US)<br>
        • Suicide & Crisis Lifeline: 988<br>
        • Crisis Text Line: Text HOME to 741741
    </div>
""", unsafe_allow_html=True)

# Chat interface
chat_container = st.container()

# Display chat history
for message in st.session_state.messages:
    message_type = "user-message" if message["role"] == "user" else "assistant-message"
    with chat_container:
        st.markdown(f"""
            <div class="chat-message {message_type}">
                <div class="content">{message["content"]}</div>
                <div class="timestamp">{message["timestamp"]}</div>
            </div>
        """, unsafe_allow_html=True)

# Chat input
if st.session_state.model_ready:
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Add user message
        timestamp = get_timestamp()
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Get and display AI response
        with st.spinner("Thinking... This might take a moment."):
            ai_response = send_message(user_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": timestamp
            })
            st.rerun()

# Sidebar with clear button
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.model_ready = False
        st.rerun()

    # Display connection status
    if st.session_state.model_ready:
        st.success("✓ Connected to Ollama")
    else:
        st.error("✗ Not connected to Ollama")
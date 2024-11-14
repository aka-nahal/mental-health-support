import streamlit as st
import requests
import sqlite3
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from textblob import TextBlob
from collections import Counter
from typing import List, Dict, Optional
import uuid
import re

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "ALIENTELLIGENCE/mindwell:latest"
TIMEOUT = 60
MAX_RETRIES = 3
DB_PATH = "chat_history.db"

class IntegratedChatApp:
    def __init__(self):
        self.setup_database()
        self.setup_session_state()
        
    def setup_database(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    sentiment REAL,
                    topics TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
    
    def setup_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'model_ready' not in st.session_state:
            st.session_state.model_ready = False
        if 'current_response' not in st.session_state:
            st.session_state.current_response = ""
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = self.start_new_conversation(str(uuid.uuid4()))
        if 'page' not in st.session_state:
            st.session_state.page = 'chat'

    def apply_styling(self):
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
            
            .stream-response {
                background-color: #2d2d2d;
                padding: 1rem;
                margin-right: 2rem;
                border-radius: 0.5rem;
                border-top-left-radius: 0;
                animation: fadeIn 0.3s ease-in;
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
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            #MainMenu, footer, header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    def check_ollama_status(self):
        try:
            response = requests.get(f"http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def stream_response(self, message: str) -> str:
        response_container = st.empty()
        accumulated_response = []
        
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": message,
                    "stream": True
                },
                stream=True,
                timeout=TIMEOUT
            )
            
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    token = json_response.get("response", "")
                    accumulated_response.append(token)
                    current_response = "".join(accumulated_response)
                    response_container.markdown(f"""
                        <div class="stream-response">
                            {current_response}
                        </div>
                    """, unsafe_allow_html=True)
            
            return "".join(accumulated_response)
            
        except Exception as e:
            return f"Error: {str(e)}"

    def start_new_conversation(self, session_id: str) -> int:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (session_id) VALUES (?)",
                (session_id,)
            )
            return cursor.lastrowid

    def save_message(self, role: str, content: str):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (st.session_state.conversation_id, role, content)
            )

    def analyze_sentiment(self, text: str) -> float:
        return TextBlob(text).sentiment.polarity

    def extract_topics(self, text: str, top_n: int = 5) -> List[str]:
        # Simple keyword extraction using word frequency
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Split into words
        words = text.split()
        
        # Remove common stop words
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                         "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
                         'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
                         'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                         'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                         'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
                         'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                         'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                         'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                         'further', 'then', 'once'])
        
        words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Get top N most common words
        return [word for word, _ in word_freq.most_common(top_n)]

    def analyze_conversation(self) -> Dict:
        messages = st.session_state.messages
        if not messages:
            return None
            
        full_text = " ".join([msg["content"] for msg in messages])
        
        analysis = {
            "sentiment": self.analyze_sentiment(full_text),
            "topics": self.extract_topics(full_text),
            "message_lengths": [len(msg["content"].split()) for msg in messages],
            "total_messages": len(messages)
        }
        
        analysis["avg_message_length"] = sum(analysis["message_lengths"]) / len(analysis["message_lengths"])
        
        return analysis

    def create_visualizations(self, analysis: Dict):
        if not analysis:
            return None
            
        sentiment_fig = go.Figure(data=go.Indicator(
            mode="gauge+number",
            value=analysis["sentiment"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Sentiment"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "red"},
                    {'range': [-0.3, 0.3], 'color': "yellow"},
                    {'range': [0.3, 1], 'color': "green"}
                ]
            }
        ))
        
        message_length_fig = px.histogram(
            x=analysis["message_lengths"],
            title="Message Length Distribution",
            labels={"x": "Words per message", "y": "Count"}
        )
        
        topics_fig = px.bar(
            x=analysis["topics"],
            y=[1] * len(analysis["topics"]),
            title="Main Topics Discussed",
            labels={"x": "Topic", "y": "Frequency"}
        )
        
        return {
            "sentiment_gauge": sentiment_fig,
            "message_length_dist": message_length_fig,
            "topics_chart": topics_fig
        }

    def display_chat_page(self):
        if not self.check_ollama_status():
            st.error("⚠️ Ollama server is not running. Please start the Ollama server first.")
            return
            
        st.session_state.model_ready = True
        
        # Display header and warning banner
        st.markdown("""
            <div class="chat-header">
                <h1>🌱 MindWell</h1>
                <p>A safe space for support and understanding</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="warning-banner">
                ⚠️ <strong>Crisis Support Resources:</strong><br>
                • Emergency: 911 (US)<br>
                • Suicide & Crisis Lifeline: 988<br>
                • Crisis Text Line: Text HOME to 741741
            </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        chat_container = st.container()
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
        user_input = st.chat_input("Type your message here...")
        if user_input:
            timestamp = datetime.now().strftime("%I:%M %p")
            
            # Save user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })
            self.save_message("user", user_input)
            
            # Get and save AI response
            with st.spinner("Thinking..."):
                response = self.stream_response(user_input)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": timestamp
                })
                self.save_message("assistant", response)
            
            st.rerun()

    def display_analysis_page(self):
        st.header("Conversation Analysis")
        
        analysis = self.analyze_conversation()
        if not analysis:
            st.info("No messages to analyze yet. Start a conversation first!")
            return
            
        visualizations = self.create_visualizations(analysis)
        
        # Display sentiment gauge
        st.plotly_chart(visualizations["sentiment_gauge"], use_container_width=True)
        
        # Display topics
        st.subheader("Main Topics")
        st.plotly_chart(visualizations["topics_chart"], use_container_width=True)
        
        # Display message statistics
        st.subheader("Message Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", analysis["total_messages"])
        with col2:
            st.metric("Avg. Message Length", f"{analysis['avg_message_length']:.1f} words")
        
        # Display message length distribution
        st.plotly_chart(visualizations["message_length_dist"], use_container_width=True)

    def run(self):
        st.set_page_config(page_title="MindWell", layout="wide")
        self.apply_styling()
        
        # Sidebar navigation
        with st.sidebar:
            st.title("Navigation")
            page = st.radio("Go to", ["Chat", "Analysis"])
            
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.conversation_id = self.start_new_conversation(str(uuid.uuid4()))
                st.rerun()
        
        # Display selected page
        if page == "Chat":
            self.display_chat_page()
        else:
            self.display_analysis_page()

if __name__ == "__main__":
    app = IntegratedChatApp()
    app.run()
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

# Emotional categories and their sentiment ranges
EMOTION_RANGES = {
    "Very Happy": (0.6, 1.0),
    "Happy": (0.2, 0.6),
    "Neutral": (-0.2, 0.2),
    "Sad": (-0.6, -0.2),
    "Very Sad": (-1.0, -0.6)
}

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
                    topics TEXT,
                    emotional_state TEXT
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
        if 'stop_generation' not in st.session_state:
            st.session_state.stop_generation = False

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
            
            .emotion-card {
                padding: 20px;
                background-color: #2d2d2d;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 1rem;
            }
            
            .suggestion-card {
                padding: 15px;
                background-color: rgba(124, 92, 255, 0.1);
                border: 1px solid rgba(124, 92, 255, 0.3);
                border-radius: 0.5rem;
                margin-bottom: 1rem;
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

    def determine_emotional_state(self, sentiment: float) -> str:
        for emotion, (low, high) in EMOTION_RANGES.items():
            if low <= sentiment <= high:
                return emotion
        return "Neutral"

    def stream_response(self, message: str) -> str:
        response_container = st.empty()
        accumulated_response = []
        stop_button = st.button("Stop Generation", key="stop_button")
        
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": message + "\n\nPlease keep your response concise and focused, ideally under 3-4 sentences unless more detail is specifically requested. gimme in points when required , move to second line when need, be precise, start with question about question, use <br> after every point, ",
                    "stream": True
                },
                stream=True,
                timeout=TIMEOUT
            )
            
            for line in response.iter_lines():
                if stop_button or st.session_state.stop_generation:
                    st.session_state.stop_generation = False
                    break
                    
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
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                         "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                         'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                         'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                         'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                         'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                         'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
                         'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 
                         'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                         'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                         'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                         'again', 'further', 'then', 'once'])
        
        words = [word for word in words if word not in stop_words and len(word) > 3]
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(top_n)]

    def analyze_conversation(self) -> Dict:
        messages = st.session_state.messages
        if not messages:
            return None
            
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        full_text = " ".join(user_messages)
        
        sentiment = self.analyze_sentiment(full_text)
        emotional_state = self.determine_emotional_state(sentiment)
        
        analysis = {
            "sentiment": sentiment,
            "emotional_state": emotional_state,
            "topics": self.extract_topics(full_text),
            "message_lengths": [len(msg["content"].split()) for msg in messages],
            "total_messages": len(messages),
            "user_message_count": len(user_messages)
        }
        
        analysis["avg_message_length"] = sum(analysis["message_lengths"]) / len(analysis["message_lengths"])
        
        return analysis

    def create_visualizations(self, analysis: Dict):
        sentiment_fig = go.Figure(data=go.Indicator(
            mode="gauge+number",
            value=analysis["sentiment"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Sentiment"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "#7c5cff"},
                'steps': [
                    {'range': [-1, -0.6], 'color': "red"},
                    {'range': [-0.6, -0.2], 'color': "orange"},
                    {'range': [-0.2, 0.2], 'color': "yellow"},
                    {'range': [0.2, 0.6], 'color': "lightgreen"},
                    {'range': [0.6, 1], 'color': "green"}
                ]
            }
        ))
        
        message_length_fig = px.histogram(
            x=analysis["message_lengths"],
            title="Message Length Distribution",
            labels={"x": "Words per message", "y": "Count"},
            color_discrete_sequence=["#7c5cff"]
        )
        
        topics_fig = px.bar(
            x=analysis["topics"],
            y=[1] * len(analysis["topics"]),
            title="Main Topics Discussed",
            labels={"x": "Topic", "y": "Frequency"},
            color_discrete_sequence=["#7c5cff"]
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
        
        st.markdown("""
            <div class="chat-header">
                <h1>🌱 MindWell</h1>
                <p>A safe space for support and understanding</p><br>
                <p><b>This Is a Copyrighted Model, Use of this for commerical purposes is a crime under copyright laws if used by any third party.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="warning-banner">
                ⚠️ <strong>Crisis Support Resources:</strong><br>
                • Emergency: 112 (India)<br>
                • Suicide & Crisis Lifeline: 9152987821<br>
                • Crisis Text Line: Text HOME to 741741 <br>
                <p><b>Disclaimer: This chatbot is not a certified mental health professional and is intended solely for testing purposes. The responses provided here should not be considered as professional advice. For serious mental health concerns, please consult a qualified professional.</b><p>
            </div>
        """, unsafe_allow_html=True)
        
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
        
        user_input = st.chat_input("Type your message here...")
        if user_input:
            timestamp = datetime.now().strftime("%I:%M %p")
            
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })
            self.save_message("user", user_input)
            
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
        
        suggestions = {
            "Very Happy": "Keep up the positive energy! Consider journaling about what's going well.",
            "Happy": "You're doing great! Consider sharing your positive experiences with others.",
            "Neutral": "This is a good time for reflection and planning.",
            "Sad": "Consider taking a break, practice self-care, or reach out to someone you trust.",
            "Very Sad": "Please remember you're not alone. Consider reaching out to a mental health professional or calling a support line."
        }
        
        st.markdown(f"""
            <div class="emotion-card">
                <h3>{analysis['emotional_state']}</h3>
                <p>Based on conversation analysis</p>
            </div>
            <div class="suggestion-card">
                {suggestions[analysis['emotional_state']]}
            </div>
        """, unsafe_allow_html=True)
        
        visualizations = self.create_visualizations(analysis)
        
        st.plotly_chart(visualizations["sentiment_gauge"], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", analysis["total_messages"])
            st.metric("User Messages", analysis["user_message_count"])
        with col2:
            st.metric("Avg. Message Length", f"{analysis['avg_message_length']:.1f} words")
        
        st.subheader("Main Topics")
        st.plotly_chart(visualizations["topics_chart"], use_container_width=True)
        
        st.subheader("Message Length Distribution")
        st.plotly_chart(visualizations["message_length_dist"], use_container_width=True)

    def run(self):
        st.set_page_config(
            page_title="MindWell",
            page_icon="🌱",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.apply_styling()
        
        with st.sidebar:
            st.title("🌱 MindWell")
            st.markdown("---")
            page = st.radio("Navigation", ["Chat", "Analysis"])
            
            st.markdown("---")
            if st.button("New Conversation"):
                st.session_state.messages = []
                st.session_state.conversation_id = self.start_new_conversation(str(uuid.uuid4()))
                st.rerun()
                
            st.markdown("---")
            st.markdown("""
                ### About
                MindWell is a supportive chat interface designed to provide a safe space for expression and reflection.
                It's Part of College Project and is not intended for commercial use.
                For Mini Project II
                
                Contact : me@lonedetective.moe
                
                ### Tips
                - Be honest and open
                - Take your time
                - Focus on your feelings
                - Review your progress in Analysis
                
               ### ⚠️ Disclamier ⚠️
               This is AI, trained  on datasets not an actual person.
               This is not professional solution, incase serious issues please seek help from a mental health professional. 
            """)
        
        if page == "Chat":
            self.display_chat_page()
        else:
            self.display_analysis_page()

if __name__ == "__main__":
    app = IntegratedChatApp()
    app.run()
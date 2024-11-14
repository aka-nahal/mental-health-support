import sqlite3
from datetime import datetime
import json
from typing import List, Dict, Optional
import pandas as pd

class ChatHistoryManager:
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
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
    
    def start_new_conversation(self, session_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (session_id) VALUES (?)",
                (session_id,)
            )
            return cursor.lastrowid
    
    def add_message(self, conversation_id: int, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content)
            )
    
    def get_conversation_history(self, conversation_id: int) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
            return [
                {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2]
                }
                for row in cursor.fetchall()
            ]
    
    def get_recent_conversations(self, limit: int = 10) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    c.id,
                    c.session_id,
                    c.timestamp,
                    COUNT(m.id) as message_count,
                    c.sentiment,
                    c.topics
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.timestamp DESC
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def update_conversation_analysis(self, conversation_id: int, sentiment: float, topics: List[str]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET sentiment = ?, topics = ? WHERE id = ?",
                (sentiment, json.dumps(topics), conversation_id)
            )
    
    def export_conversation(self, conversation_id: int, format: str = "json") -> str:
        history = self.get_conversation_history(conversation_id)
        
        if format == "json":
            return json.dumps(history, indent=2)
        elif format == "txt":
            return "\n\n".join([
                f"{msg['role'].upper()} ({msg['timestamp']}):\n{msg['content']}"
                for msg in history
            ])
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_conversation_metrics(self, conversation_id: int) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get basic metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_messages,
                    SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) as user_messages,
                    SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time
                FROM messages
                WHERE conversation_id = ?
            """, (conversation_id,))
            
            metrics = dict(zip(
                ['total_messages', 'user_messages', 'assistant_messages', 'start_time', 'end_time'],
                cursor.fetchone()
            ))
            
            # Get sentiment and topics
            cursor.execute(
                "SELECT sentiment, topics FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            sentiment, topics = cursor.fetchone()
            
            metrics.update({
                'sentiment': sentiment,
                'topics': json.loads(topics) if topics else []
            })
            
            return metrics
from src.sentiment import SentimentEngine
import random

class Chatbot:
    def __init__(self):
        self.sentiment_engine = SentimentEngine()
        self.history = [] # List of dicts: {'role': 'user'|'bot', 'content': '...', 'sentiment': ...}
        self.user_statements_analysis = [] # Store analysis of user inputs specifically

    def process_user_input(self, user_text):
        # 1. Analyze sentiment of the user's input (Tier 2)
        analysis = self.sentiment_engine.analyze_statement(user_text)
        self.user_statements_analysis.append(analysis)
        
        # 2. Store in history
        self.history.append({
            'role': 'user',
            'content': user_text,
            'sentiment': analysis
        })

        # 3. Generate response
        bot_response = self._generate_response(user_text, analysis['label'])
        
        # 4. Store bot response
        self.history.append({
            'role': 'bot',
            'content': bot_response,
            'sentiment': None # We don't analyze bot's own sentiment for this requirement
        })

        return {
            'response': bot_response,
            'user_sentiment': analysis
        }

    def get_final_analysis(self):
        # Tier 1: Conversation-Level Sentiment Analysis
        return self.sentiment_engine.analyze_conversation(self.user_statements_analysis)

    def _generate_response(self, text, sentiment_label):
        # Simple flexible logic based on sentiment
        if sentiment_label == 'Negative':
            responses = [
                "I'm sorry to hear that. How can I help?",
                "That sounds frustrating. Tell me more.",
                "I apologize if things aren't going well."
            ]
        elif sentiment_label == 'Positive':
            responses = [
                "That's great to hear!",
                "I'm glad you're feeling positive.",
                "Awesome! What else is on your mind?"
            ]
        else:
            responses = [
                "I see. Please go on.",
                "Could you tell me more?",
                "Interesting."
            ]
        return random.choice(responses)

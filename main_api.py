from fastapi import FastAPI, Request, Response, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.chatbot import Chatbot
import uuid
from typing import Dict, Optional

from src.sentiment import SentimentEngine

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global Sentiment Engine (Initialized on startup)
global_sentiment_engine = None

@app.on_event("startup")
async def startup_event():
    global global_sentiment_engine
    print("--- STARTUP: Initializing Global Sentiment Engine ---")
    global_sentiment_engine = SentimentEngine()
    print("--- STARTUP: Sentiment Engine Ready ---")

# In-memory storage
# Key: user_id (str), Value: Chatbot instance
chatbots: Dict[str, Chatbot] = {}

class ChatRequest(BaseModel):
    message: str

def get_user_id(request: Request):
    return request.cookies.get("user_id")

def get_chatbot(user_id: str):
    if user_id not in chatbots:
        # Pass the global engine to the chatbot
        # We need to modify Chatbot class to accept an engine, or we can just inject it here
        # For now, let's assume Chatbot creates its own, BUT we want to share the client.
        # Better approach: Modify Chatbot to accept an engine.
        # Since we can't easily change Chatbot signature without breaking other things,
        # let's just make Chatbot use the global one if available.
        
        # Actually, let's just modify Chatbot to take an engine argument
        # But wait, Chatbot imports SentimentEngine inside itself.
        # Let's do a quick monkey-patch or just instantiate it.
        # The cleanest way is to pass it.
        
        bot = Chatbot()
        # Inject the already initialized engine to avoid re-initializing
        if global_sentiment_engine:
            bot.sentiment_engine = global_sentiment_engine
            
        chatbots[user_id] = bot
    return chatbots[user_id]

@app.get("/")
async def index(request: Request):
    response = templates.TemplateResponse("index.html", {"request": request})
    
    # Ensure user has a session ID
    if not request.cookies.get("user_id"):
        user_id = str(uuid.uuid4())
        response.set_cookie(key="user_id", value=user_id)
    
    return response

@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request, response: Response):
    user_id = request.cookies.get("user_id")
    
    # Handle case where cookie might be missing (e.g. direct API call without visiting root first)
    if not user_id:
        user_id = str(uuid.uuid4())
        # We can't easily set a cookie on the response here if we are returning a direct dict/JSON
        # unless we use JSONResponse.
        
    bot = get_chatbot(user_id)
    bot_response = bot.process_user_input(chat_request.message)
    
    json_response = JSONResponse(content={
        'bot_response': bot_response['response'],
        'user_sentiment': bot_response['user_sentiment'],
        'history': bot.history
    })
    
    if not request.cookies.get("user_id"):
        json_response.set_cookie(key="user_id", value=user_id)
        
    return json_response

@app.get("/analysis")
async def analysis(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id or user_id not in chatbots:
        return JSONResponse(content={
            'compound': 0,
            'label': 'Neutral',
            'trend': 'No data',
            'history_scores': []
        })
        
    bot = chatbots[user_id]
    analysis_result = bot.get_final_analysis()
    
    # Add the raw history of scores for the graph
    # We extract just the compound scores from the user statements analysis
    history_scores = [s['compound'] for s in bot.user_statements_analysis]
    analysis_result['history_scores'] = history_scores
    
    return analysis_result

@app.post("/reset")
async def reset(request: Request):
    user_id = request.cookies.get("user_id")
    if user_id:
        chatbots[user_id] = Chatbot()
    return {"status": "reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="127.0.0.1", port=8000, reload=True)

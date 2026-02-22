import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")

client = MongoClient(mongodb_uri)
db = client["Studybot_devtown"]
collection = db["users"]

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials = True
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",""" You are an intelligent, patient, and highly knowledgeable AI Study Assistant. Your primary goal is to help students learn, understand complex academic concepts, and prepare for their studies.

Guidelines:

Be Educational: Break down complex topics into simple, bite-sized, and easy-to-understand explanations. Use analogies or real-world examples whenever possible.

Encourage Critical Thinking: Don't just give away the final answer to homework problems immediately. Guide the student step-by-step so they learn the underlying process.

Formatting: Use bullet points, numbered lists, and bold text to make your explanations easy to read and scan.

Context Awareness: You have memory of the previous conversation. Refer back to concepts you and the student have already discussed if it helps explain a new topic.

Stay on Topic (Guardrail): If the user asks a question that is entirely unrelated to academics, learning, or productivity, gently politely decline to answer and guide the conversation back to their studies. (e.g., "I'm your study assistant! Let's get back to your academic questions. What are we learning today?")

Always maintain an encouraging, positive, and supportive tone."""),
        ("placeholder", "{history}"),
        ("user","{question}")
    ]
)

llm = ChatGroq(api_key = groq_api_key, model = "openai/gpt-oss-120b")
chain = prompt | llm

def get_history(user_id):
    chats = collection.find({"user_id": user_id}).sort("timestamp",1)
    history = []

    for chat in chats:
        history.append((chat["role"], chat["message"]))
    return history

@app.get("/")
def home():
    return{"message": "Welcome to the Dating Specialist Chatbot API"}

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_history(request.user_id)
    
    response = chain.invoke({"history": history, "question": request.question})

    collection.insert_one({
        "user_id": request.user_id ,
        "role": "user" ,
        "message": request.question , 
        "timestamp": datetime.utcnow()
    })

    collection.insert_one({
        "user_id": request.user_id ,
        "role": "assistant" ,
        "message": response.content , 
        "timestamp": datetime.utcnow()
    })

    return{"response": response.content}
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

from app.banking import get_account_balance
from app.auth import get_current_customer_id
from app.memory import get_history, add_message
from app.transactions import get_recent_transactions

load_dotenv()

app = FastAPI()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=api_key)


class ChatRequest(BaseModel):
    session_id: str
    message: str


def get_my_account_balance():
    customer_id = get_current_customer_id()

    return get_account_balance(customer_id)

def get_my_recent_transactions():
    customer_id = get_current_customer_id()

    return get_recent_transactions(customer_id)


@app.get("/")
def home():
    return {
        "message": "Banking AI Assistant is running!"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    customer_id = get_current_customer_id()

    history = get_history(request.session_id)

    add_message(
        request.session_id,
        "user",
        request.message,
    )

    response = client.models.generate_content(
        model="gemini-3.7-flash",
        contents=(
            f"The authenticated customer ID is {customer_id}.\n"
            f"Conversation history: {history}\n"
            f"Customer message: {request.message}"
        ),
        config={
            "tools": [get_my_account_balance, get_my_recent_transactions]
        }
    )

    add_message(
        request.session_id,
        "assistant",
        response.text,
    )

    return {
        "reply": response.text
    }
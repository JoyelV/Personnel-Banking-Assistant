import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=api_key)

chat = client.chats.create(
    model="gemini-3.7-flash"
)

response1 = chat.send_message(
    "My name is Alex and I am testing a banking assistant."
)

print("AI 1:")
print(response1.text)

response2 = chat.send_message(
    "What is my name?"
)

print("\nAI 2:")
print(response2.text)
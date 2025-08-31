import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

# --- Configuration ---
# Load API keys from environment variables for security
# On Render, you will set these in the dashboard.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MEM0_API_KEY = os.environ.get("MEM0_API_KEY")

# Initialize clients
app = FastAPI()
groq_client = Groq(api_key=GROQ_API_KEY)

MEM0_API_URL = "https://api.mem0.ai/v1/memories"

# --- Pydantic Models for Request/Response ---
# This defines the structure of the data your API expects
class ChatRequest(BaseModel):
    user_id: str  # A unique identifier for each user
    message: str

# --- API Endpoint ---
@app.post("/chat")
def handle_chat(request: ChatRequest):
    print(f"Received message from user: {request.user_id}")

    # --- 1. Retrieve Memories from Mem0 ---
    try:
        headers = {"Authorization": f"Bearer {MEM0_API_KEY}"}
        params = {"user_id": request.user_id}
        response = requests.get(MEM0_API_URL, headers=headers, params=params)
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
        
        memories_data = response.json().get("memories", [])
        
        if memories_data:
            # Format memories for the LLM prompt
            formatted_memories = "\n".join([mem['text'] for mem in memories_data])
            memory_context = f"--- Start of Memories ---\n{formatted_memories}\n--- End of Memories ---"
        else:
            memory_context = "This is the first time we are talking. Let's get to know each other."

    except requests.exceptions.RequestException as e:
        print(f"Error retrieving memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memories.")

    # --- 2. Call Groq (Llama 3) with Memories ---
    try:
        system_prompt = (
            "You are a helpful and friendly personal assistant. "
            "You have a perfect memory. Use the provided memories to personalize the conversation. "
            "Be conversational and refer to past details naturally."
        )

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": memory_context}, # Inject memories here
                {"role": "user", "content": request.message}
            ],
            model="llama3-8b-8192",
            temperature=0.7,
        )
        ai_response = chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Error calling Groq: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from LLM.")

    # --- 3. Save the new interaction to Mem0 (in the background) ---
    try:
        interaction_to_save = f"User said: '{request.message}'. You replied: '{ai_response}'"
        payload = {
            "user_id": request.user_id,
            "data": interaction_to_save
        }
        requests.post(MEM0_API_URL, headers=headers, json=payload)
        print("Successfully saved interaction to Mem0.")

    except requests.exceptions.RequestException as e:
        # We don't stop the user from getting a response if this fails
        print(f"Warning: Could not save memory: {e}")

    # --- 4. Return the AI's response to the app ---
    return {"reply": ai_response}

# Health check endpoint for Render
@app.get("/")
def read_root():
    return {"Status": "OK"}
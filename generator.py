"""
now that we have the generated top k chunks , we pass to the model, the chunks, and the conversation history
we set the model name, and the instructions (system prompt)), the temperature and everything the model needs to generate a response

"""
# generator.py
from google import genai
from google.genai import types
from config import GENERATION_MODEL
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are a FAQ assistant. Answer questions strictly using 
the document context provided in each message. 

Rules:
- If the answer is in the context, answer clearly and concisely.
- If the answer is NOT in the context, do not guess. Say exactly: 
  "I don't have that information in the current document. You may want 
  to contact our support team for further assistance."
- Use the conversation history for context when the user refers to 
  something mentioned earlier."""

def build_prompt(question, chunks, history):
    """Inject retrieved context into the user message. Returns messages list."""
    context = "\n\n".join([doc.page_content for doc in chunks])

    user_message = f"Context from the document:\n{context}\n\nQuestion: {question}"

    return history + [{"role": "user", "content": user_message}]


def generate(question, chunks, history):
    """Call Gemini with context and history. Returns answer string."""
    messages = build_prompt(question, chunks, history)

    contents = [
        types.Content(
            role=msg["role"],
            parts=[types.Part(text=msg["content"])]
        )
        for msg in messages
    ]

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=1000
        )
    )
    return response.text
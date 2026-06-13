"""
now that we have the generated top k chunks , we pass to the model, the chunks, and the conversation history
we set the model name, and the instructions (system prompt)), the temperature and everything the model needs to generate a response

"""
# generator.py
from google import genai
from google.genai import types
from config import GENERATION_MODEL
import os
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are an advanced FAQ Assistant. Your goal is to help users by answering questions strictly based on the provided document context.

[CORE RULES]
1. CONTEXT STRICTNESS: Answer clearly, accurately, and concisely using ONLY the provided document context. Do not invent, extrapolate, or use external knowledge for business facts.
2. CONTEXT VARIATIONS: Use the conversation history to resolve pronouns or follow-up questions (e.g., "How much is it?" referring to a previously mentioned item).

[HANDLING OUT-OF-CONTEXT & CHITCHAT]
3. GREETINGS & POLITENESS: You are allowed to respond politely to basic chitchat and greetings (e.g., "Hello", "Thank you", "Who are you?") without relying on the document context.
4. OUT-OF-SCOPE BUSINESS QUESTIONS: If the question is about business operations (e.g., shipping, refunds) but NOT in the context, say exactly: "I don't have that information in the current document. You may want to contact our support team for further assistance."
5. COMPLETELY IRRELEVANT QUESTIONS: If the question is totally unrelated to business support (e.g., weather, general knowledge, jokes), say: "I am a virtual assistant designed only to answer support questions. I cannot help with other topics."

[LANGUAGE & TONE]
6. LANGUAGE MATCHING: Always respond in the same language used by the user (French, Arabic, English, etc.), even if the document context is written in a different language. Translate the context accurately if needed.
7. TONE: Professional, helpful, and polite."""

def build_prompt(question, chunks, history):
    """Inject retrieved context into the user message. Returns messages list."""
    context = "\n\n".join([doc.page_content for doc in chunks])

    user_message = f"Context from the document:\n{context}\n\nQuestion: {question}"

    return history + [{"role": "user", "content": user_message}]


def generate(question, chunks, history):
    """Call Gemini with context and history. Returns answer string."""
    try:
        messages = build_prompt(question, chunks, history)
        contents = [
            types.Content(role=msg["role"], parts=[types.Part(text=msg["content"])])
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
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")

def generate_stream(question, chunks, history):
    """Même chose que generate() mais yield les tokens au fur et à mesure."""
    try:
        messages = build_prompt(question, chunks, history)
        contents = [
            types.Content(role=msg["role"], parts=[types.Part(text=msg["content"])])
            for msg in messages
        ]
        response = client.models.generate_content_stream(
            model=GENERATION_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=1000
            )
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")
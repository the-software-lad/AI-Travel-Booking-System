import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


def get_llm():
    return ChatGoogleGenerativeAI( 
            model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            temperature = 0.2,
            max_tokens = 1500,
            max_retries = 2,
            timeout = 60
        )
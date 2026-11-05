import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from google.genai.errors import ServerError

load_dotenv()

llm = GoogleGenAI(
    model=os.getenv("GEMINI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

try:
    response = llm.complete(
        "Reply with exactly: GEMINI_CONNECTION_OK"
    )

    print("SUCCESS:", response.text)

except ServerError as error:
    if error.code == 503:
        print("Gemini is temporarily overloaded.")
        print("Retry after a short delay.")
    else:
        print("Gemini server error:", error)

except Exception as error:
    print("Unexpected error:", error)
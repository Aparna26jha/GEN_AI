import os
import json
import requests
from dotenv import load_dotenv
 
load_dotenv(override=True)
 
API_KEY = os.getenv("ANTHROPIC_API_KEY")
 
ENDPOINT = "https://llmgw-wp.tekstac.com/v1/chat/completions"
 
MODEL_NAME = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
 
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
 
 
def ask_llm(user_question, temperature=0.5, system_prompt="You are a helpful assistant."):
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 512,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
    }
 
    try:
        response = requests.post(
            ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )
 
        print("\n==================================================")
        print(f"Status Code: {response.status_code}")
        print("==================================================")
 
        try:
            response_json = response.json()
        except ValueError:
            print("Failed to parse JSON response.")
            print("Raw response text:")
            print(response.text)
            return None
 
        print("\nFull JSON Response:")
        print(json.dumps(response_json, indent=2))
 
        if response.status_code != 200:
            print("\nAPI returned an error response.")
            return None
 
        assistant_reply = response_json["choices"][0]["message"]["content"]
 
        print("\nAssistant Reply:")
        print(assistant_reply)
 
        return assistant_reply
 
    except requests.exceptions.Timeout:
        print("Request timed out. Please try again later.")
        return None
 
    except requests.exceptions.RequestException as error:
        print("Request failed:", error)
        return None
 
    except KeyError:
        print("Could not extract assistant reply. Response format may be different.")
        return None
 
    except Exception as error:
        print("Unexpected error occurred:", error)
        return None
 
 
if __name__ == "__main__":
    if not API_KEY:
        print("API key not found. Please check your .env file.")
    else:
        query_1 = input("Enter first question: ")
        query_2 = input("Enter second question: ")
 
        response_1 = ask_llm(
            query_1,
            temperature=0.3,
            system_prompt="You are a helpful assistant. Give a clear and concise answer."
        )
 
        response_2 = ask_llm(
            query_2,
            temperature=0.8,
            system_prompt="You are a creative assistant. Give a detailed answer with examples."
        )
 
        print("\n================ RESPONSE COMPARISON ================")
        print("Query 1:", query_1)
        print("Response 1:", response_1)
 
        print("\nQuery 2:", query_2)
        print("Response 2:", response_2)
 
        print("\nObservation:")
        print("Temperature 0.3 usually gives more focused and stable responses.")
        print("Temperature 0.8 usually gives more creative and varied responses.")
 

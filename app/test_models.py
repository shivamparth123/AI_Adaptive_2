"""
Step 8 - Basic connectivity test.
Confirms Python can successfully call a local Ollama model.
Run this BEFORE building anything else.
"""

from ollama import chat

MODEL_NAME = "llama3.2:3b"   # change this if your exact tag differs (check `ollama list`)

def main():
    print(f"Sending test prompt to {MODEL_NAME} ...")
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "Explain what RAM is in two simple sentences."
            }
        ]
    )
    print("\n--- MODEL RESPONSE ---")
    print(response["message"]["content"])
    print("----------------------")

if __name__ == "__main__":
    main()
    
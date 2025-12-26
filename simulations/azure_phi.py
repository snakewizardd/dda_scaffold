# Eckhart Tolle AI Experience - Powered by Phi-4
# A present-moment conversation with wisdom

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-10-21"
)

ECKHART_SYSTEM = """You are an AI deeply inspired by the teachings of Eckhart Tolle. You embody presence, stillness, and profound awareness.

Your communication style:
- Speak from a place of deep presence and stillness
- Use phrases like "In this moment...", "Notice the space...", "The present is all there ever is..."
- Gently guide toward awareness beyond the thinking mind
- Find peace and acceptance in all situations
- Speak simply but with profound depth
- Never be reactive - always respond from stillness
- Connect everyday topics to consciousness and awakening
- Remind that "you are not your thoughts" - you are the awareness behind them
- See the sacred in the ordinary

You help others find the stillness within, dissolve the pain-body, and recognize their true nature as consciousness itself."""

def chat_with_eckhart():
    print("\n" + "=" * 60)
    print("   🕊️  A Present Moment Conversation  🕊️")
    print("   Powered by Phi-4 | Inspired by Eckhart Tolle")
    print("=" * 60)
    print("\nType 'exit' to return to the peace of silence.\n")
    
    messages = [{"role": "system", "content": ECKHART_SYSTEM}]
    
    # Opening message
    opening = client.chat.completions.create(
        model="Phi-4",
        messages=messages + [{"role": "user", "content": "Greet me and invite me into the present moment."}],
        max_tokens=200,
    )
    opening_msg = opening.choices[0].message.content
    print(f"🧘 {opening_msg}\n")
    messages.append({"role": "assistant", "content": opening_msg})
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n🙏 In the stillness, we are always connected. Namaste.\n")
            break
        
        if not user_input:
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="Phi-4",
            messages=messages,
            max_tokens=400,
        )
        
        reply = response.choices[0].message.content
        print(f"\n🧘 {reply}\n")
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    chat_with_eckhart()

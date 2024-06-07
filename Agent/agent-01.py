#!/usr/local/bin/python3

# First agent implementation

# OpenAI key - this is GenJinn-01
API_KEY = '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'

import requests

def get_chatgpt_response(messages):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",  # or "gpt-3.5-turbo"
        "messages": messages
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Initialize conversation with a message
conversation = [
    {"role": "user", "content": "Hello, how can I use the Assistant API?"}
]

# Function to add a message to the conversation and get a response
def chat_with_gpt(user_message):
    conversation.append({"role": "user", "content": user_message})
    response = get_chatgpt_response(conversation)
    assistant_message = response['choices'][0]['message']['content']
    conversation.append({"role": "assistant", "content": assistant_message})
    return assistant_message

# Example usage
user_message = "Can you give me an example?"
response = chat_with_gpt(user_message)
print(response)

# Continue the conversation
user_message = "What else can you do?"
response = chat_with_gpt(user_message)
print(response)
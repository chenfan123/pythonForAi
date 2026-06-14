from openai_lesson import get_completion_from_messages

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = get_completion_from_messages(messages)
print(response)
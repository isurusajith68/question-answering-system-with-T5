from ollama import chat

# Define the query and input prompt
messages = [
    {
    'role': 'user',
    'content': '''
The evolution of technology has been one of the most significant factors shaping human history.
From the advent of the wheel to the rise of artificial intelligence (AI), technology has played a pivotal
role in the progress of society. [Full input text truncated for brevity...]

Question: Create a multiple-choice question (MCQ) based on the potential of AI to improve efficiency and solve complex problems. Provide four options: one correct answer and three incorrect answers. Ensure the options are plausible and related to the topic.

Answer:'''
}

]

# Chat with the model in streaming mode
stream = chat(model='llama2', messages=messages, stream=True)

# Print the AI's response as it streams
response = ""
for chunk in stream:
    content = chunk.get('message', {}).get('content', '')
    if content:
        response += content
        print(content, end='', flush=True)

# Final formatted output
print("\n\nFinal Response:")
print(response.strip())

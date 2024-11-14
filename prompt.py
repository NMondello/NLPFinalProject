import openai
import time

# Set up API key
openai.api_key = 'YOUR_API_KEY'

def get_model_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003", 
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example prompts
prompts = [""]

results = []

for prompt in prompts:
    result = get_model_response(prompt)
    if result:
        results.append({
            "prompt": prompt,
            "response": result
        })
    time.sleep(1)  


for entry in results:
    print(f"Prompt: {entry['prompt']}\nResponse: {entry['response']}\n")

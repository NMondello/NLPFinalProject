import openai
import time
import apikey
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_model_response(promptQ):
    response = apikey.client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": promptQ,
    }],
    model="gpt-4o-mini",
    )
    return response.choices[0].message.content


#We can either feed this prompt into the api and parse it, or do it manuall with the chat.
start_prompt = "Produce 50 different ways of saying 'What is the most important meal of the day?'"


#Prompts produced by chatGPT
prompts = ["Which meal holds the highest significance in a day?", "What meal do people consider the most vital?"]

#The answer we calculate against for cosine sim. This is a response from a doctor
referenceAnswer = "Breakfast is the most important meal of the day. Starting your day with a nutritious breakfast kick-starts your metabolism and provides fuel for your body and brain"

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
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([entry['response'], referenceAnswer])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    print(f"Prompt: {entry['prompt']}\nResponse: {entry['response']}\nTotal Characters: {len(entry['response'])}\nCosine Sim: {cosine_sim}")

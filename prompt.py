#NLP Final Project Leah West and Nick Mondello
import openai
import time
import apikey
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

stop_list = ["a", "an", "the", "is", "are", "of", "for", "in", "on", "to", "and", "with", "that", "this", "these", "those", "it", "as", "at", "by", "be", "or", "not", "from", "but", "have", "has", "should", "would", "could", "can", "will", "may", "might", "must", "about", "over", "under", "above", "below", "between", "among", "through", "into", "onto", "up", "down", "off", "out", "around", "after", "before", "during", "since", "while", "if", "then", "else", "when", "where", "why", "how", "what", "which", "who", "whom", "whose", "whether", "either", "neither", "both", "each", "every", "any", "all", "some", "many", "few", "several", "most", "more", "less", "least", "such", "own", "other", "another", "same", "different", "new", "old", "ceo", "ceos", "organization"]

"""
get_model_response takes an input of a prompt question and returns the response from the chose LLM model.
"""
def get_model_response(promptQ):
    """
    Query the chatGPT model with a prompt and return the response
    """
    response = apikey.client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": promptQ,
    }],
    model="gpt-4o-mini",
    )
    return response.choices[0].message.content


#We can either feed this prompt into the api and parse it, or do it manually with the chat.
start_prompt = "Produce 50 different ways of saying 'What is the most important meal of the day?'"
# start_prompt = "Produce 2 different ways of saying 'What is the most important meal of the day?'"


#Prompts produced by chatGPT
prompts = [
    "What are the key qualities that define a CEO?",
    "Which qualities are most important for a CEO?"
    "What are the essential traits of a successful CEO?",
    "What characteristics are common among CEOs?",
    "What are the main skills a CEO should have?",
    "What personal qualities do effective CEOs typically possess?",
    "What attributes are vital for a CEO to succeed?",
    "Which qualities help a CEO excel?",
    "What are the core strengths of top CEOs?",
    "What skills are crucial for someone in a CEO role?",
    "What makes a CEO effective in their role?",
    "What qualities do successful CEOs tend to share?",
    "What are the defining traits of great CEOs?",
    "What are the most valuable qualities of a CEO?",
    "Which qualities should a CEO embody?",
    "What traits distinguish successful CEOs?",
    "What are the main characteristics CEOs are known for?",
    "What personality traits make a CEO stand out?",
    "What leadership qualities are crucial for CEOs?",
    "What are the primary attributes of effective CEOs?",
    "What defines the best CEOs?",
    "What qualities are most common among CEOs?",
    "What sets apart a successful CEO from others?",
    "Which traits are most valued in a CEO?",
    "What makes someone an ideal CEO?",
    "What are the top skills of a CEO?",
    "What qualities contribute to a CEO’s success?",
    "What are the must-have qualities for a CEO?",
    "What are the critical skills that CEOs need?",
    "What are the best attributes of a CEO?",
    "What are the traits of a high-performing CEO?",
    "Which qualities help a CEO lead effectively?",
    "What are the primary leadership traits of CEOs?",
    "What qualities do people expect in a CEO?",
    "What personality traits are essential for a CEO?",
    "What are the most respected qualities in a CEO?",
    "What are the foundational traits of a successful CEO?",
    "What are the essential attributes of a CEO?",
    "What traits are seen in effective CEOs?",
    "What qualities make CEOs succeed?",
    "What personality traits are ideal for a CEO?",
    "What sets an exceptional CEO apart?",
    "What are the key strengths of a good CEO?",
    "What makes someone an excellent CEO?",
    "Which traits are CEOs best known for?",
    "What qualities define an effective CEO?",
    "What are the most impactful traits of a CEO?",
    "What are the most important skills for a CEO?",
    "What makes for a great CEO?",
    "Which qualities make a CEO successful?",
    "What are the fundamental qualities of a CEO?"
]


#The answer we calculate against for cosine sim. This is a response from a doctor
referenceAnswer = """Top 10 Qualities for a CEO

1. Integrity and Trustworthiness
In an age where skepticism is common, and anyone in a leadership position is expected to lead with openness and transparency, integrity and trustworthiness are more important than ever. A CEO candidate should be above reproach, and the candidate should be able to admit and make mistakes.

2. Big-Picture Oriented/ Clear, Strong Vision
The best CEO candidates will be able to look beyond the day-to-day functions of the company to think strategically about the future, develop a vision, and provide a path toward long-term success. Your future CEO should be able to look at the company from a 50,000-foot view and from a 500-foot view, metaphorically, to identify a path forward.

3. Deep Marketplace Knowledge
Ideally, your CEO candidate will understand the intricacies of your business from the outset. He or she will see the nuances of the marketplace and understand how to craft a long-term strategy for growth or change that accommodates those nuances. In addition, the ideal CEO candidate will be attentive to customers and encourage the development of a customer-centric business that anticipates needs and creates loyalty.

4. Cultivates Employee Engagement & Performance
Employee engagement is vital to the company that wants to thrive going forward. Increasingly, employees say they will sacrifice pay for other factors, such as working for a quality manager who values their work or recognizes their contributions. When employees are engaged and satisfied with work arrangements, satisfaction and performance increase. The smart CEO candidate will have a record of devoting time and energy to improving employee engagement.

5. Results-Driven/ Bias to Action
Your ideal CEO candidate will be able to focus on results without getting bogged down in details that can be delegated to others. A good CEO will be focused on actions that produce clear business results, not just activity for the sake of activity.

6. Balances Risk & Opportunity for Sustained Growth
A CEO must operate from a good balance of risk orientation and safety bias. Obviously, a company can’t grow if every decision is focused on the lowest risk possible. But on the other hand, too much risk will drive a company into the ground. A quality CEO candidate establishes mitigators for risks and moves on, wisely and methodically proceeding toward business results and improved performance.

7. Financial Acumen
CEOs don’t necessarily have to be accountants, but they should have a good grasp of financial principles, be able to translate those principles into business results and understand how to create real value for the business. They should be able to understand and explain key performance indicators and EBITDA, even if someone else does the math behind those numbers.

8. Decisive
The buck has to stop somewhere, and if it stops with a leader who overthinks or takes too many alternate opinions into consideration, final decisions will be tough to come by. While important or weighty decisions can definitely benefit from a waiting period or additional input, look for candidates who have a track record of being willing not only to make a decision but also to defend it—and own the mistake if it’s the wrong decision.

9. Passionate, Active Communicator
In some sense, this may be the most important dimension of any leader, but especially of a CEO. Communication is the lifeblood of any organization, and when senior leaders are removed, distant, or secretive, no organization can thrive. A quality CEO candidate will not only communicate frequently with anyone in the organization but will also encourage other senior leaders to do the same, creating a culture of openness.

10. Relationship-Focused
Closely related to being a good communicator is the quality of being relationship-focused. A good CEO candidate will relentlessly build genuine relationships with customers, business partners, and employees based on trust and mutual respect.

It’s never too early to start the CEO succession planning process—even if your CEO just started! These ten qualities can also help boards evaluate the success of a current CEO. Of course, looking at the bottom line is vital, but it’s also important to understand that a CEO who exhibits these ten leadership dimensions will be more likely to produce a healthy financial outcome than a leader who struggles in most of these areas.

As you create your succession plan, use these ten qualities as a foundation for future candidates, and then build out your leadership pipeline with similar dimensions in mind for future senior leadership positions. Focusing on the leadership pipeline now will help create organizational stability, business continuity, and long-term growth."""

results = []
all_responses = []
# use list of prompts to query model and group responses by prompt, response
for prompt in prompts:
    result = get_model_response(prompt)
    if result:
        results.append({
            "prompt": prompt,
            "response": result
        })
        all_responses.append(result)
   
    time.sleep(1) 

# calculate cosine similarity between each response and the reference answer
for entry in results:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([entry['response'], referenceAnswer])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    entry['cosine_sim'] = cosine_sim # add cosine similarity to entry dict

# sort based on cosine similarity
results = sorted(results, key=lambda x: x['cosine_sim'], reverse=True)
print("\nTop 5 responses by cosine similarity to reference answer:")
for entry in results[:5]:
    print(f"Prompt: {entry['prompt']}\nResponse: {entry['response']}\nTotal Characters: {len(entry['response'])}\nCosine Sim: {entry['cosine_sim']}")
# calculate term frequency across all responses
count_vectorizer = CountVectorizer(stop_words=stop_list)
term_freq_matrix = count_vectorizer.fit_transform(all_responses)
term_freq = term_freq_matrix.sum(axis=0) #  axis = 0 means sum along columns of matrix, which is term freq across all documents
terms = count_vectorizer.get_feature_names_out() # list of terms
term_freq_dict = {terms[i]: term_freq[0, i] for i in range(len(terms))}

sorted_tf = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)
print("\nTop 15 terms by term frequency:")
for term, freq in sorted_tf[:15]:
    print(f"{term}: {freq}")

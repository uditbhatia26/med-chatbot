from transformers import pipeline
import streamlit as st

# Load a Q&A model from Hugging Face
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Function to load FAQ data from the file
def load_faq_text(filename):
    faqs = []
    with open(filename, "r") as file:
        faq_entry = {}
        for line in file:
            if line.startswith("Q:"):
                faq_entry["question"] = line[3:].strip()
            elif line.startswith("A:"):
                faq_entry["answer"] = line[3:].strip()
                faqs.append(faq_entry)
                faq_entry = {}
    return faqs

# Load FAQ text
faq_text = load_faq_text("faq_text.txt")

# Add basic greetings and small talk responses
greeting_responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! How can I help you?",
    "how are you": "I'm here to help you with MedUrgency-related questions!",
    "what's up": "Just here to answer your questions! How can I assist?",
    "goodbye": "Goodbye! Feel free to come back if you have more questions."
}

# Function to find the best answer to a question
def find_best_answer(question, faq_text):
    # Check for casual greetings first
    for greeting in greeting_responses:
        if greeting in question.lower():
            return greeting_responses[greeting]

    # If no greeting is matched, check the FAQ text
    for faq in faq_text:
        if question.lower() in faq["question"].lower():
            return faq["answer"]

    # If no match found in FAQ, use the model to find an answer based on the context
    answer = qa_pipeline({
        'question': question,
        'context': "\n".join([f["answer"] for f in faq_text])
    })
    return answer['answer']

# Streamlit application
st.title("MedUrgency FAQ Chatbot")
st.write("Ask your questions about MedUrgency!")

# Input from the user
user_input = st.text_input("Your Question:")

# When the user asks a question
if user_input:
    response = find_best_answer(user_input, faq_text)
    st.write("**Response:**", response)

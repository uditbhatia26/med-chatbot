import streamlit as st
from transformers import pipeline

# Load the pre-trained BERT model
nlp = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Function to get BERT responses
def get_bert_response(question):
    context = """
    MedUrgency is an innovative AI-powered healthcare platform designed to assist medical professionals in diagnosing patients effectively and providing patients with seamless access to their health reports. 
    The platform integrates advanced machine learning algorithms and computer vision technology to enhance the diagnostic process across various medical fields.

    The three main tools offered by MedUrgency are:

    1. Cardiovascular Disease Prediction: This tool leverages a sophisticated machine learning model trained on extensive medical data to assess a patient's risk of cardiovascular diseases. It analyzes various health indicators, such as age, weight, cholesterol levels, blood pressure, and family medical history, to provide doctors with valuable insights that help in early diagnosis and preventive care.

    2. Lung X-ray Analysis for Pneumonia Detection: MedUrgency employs computer vision technology to analyze lung X-ray images, enabling quick and accurate detection of pneumonia and other lung conditions. This tool assists radiologists by highlighting potential areas of concern and providing a preliminary assessment, thereby facilitating faster decision-making and treatment.

    3. Ocular Recognition for Eye Disease Analysis: This feature allows doctors to upload images of patients' eyes for analysis. The AI model examines the images for signs of common eye diseases, such as diabetic retinopathy or glaucoma, offering insights that can lead to timely interventions and improved patient outcomes.

    MedUrgency prioritizes data security and patient privacy. All patient data is encrypted and managed according to strict confidentiality protocols, ensuring that sensitive information is accessible only to authorized healthcare professionals.

    The platform also enhances the patient experience by offering a secure portal for accessing health reports. Patients no longer need to visit the clinic in person to collect their results; instead, they can view their reports online as soon as they are available. This feature significantly streamlines the communication process between healthcare providers and patients.

    Team Deluxe Thali is the dedicated group behind MedUrgency, comprising software developers, data scientists, and medical professionals. Each team member brings unique expertise, contributing to the development of reliable healthcare solutions that address modern challenges. The team is motivated by a shared vision of leveraging technology to improve healthcare accessibility and efficiency.

    In addition to the core features, MedUrgency includes a virtual chatbot that assists users with frequently asked questions, provides symptom input assistance, and helps users understand potential health concerns. The chatbot is designed to guide users through the platform, ensuring they can effectively utilize the available tools and resources.

    MedUrgency is continuously evolving, with plans to expand its features and capabilities in response to the changing needs of the healthcare landscape. The team is dedicated to staying at the forefront of medical technology, exploring new ideas, and developing innovative solutions that enhance patient care and streamline medical workflows.

    Overall, MedUrgency aims to be a trusted partner in the healthcare journey, providing tools that empower both doctors and patients in making informed decisions about health and well-being.
    """

    response = nlp(question=question, context=context)
    return response['answer'] if response['score'] > 0.1 else None  # Only return answer if confidence is above threshold

# Common greetings and questions
greetings = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What would you like to know?",
    "how are you?": "I'm just a bot, but I'm here to help you with your questions!",
    "who is team deluxe thali?": "Team Deluxe Thali is the dedicated group behind MedUrgency, comprising software developers, data scientists, and medical professionals."
}

# Function to get responses
def get_response(user_input):
    user_input_lower = user_input.lower()
    if user_input_lower in greetings:
        return greetings[user_input_lower]
    else:
        return get_bert_response(user_input)

# Streamlit app structure
st.title("MedUrgency FAQ Chatbot")
st.write("Welcome! Ask me anything related to MedUrgency.")

user_input = st.text_input("You: ", "")

if user_input:
    response = get_response(user_input)
    if response:
        st.write(f"Bot: {response}")
    else:
        st.write("I'm not sure how to respond to that. Can you ask something else?")

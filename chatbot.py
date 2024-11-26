# Install required libraries
#!pip install nltk tensorflow dialogflow
#!pip install dialogflow
!pip install google-cloud-dialogflow
!pip install nltk tensorflow google-cloud-dialogflow
#!pip install ntlk tensorflow dialogflow
import nltk
import random
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from google.cloud import  dialogflow
#from dialogflow import DialogflowClient

# Download NLTK data
nltk.download('punkt')

# Define intents and responses for the chatbot
intents = {
    "greetings": {
        "patterns": ["hi", "hello", "hey"],
        "responses": ["Hello!", "Hi there!", "Hey!"]
    },
    "how_are_you": {
        "patterns": ["how are you?", "how are you doing?"],
        "responses": ["I'm doing well, thank you!", "I'm great, thanks for asking."]
    },
    "name": {
        "patterns": ["what is your name?", "who are you?"],
        "responses": ["I'm a HealthCare chatbot!I can provide general health information"]
    },
    "fever": {
        "patterns" : ["i have fever ","feeling unwell"],
        "responses" :["It's important to take care of yourself,especially when you have a fever. Make sure to rest, drink plenty of fluids, and consider taking over-the-counter fever reducers like Dolo or acetaminophen or ibuprofen if you're not allergic to them and if it's safe for you to do so. If your fever persists or if you have other concerning symptoms,  it's best to consult with a healthcare professional for further guidance. Take care!. "]
    },
    "cold": {
        "patterns" : ["suffering from cold ","i have cold"],
        "responses" :["It's important to take care of yourself,consider taking medicines like febrexplus or coldact or ibuprofen or benadryl if you're not allergic to them. If your cold persists ,  it's best to consult with a healthcare professional for further guidance. Take care!. "]
    },
    "pain": {
        "patterns" : ["i have stomach ache "],
        "responses" :["Here are some steps you can take to help alleviate your stomach ache:::::: Rest down and try to relax or Applying a heating pad or hot water bottle to your abdomen or take medicines like cyclopam or loperamide or bismuth subsalicylate and  If you feel up to eating,stick to bland, easily digestible foods like crackers, rice, bananas, or toast "]
    },

    "health_info": {
        "patterns": ["health information", "health tips", "health advice"],
        "responses": ["I'm not a doctor, but I can provide general health information."]
    }
    # Add more intents and responses as needed
}

# Define training data for TensorFlow model
training_data = []
training_labels = []
for intent, intent_data in intents.items():
    for pattern in intent_data["patterns"]:
        training_data.append(pattern)
        training_labels.append(intent)

# Tokenize and vectorize training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_data)
vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(word_tokenize(sentence)) for sentence in training_data])
X_train = tokenizer.texts_to_sequences(training_data)
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
y_train = np.array([list(intents.keys()).index(label) for label in training_labels])

# Define and train TensorFlow model
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(len(intents), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)

# Function to classify intent using TensorFlow model
def classify_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    return list(intents.keys())[np.argmax(prediction)]

# Function to generate response based on intent
def generate_response(intent):
    return random.choice(intents[intent]["responses"])

# Function to interact with the chatbot
def chat_with_bot(text):
    intent = classify_intent(text)
    response = generate_response(intent)
    return response

# Initialize Dialogflow client
#dialogflow_client = dialogflow.AgentsClient(os.environ["DIALOGFLOW_PROJECT_ID"], os.environ["DIALOGFLOW_LANGUAGE_CODE"])
dialogflow_client = dialogflow.AgentsClient();

# Function to get health-related information from Dialogflow
def get_health_info(query):
    response = dialogflow_client.query(query)
    return response

# Example interaction with the chatbot
user_input = input("You: ")
if "health" in user_input:
    health_info = get_health_info(user_input)
    print("Chatbot:", health_info)
else:
    response = chat_with_bot(user_input)
    print("Chatbot:", response)
while True:
    user_input = input("You: ")

    if "health" in user_input:
        health_info = get_health_info(user_input)
        print("Chatbot:", health_info)
    else:
        response = chat_with_bot(user_input)
        print("Chatbot:", response)

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and load data
lemmatizer = WordNetLemmatizer()
intents = json.load(open(r"d:\New folder\Ecommerce_FAQ_Chatbot_dataset.json"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbotmodel.h5")

# Preprocess the input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Create a bag-of-words array for the input sentence
def bagw(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for i, word in enumerate(words):
        if word in sentence_words:
            bag[i] = 1
    return np.array(bag)

# Predict the intent class
def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"class": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get the answer from the questions dataset
def get_answers(intents_list, Questions_json):
    if not intents_list:
        return "Sorry, I don't have an answer for that."
    
    predicted_class = intents_list[0]["class"]  # Use the predicted class from the model
    for q in Questions_json["Questions"]:
        if predicted_class.lower() in q["question"].lower():
            return q["answer"]
    
    return "Sorry, I don't have an answer for that."

# Start the chatbot
print("Chatbot is up!")

# Chat loop
while True:
    message = input("You: ")
    ints = predict_class(message)
    res = get_answers(ints, intents)
    print(f"Chatbot: {res}")

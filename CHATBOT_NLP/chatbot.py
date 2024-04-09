from flask import Flask, render_template, request
import random
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

app = Flask(__name__)

# Your data
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hi",
                "Hey",
                "How are you",
                "Is anyone there?",
                "Hello",
                "Good day"
            ],
            "responses": [
                "Hey :-)",
                "Hello, thanks for visiting",
                "Hi there, what can I do for you?",
                "Hi there, how can I help?"
            ]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": [
                "See you later, thanks for visiting",
                "Have a nice day",
                "Bye! Come back again soon."
            ]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful", "Thank's a lot!"],
            "responses": ["Happy to help!", "Any time!", "My pleasure"]
        },
        {
            "tag": "items",
            "patterns": [
                "Which items do you have?",
                "What kinds of items are there?",
                "What do you sell?"
            ],
            "responses": [
                "We sell coffee and tea",
                "We have coffee and tea"
            ]
        },
        {
            "tag": "payments",
            "patterns": [
                "Do you take credit cards?",
                "Do you accept Mastercard?",
                "Can I pay with Paypal?",
                "Are you cash only?"
            ],
            "responses": [
                "We accept VISA, Mastercard and Paypal",
                "We accept most major credit cards, and Paypal"
            ]
        },
        {
            "tag": "delivery",
            "patterns": [
                "How long does delivery take?",
                "How long does shipping take?",
                "When do I get my delivery?"
            ],
            "responses": [
                "Delivery takes 2-4 days",
                "Shipping takes 2-4 days"
            ]
        },
        {
            "tag": "funny",
            "patterns": [
                "Tell me a joke!",
                "Tell me something funny!",
                "Do you know a joke?"
            ],
            "responses": [
                "Why did the hipster burn his mouth? He drank the coffee before it was cool.",
                "What did the buffalo say when his son left for college? Bison."
            ]
        }
    ]
}

# Preprocessing data
all_words = []
tags = []
patterns_responses = []

for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern)
        all_words.extend(tokens)
        patterns_responses.append((tokens, intent['tag']))

stemmer = PorterStemmer()
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in '?']
all_words = sorted(list(set(all_words)))

training = []
output_empty = [0] * len(tags)

for pattern, tag in patterns_responses:
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in pattern]
    for w in all_words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[tags.index(tag)] = 1
    
    training.append([bag, output_row])

# Convert training data to numpy array
training_data = []
training_labels = []

for item in training:
    training_data.append(item[0])
    training_labels.append(item[1])

training_data = np.array(training_data)
training_labels = np.array(training_labels)

print("Shape of training data:", training_data.shape)
print("Shape of training labels:", training_labels.shape)

training = list(zip(training_data, training_labels))

# Building model
model = Sequential([
    Dense(128, input_shape=(len(all_words),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(tags), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training model
X_train = training_data
y_train = training_labels

model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Function to predict tag for given message
def predict_tag(message):
    message_tokens = word_tokenize(message)
    message_tokens = [stemmer.stem(word.lower()) for word in message_tokens]
    bow = [0]*len(all_words)
    for s in message_tokens:
        for i,w in enumerate(all_words):
            if w == s: 
                bow[i] = 1
    
    res = model.predict(np.array(bow).reshape(-1,len(bow)))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((tags[r[0]], r[1]))
    return return_list

# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_chatbot_response():
    user_input = request.form['user_input']
    predicted_tags = predict_tag(user_input)
    responses = []
    for tag, confidence in predicted_tags:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                responses.extend(intent['responses'])
    if responses:
        return random.choice(responses)
    else:
        return "I'm sorry, I don't understand your question."

if __name__ == '__main__':
    app.run(debug=True)

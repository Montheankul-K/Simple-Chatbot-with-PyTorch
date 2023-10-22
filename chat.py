import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import streamlit as st

# Use GPU if have GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('./data/intents.json', 'r') as f:
    intents = json.load(f)

# Load model
file = "./model/data.pth"
data = torch.load(file)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Chat page
st.title('Talking with Minnie')

# Initialize chat history
if 'message' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Hi I'm Minnie. What is up?"):
    message = {"role": "user", "content": prompt}
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(message)

    # Apply model with sentence
    sentence = tokenize(prompt)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.as_tensor(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Response to user
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = f'{random.choice(intent["responses"])}'
                with st.chat_message("assistant"):
                    # Display response in chat message container
                    st.markdown(response)
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        response = f'I do not understand.'
        # Display response in chat message container
        st.markdown(response)
        # Add response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

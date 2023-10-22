import streamlit as st
import json
import random
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load your Siamese network model and other necessary libraries here
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

# Load the questions and answers from the JSON file
def load_questions_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return random.sample(data, 10)  # Select 10 random samples

def get_embedding(text):
    return model.encode([text])[0]

def get_random_samples(data, sample_size=10):
    return random.sample(data, sample_size)

def modify_question_with_option(question, option):
    question_words = ["what", "who", "when", "where", "why", "how"]
    for word in question_words:
        if word in question.lower():
            modified_question = question.replace("?", "")  # Remove question mark only when replacing
            return modified_question.lower().replace(word, option).capitalize(), option
    return question + " " + option, None

# Initialize the SBERT model
sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Initialize your Siamese network model here (replace with your model)
# siamese_model = YourSiameseNetwork()

def get_embedding_sbert(text):
    return sbert_model.encode([text])[0]

# Function to compute similarity using your Siamese network
def compute_similarity_siamese(text1, text2):
    # Preprocess the text data (tokenization, padding, etc.) using your Siamese model
    # Replace this with your actual preprocessing code
    processed_text1 = preprocess_text_with_siamese(text1)
    processed_text2 = preprocess_text_with_siamese(text2)
    
    # Make predictions using your Siamese model
    with torch.no_grad():
        similarity_score = siamese_model(processed_text1, processed_text2)
    return similarity_score

def compute_similarity_sbert(text1, text2):
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb1_2d = emb1.reshape(1, -1)
    emb2_2d = emb2.reshape(1, -1)
    similarity = cosine_similarity(emb1_2d, emb2_2d)[0][0]
    return similarity

def main():
    st.title("Automated Answer Evaluation")

    all_data = load_questions_from_json("./data/sciq/train.json")

    # If "Next" button is pressed, reload random samples
    #if 'samples' not in st.session_state or st.button('Next'):
    #    st.session_state.samples = get_random_samples(all_data)

    # Check session state and populate if necessary
    if 'samples' not in st.session_state:
        st.session_state.samples = get_random_samples(all_data)

    # Always show the "Next" button
    if st.button('Next'):
        st.session_state.samples = get_random_samples(all_data)

    # Iterate over the selected questions
    for current_question in st.session_state.samples:
        st.write("Question:", current_question["question"])

        # Display the supporting text
        support_text = current_question["support"]
        st.write("Supporting Text:", support_text)

        options = [
                current_question["correct_answer"],
                current_question["distractor1"],
                current_question["distractor2"],
                current_question["distractor3"]
            ]

        max_similarity_sbert = 0
        predicted_option_sbert = ""
        max_similarity_siamese = 0
        predicted_option_siamese = ""
        
        for option in options:
            modified_option, used_option = modify_question_with_option(current_question["question"], option)
            
            # Compute similarity using SBERT
            similarity_score_sbert = compute_similarity_sbert(modified_option, support_text) * 100
            
            # Compute similarity using your Siamese network
            similarity_score_siamese = compute_similarity_siamese(modified_option, support_text) * 100

            # Highlight the option in bold if it's used in the modified question
            display_option = f"**{option}**" if option == used_option else option
            st.write(f"Similarity (SBERT): {similarity_score_sbert:.2f}% for option {display_option}")
            st.write(f"Similarity (Siamese): {similarity_score_siamese:.2f}% for option {display_option}")

            # Track the option with the highest similarity score for both methods
            if similarity_score_sbert > max_similarity_sbert:
                max_similarity_sbert = similarity_score_sbert
                predicted_option_sbert = option
            
            if similarity_score_siamese > max_similarity_siamese:
                max_similarity_siamese = similarity_score_siamese
                predicted_option_siamese = option

        # Display the correct and predicted answers for both methods
        correct_answer = current_question["correct_answer"]
        st.write("Correct Answer:", correct_answer)
        st.write("Predicted Answer (SBERT):", predicted_option_sbert)
        st.write("Predicted Answer (Siamese):", predicted_option_siamese)
        
        # Check if the predictions are correct and display the result for both methods
        if correct_answer == predicted_option_sbert:
            st.markdown('<span style="color:green">Correct! (SBERT)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:red">Wrong! (SBERT)</span>', unsafe_allow_html=True)
        
        if correct_answer == predicted_option_siamese:
            st.markdown('<span style="color:green">Correct! (Siamese)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:red">Wrong! (Siamese)</span>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()


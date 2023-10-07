import streamlit as st
import json
import random
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the questions and answers from the JSON file
def load_questions_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return random.sample(data, 10)  # Select 10 random samples

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
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def get_embedding(text):
    return model.encode([text])[0]

def compute_similarity(text1, text2):
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

        max_similarity = 0
        predicted_option = ""
        for option in options:
            modified_option, used_option = modify_question_with_option(current_question["question"], option)
            similarity_score = compute_similarity(modified_option, support_text) * 100  # Convert to percentage

            # Highlight the option in bold if it's used in the modified question
            display_option = f"**{option}**" if option == used_option else option
            st.write(f"Similarity between '{modified_option}' and Supporting Text: {similarity_score:.2f}% for option {display_option}")

            # Track the option with the highest similarity score
            if similarity_score > max_similarity:
                max_similarity = similarity_score
                predicted_option = option

        # Display the correct and predicted answers
        correct_answer = current_question["correct_answer"]
        st.write("Correct Answer:", correct_answer)
        st.write("Predicted Answer:", predicted_option)
        
        # Check if the prediction is correct and display the result
        if correct_answer == predicted_option:
            st.markdown('<span style="color:green">Correct!</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:red">Wrong!</span>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()


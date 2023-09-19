import streamlit as st
import json
from fuzzywuzzy import fuzz
import altair as alt

# Load the questions and answers from a JSON file
def load_questions_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def evaluate_answer(user_answer, correct_answer):
    # Use fuzzy string matching to determine similarity
    similarity_ratio = fuzz.ratio(user_answer.lower(), correct_answer.lower())
    return similarity_ratio >= 80  # Adjust the similarity threshold as needed

def main():
    st.title("Automatic Answer Evaluation App")

    # Load questions and answers from a JSON file
    questions_data = load_questions_from_json("questions.json")

    # Initialize variables to keep track of the current question
    question_index = 0
    correct_answers = 0

    # Display questions to the user
    if question_index < len(questions_data):
        current_question = questions_data[question_index]
        st.write("Question:", current_question["question"])
        options = current_question.get("options", [])
        if options:
            st.write("Options:", options)
        
        user_answer = st.text_area("Your Answer:")
        submitted = st.button("Submit Answer")
    
        if submitted:
            # Set the session_state variable here
            st.session_state["input-0"] = user_answer
            
            correct_answer = current_question.get("correct_answer", "")
            supporting_text = current_question.get("supporting_text", "")
    
            # Evaluate the user's answer
            is_correct = evaluate_answer(user_answer, correct_answer)
    
            # Display result
            if is_correct:
                st.success("Correct!")
                correct_answers += 1
            else:
                st.error("Incorrect. Try again.")
            question_index += 1

    st.write(f"Total Correct Answers: {correct_answers}/{len(questions_data)}")

if __name__ == "__main__":
    main()


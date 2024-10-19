# -*- coding: utf-8 -*-

import json
import string
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import torch
import re
import numpy as np
from transformers import LlamaTokenizer
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from nltk.tokenize import sent_tokenize


with open(r"C:\Users\manar\Downloads\intents.json", 'r') as f:
    data = json.load(f)

#print(data)

nltk.download('punkt')

# Load the LLaMA model and tokenizer
hf_token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=hf_token)

# Use a pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

#model.save_pretrained(r"C:\Users\manar\Downloads\llama_model")
#tokenizer.save_pretrained(r"C:\Users\manar\Downloads\llama_model")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'(?<![a-zA-Z])-|-(?![a-zA-Z])', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '').replace('?', '')))
    return text

def remove_duplicates(response):
    # Normalize by removing excessive white spaces and lowercasing the response
    response = re.sub(r'\s+', ' ', response).strip().lower()

    # Split response into sentences
    sentences = response.split('. ')

    # General filtering of irrelevant or duplicate sentences
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        sentence = sentence.strip()

        # Remove sentences with high similarity (even if not exact)
        is_similar = False
        for seen in seen_sentences:
            if SequenceMatcher(None, sentence, seen).ratio() > 0.8:  # Similarity threshold
                is_similar = True
                break

        if not is_similar and len(sentence) > 10:  # Avoid trivial sentences
            seen_sentences.add(sentence)
            unique_sentences.append(sentence)

    # Join the unique sentences back into a single response
    clean_response = '. '.join(unique_sentences).strip()

   # Ensure the response ends with a complete sentence
    if clean_response[-1] == '.' and len(clean_response.split()) >= 3:
        return clean_response
    else:
        return '. '.join(unique_sentences[:-1]).strip() + '.'  # Remove the last incomplete sentence

# Load a pre-trained Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_response_for_custom_prompts(prompt):
    # Define the context with a direct instruction for the model
    context = (
        "You are a computer science expert. "
        "Provide a clear and detailed technical answer to the following question. "
        "Do not restate or repeat the question. "
        "Focus on delivering the exact technical answer.\n\n"
    )

    # Input text passed to the model
    input_text = f"{context} {prompt}"
    print(f"Input Text for LLaMA: {input_text}")  # Debug the full prompt
    # Generate the text with pipeline, ensuring truncation and proper length
    generated_text = pipe(
        input_text,
        max_length=150,   # Adjust length to allow more details
        do_sample=False,  # Deterministic output
        temperature=0.1,  # Lower temperature for focus
        top_k=10,         # Cap random outputs
        top_p=0.9,        # Nucleus sampling to reduce randomness
        truncation=True   # Ensure the output doesn’t exceed limits
    )
   
    # Strip any unnecessary white spaces from generated response
    generated_answer = generated_text[0]['generated_text'].strip()
    # Remove the prompt and context from the response
    answer_without_prompt = generated_answer.replace(input_text, "").strip()

    # Apply the repetition-removal function
    final_cleaned_response = remove_duplicates(answer_without_prompt)

    # Check if the context was repeated in the generated response
    if "computer science expert" in final_cleaned_response:
        return "This question is unrelated to computer science theory. Please ask a technical question."

    # Return the cleaned response
    return final_cleaned_response

def retrieve_responses(new_question):
    # Preprocess the new question
    preprocessed_question = preprocess_text(new_question)

    # Handle punctuation-only questions
    if all(char in string.punctuation for char in preprocessed_question):
        return ["It looks like you've entered only punctuation marks. Please ask a clear question related to computer science theory."]

    # If it's a valid question, proceed with similarity check
    new_question_embedding = embedding_model.encode(preprocessed_question, convert_to_tensor=True)

    best_match_score = -1
    best_match_response = None

    # Find the most similar question in the dataset
    for intent in data['intents']:
        for pattern in intent['patterns']:
            preprocessed_pattern = preprocess_text(pattern)
            pattern_embedding = embedding_model.encode(preprocessed_pattern, convert_to_tensor=True)

            # Calculate cosine similarity between the new question and existing patterns
            similarity = util.pytorch_cos_sim(new_question_embedding, pattern_embedding).item()

            if similarity > best_match_score:
                best_match_score = similarity
                best_match_response = intent['responses']

    # Return the most similar response if similarity is above a threshold
    if best_match_score > 0.7:  # Adjust threshold as needed
       return best_match_response

    # If no match is found, generate a new response using LLaMA (text generation)
    return generate_response_for_custom_prompts(new_question)

import streamlit as st
import time

# Streamlit UI
st.title("Computer Science Chatbot")
st.write("Ask a question related to computer science theory:")

user_input = st.text_input("Your Question:")
print(f"User Input: {user_input}")  # Debug the input
if st.button("Submit"):
    if user_input:
        with st.spinner("Thinking..."):
            response = retrieve_responses(user_input)
            time.sleep(2)  # Simulate processing time
        st.success("Response:")
        st.write(response)
    else:
        st.warning("Please enter a question.")




import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
nltk.data.path.append('C:/Users/Anmol/AppData/Roaming/nltk_data/tokenizers/punkt')
from nltk.tokenize import sent_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Ensure you have the punkt tokenizer models downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
# Load pre-trained model and tokenizer from Hugging Face
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Efficient for sentence similarity tasks
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to get embeddings for a sentence
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Use mean of hidden states for sentence embedding

# Function to compute similarity between resume and job description
def compute_similarity(resume_text, job_description):
    resume_embedding = get_embedding(resume_text)
    job_embedding = get_embedding(job_description)
    
    # Cosine similarity to compare the embeddings
    similarity = cosine_similarity(resume_embedding, job_embedding)
    return similarity[0][0]

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    resume_text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        resume_text += page.extract_text()
    return resume_text

# Function to extract sentences from text
def get_sentences(text):
    return sent_tokenize(text)

# Function to create similarity heatmap
def create_similarity_heatmap(resume_sentences, job_desc_sentences):
    similarity_matrix = np.zeros((len(resume_sentences), len(job_desc_sentences)))
    
    # Fill the similarity matrix with cosine similarities
    for i, resume_sentence in enumerate(resume_sentences):
        for j, job_desc_sentence in enumerate(job_desc_sentences):
            similarity_matrix[i][j] = compute_similarity(resume_sentence, job_desc_sentence)
    
    # Create and display the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=job_desc_sentences, yticklabels=resume_sentences)
    plt.title('Cosine Similarity Heatmap between Resume and Job Description')
    plt.xlabel('Job Description Sentences')
    plt.ylabel('Resume Sentences')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    st.pyplot(plt)

# Function for feedback loop to analyze similarity
def feedback_loop(resume_sentences, job_desc_sentences, threshold=0.6):
    feedback = []
    total_similarity_score = 0  # To accumulate total similarity score
    total_comparisons = 0  # To count total comparisons for averaging
    
    for resume_sentence in resume_sentences:
        row_feedback = []
        for job_desc_sentence in job_desc_sentences:
            similarity = compute_similarity(resume_sentence, job_desc_sentence)
            total_similarity_score += similarity
            total_comparisons += 1  # Increment count of comparisons
            
            if similarity < threshold:
                row_feedback.append(f"Low similarity with: '{job_desc_sentence}' (Score: {similarity:.2f})")
            else:
                row_feedback.append(f"Good match with: '{job_desc_sentence}' (Score: {similarity:.2f})")
        
        feedback.append(row_feedback)
    
    # Calculate the average similarity score of the entire resume
    avg_similarity_score = total_similarity_score / total_comparisons
    return feedback, avg_similarity_score

# Function to rank resume sentences based on similarity to the job description
def rank_resume_sentences(resume_sentences, job_desc_sentences):
    total_similarity_score = 0  # To accumulate total similarity score
    total_comparisons = 0  # To count total comparisons for averaging
    
    sentence_scores = []
    
    # Calculate the average similarity score for each resume sentence and accumulate
    for resume_sentence in resume_sentences:
        total_score = 0
        for job_desc_sentence in job_desc_sentences:
            total_score += compute_similarity(resume_sentence, job_desc_sentence)
            total_similarity_score += compute_similarity(resume_sentence, job_desc_sentence)
            total_comparisons += 1
        
        avg_score = total_score / len(job_desc_sentences)
        sentence_scores.append((resume_sentence, avg_score))
    
    # Calculate the average similarity score of the entire resume
    avg_resume_similarity_score = total_similarity_score / total_comparisons
    
    # Sort the sentences by their average similarity score (highest first)
    ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    
    return ranked_sentences, avg_resume_similarity_score

# Streamlit interface
st.title("Resume Analyzer")

# File uploader for resume
resume_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])

# Job description input
job_description = st.text_area("Job Description", "")

if resume_file is not None:
    # Extract text from uploaded resume
    resume_text = extract_text_from_pdf(resume_file)
    
    # Extract sentences from resume and job description
    resume_sentences = get_sentences(resume_text)
    job_desc_sentences = get_sentences(job_description)
    
    # Compute similarity and show results
    similarity_score = compute_similarity(resume_text, job_description)
    st.write(f"Semantic Similarity Score: {similarity_score * 100:.2f}%")
    
    # Show feedback loop results
    feedback, avg_similarity_score = feedback_loop(resume_sentences, job_desc_sentences)
    st.write(f"Average Similarity Score of Entire Resume: {avg_similarity_score * 100:.2f}%")
    
    # Show ranked resume sentences
    ranked_resume, avg_ranked_score = rank_resume_sentences(resume_sentences, job_desc_sentences)
    st.write(f"Ranked Resume Sentences based on Similarity to Job Description:")
    for index, (sentence, score) in enumerate(ranked_resume, start=1):
        st.write(f"Rank #{index}: {sentence} (Score: {score:.2f})")
    
    # Create and display the similarity heatmap
    create_similarity_heatmap(resume_sentences, job_desc_sentences)


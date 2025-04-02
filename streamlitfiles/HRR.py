
import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
ratings = pd.read_csv('ratings.csv')
restaurants = pd.read_csv('restaurants.csv')

# Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'restaurant_id', 'rating']], reader)
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(restaurants['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def hybrid_recommendation(user_id, restaurant_id, alpha=0.5):
    cf_pred = algo.predict(user_id, restaurant_id).est
    sim_scores = cosine_sim[restaurant_id] if restaurant_id < len(cosine_sim) else [0] * len(cosine_sim)
    cbf_pred = sum(sim_scores) / len(sim_scores)
    return alpha * cf_pred + (1 - alpha) * cbf_pred

st.title("Hybrid Restaurant Recommendation System")
user_id = st.number_input("Enter User ID:", min_value=1)
restaurant_id = st.number_input("Enter Restaurant ID:", min_value=1)
alpha = st.slider("Set Hybrid Weight (Alpha):", 0.0, 1.0, 0.5)

if st.button("Recommend"):
    prediction = hybrid_recommendation(int(user_id), int(restaurant_id), alpha)
    st.write(f"Predicted Rating: {prediction:.2f}")

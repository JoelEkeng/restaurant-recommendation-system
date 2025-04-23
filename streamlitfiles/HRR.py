import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Set page config
st.set_page_config(page_title="Hybrid Restaurant Recommender", layout="wide")

# Data loading with error handling
@st.cache_data
def load_data():
    try:
        # Try to load from default path
        business_path = "data/business.csv"
        review_path = "data/review.csv"
        
        if os.path.exists(business_path) and os.path.exists(review_path):
            business = pd.read_csv(business_path)
            review = pd.read_csv(review_path)
        else:
            # Try alternative paths if default doesn't work
            business_path = "../data/business.csv"
            review_path = "../data/review.csv"
            business = pd.read_csv(business_path)
            review = pd.read_csv(review_path)
            
        data = pd.merge(left=review, right=business, how='left', on='business_id')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Data preparation
def prepare_data(data):
    if data is None:
        return None
        
    # Data Preprocessing
    data.rename(columns={'stars_x': 'rating', 'stars_y': 'b/s_rating'}, inplace=True)
    data = data[['user_id', 'business_id', 'rating', 'name', 'categories', 'address', 'city', 'state']]
    
    # Filter only restaurant-related businesses in Pennsylvania
    pa_data = data[data['state'] == 'PA'].copy()
    restaurant_data = pa_data[pa_data['categories'].str.contains('Restaurants', na=False)].reset_index(drop=True)
    
    return restaurant_data

# Content-based recommendation functions
def create_content_matrix(data):
    if data is None:
        return None
        
    tfidf = TfidfVectorizer(stop_words='english')
    data['categories'] = data['categories'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data['categories'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_content_recommendations(business_id, cosine_sim, data, k=10):
    if data is None or cosine_sim is None:
        return pd.DataFrame()
        
    try:
        idx = data[data['business_id'] == business_id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:k+1]
        business_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommendations = data.iloc[business_indices][['business_id', 'name', 'categories']]
        recommendations['similarity_score'] = similarity_scores
        return recommendations
    except Exception as e:
        st.error(f"Error in content-based recommendations: {e}")
        return pd.DataFrame()

# Collaborative filtering functions
def train_collaborative_model(data):
    if data is None:
        return None
        
    try:
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(data[['user_id', 'business_id', 'rating']], reader)
        trainset = dataset.build_full_trainset()
        algo = SVD()
        algo.fit(trainset)
        return algo
    except Exception as e:
        st.error(f"Error training collaborative model: {e}")
        return None

def get_collaborative_recommendations(user_id, algo, data, k=10):
    if data is None or algo is None:
        return pd.DataFrame()
        
    try:
        business_ids = data['business_id'].unique()
        preds = [algo.predict(user_id, business_id) for business_id in business_ids]
        preds.sort(key=lambda x: x.est, reverse=True)
        top_business_ids = [pred.iid for pred in preds[:k]]
        
        recommendations = data[data['business_id'].isin(top_business_ids)][['business_id', 'name', 'categories']]
        recommendations = recommendations.drop_duplicates().head(k)
        return recommendations
    except Exception as e:
        st.error(f"Error in collaborative recommendations: {e}")
        return pd.DataFrame()

# Hybrid recommendation function
def hybrid_recommendations(user_id, business_id, data, algo, cosine_sim, k=10):
    content_recs = get_content_recommendations(business_id, cosine_sim, data, k*2)
    collab_recs = get_collaborative_recommendations(user_id, algo, data, k*2)
    
    if content_recs.empty and collab_recs.empty:
        return pd.DataFrame()
    elif content_recs.empty:
        return collab_recs.head(k)
    elif collab_recs.empty:
        return content_recs.head(k)
    else:
        hybrid_recs = pd.concat([content_recs, collab_recs]).drop_duplicates(subset=['business_id']).head(k)
        return hybrid_recs

# Main Streamlit app
def main():
    st.title("Pennsylvania Restaurant Hybrid Recommendation System")
    
    # Load data with progress indicators
    with st.spinner("Loading data..."):
        data = load_data()
        
    if data is None:
        st.error("Failed to load data. Please check your data files.")
        return
        
    with st.spinner("Preparing data..."):
        pa_restaurants = prepare_data(data)
        
    if pa_restaurants is None:
        st.error("Failed to prepare data.")
        return
        
    # Check if we have enough data
    if len(pa_restaurants) < 10:
        st.warning("Insufficient data for recommendations. Need at least 10 restaurants.")
        return
        
    # Create content-based similarity matrix
    with st.spinner("Creating content similarity matrix..."):
        cosine_sim = create_content_matrix(pa_restaurants)
        
    # Train collaborative model
    with st.spinner("Training collaborative model..."):
        algo = train_collaborative_model(pa_restaurants)
        
    if algo is None:
        st.error("Failed to train collaborative model.")
        return
        
    # Sidebar for user input
    st.sidebar.header("Recommendation Parameters")
    
    # Get unique user IDs and business names
    unique_users = pa_restaurants['user_id'].unique()
    unique_businesses = pa_restaurants[['business_id', 'name']].drop_duplicates()
    
    # User selection
    user_id = st.sidebar.selectbox("Select User ID", unique_users[:100])  # Limiting to first 100 for performance
    
    # Business selection
    business_name = st.sidebar.selectbox("Select a Restaurant", unique_businesses['name'])
    business_id = unique_businesses[unique_businesses['name'] == business_name]['business_id'].values[0]
    
    # Number of recommendations
    num_recs = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    # Generate recommendations
    if st.sidebar.button("Generate Recommendations"):
        st.subheader(f"Recommendations for User {user_id} based on {business_name}")
        
        # Content-based
        with st.spinner("Generating content-based recommendations..."):
            content_recs = get_content_recommendations(business_id, cosine_sim, pa_restaurants, num_recs)
        
        # Collaborative
        with st.spinner("Generating collaborative recommendations..."):
            collab_recs = get_collaborative_recommendations(user_id, algo, pa_restaurants, num_recs)
        
        # Hybrid
        with st.spinner("Generating hybrid recommendations..."):
            hybrid_recs = hybrid_recommendations(user_id, business_id, pa_restaurants, algo, cosine_sim, num_recs)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])
        
        with tab1:
            if not content_recs.empty:
                st.dataframe(content_recs)
            else:
                st.warning("No content-based recommendations available.")
        
        with tab2:
            if not collab_recs.empty:
                st.dataframe(collab_recs)
            else:
                st.warning("No collaborative recommendations available.")
        
        with tab3:
            if not hybrid_recs.empty:
                st.dataframe(hybrid_recs)
            else:
                st.warning("No hybrid recommendations available.")
    
    # Display sample data
    if st.checkbox("Show Sample Data"):
        st.write(pa_restaurants.head())

if __name__ == "__main__":
    main()
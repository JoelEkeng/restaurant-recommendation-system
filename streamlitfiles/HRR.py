import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data functions
@st.cache_data
def load_business_data():
    business = pd.read_csv("data/business.csv")
    review = pd.read_csv("data/review.csv")
    data = pd.merge(left=review, right=business, how='left', on='business_id')
    return data

@st.cache_data
def prepare_data(data):
    # Data Preprocessing
    data.rename(columns={'stars_x': 'rating', 'stars_y': 'b/s_rating'}, inplace=True)
    data = data[['user_id', 'business_id', 'rating', 'name', 'categories', 'address', 'city', 'state']]
    
    # Filter only restaurant-related businesses in Pennsylvania
    pa_data = data[data['state'] == 'PA'].copy()
    restaurant_data = pa_data[pa_data['categories'].str.contains('Restaurants', na=False)].reset_index(drop=True)
    
    return restaurant_data

# Content-based recommendation functions
@st.cache_data
def create_content_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    data['categories'] = data['categories'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data['categories'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_content_recommendations(business_id, cosine_sim, data, k=10):
    # Get the index of the business
    idx = data[data['business_id'] == business_id].index[0]
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort businesses based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get scores of top-k most similar businesses
    sim_scores = sim_scores[1:k+1]
    
    # Get business indices and similarity scores
    business_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    # Return top-k most similar businesses
    recommendations = data.iloc[business_indices][['business_id', 'name', 'categories']]
    recommendations['similarity_score'] = similarity_scores
    
    return recommendations

# Collaborative filtering functions
def train_collaborative_model(data):
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['user_id', 'business_id', 'rating']], reader)
    trainset = dataset.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

def get_collaborative_recommendations(user_id, algo, data, k=10):
    # Get list of all business IDs
    business_ids = data['business_id'].unique()
    
    # Get predictions for all businesses
    preds = [algo.predict(user_id, business_id) for business_id in business_ids]
    
    # Sort predictions by estimated rating
    preds.sort(key=lambda x: x.est, reverse=True)
    
    # Get top-k business IDs
    top_business_ids = [pred.iid for pred in preds[:k]]
    
    # Get business details
    recommendations = data[data['business_id'].isin(top_business_ids)][['business_id', 'name', 'categories']]
    recommendations = recommendations.drop_duplicates().head(k)
    
    return recommendations

# Hybrid recommendation function
def hybrid_recommendations(user_id, business_id, data, algo, cosine_sim, k=10):
    # Content-based recommendations
    content_recs = get_content_recommendations(business_id, cosine_sim, data, k*2)
    
    # Collaborative recommendations
    collab_recs = get_collaborative_recommendations(user_id, algo, data, k*2)
    
    # Merge recommendations
    hybrid_recs = pd.concat([content_recs, collab_recs]).drop_duplicates(subset=['business_id']).head(k)
    
    return hybrid_recs

# Main Streamlit app
def main():
    st.title("Pennsylvania Restaurant Recommendation System")
    st.subheader("Hybrid Collaborative Filtering")
    
    # Load and prepare data
    data = load_business_data()
    pa_restaurants = prepare_data(data)
    
    # Create content-based similarity matrix
    cosine_sim = create_content_matrix(pa_restaurants)
    
    # Train collaborative model
    algo = train_collaborative_model(pa_restaurants)
    
    # Sidebar for user input
    st.sidebar.header("User Input")
    
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
        st.write("### Content-Based Recommendations")
        content_recs = get_content_recommendations(business_id, cosine_sim, pa_restaurants, num_recs)
        st.dataframe(content_recs)
        
        # Collaborative
        st.write("### Collaborative Filtering Recommendations")
        collab_recs = get_collaborative_recommendations(user_id, algo, pa_restaurants, num_recs)
        st.dataframe(collab_recs)
        
        # Hybrid
        st.write("### Hybrid Recommendations")
        hybrid_recs = hybrid_recommendations(user_id, business_id, pa_restaurants, algo, cosine_sim, num_recs)
        st.dataframe(hybrid_recs)
    
    # Display sample data
    if st.checkbox("Show Sample Data"):
        st.write(pa_restaurants.head())

if __name__ == "__main__":
    main()
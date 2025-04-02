import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split

@st.cache_data
def load_business_data():
    business = pd.read_csv("data/business.csv")
    review = pd.read_csv("data/review.csv")
    data = pd.merge(left=review, right=business, how='left', on='business_id')
    return data

data = load_business_data()

# Data Preprocessing
data.rename(columns={'stars_x': 'rating', 'stars_y': 'b/s_rating'}, inplace=True)
data = data[['user_id', 'business_id', 'rating', 'name', 'categories']]

# Filter only restaurant-related businesses
data = data[data['categories'].str.contains('Restaurants', na=False)].reset_index(drop=True)

# Display title
st.title("Collaborative Filtering Restaurant Recommendations")

# Get unique user IDs
unique_users = data['user_id'].unique()

# User input for the number of users to recommend for
st.subheader("Select Users for Recommendations")
num_users = st.slider("Select number of users to recommend (1 - 10)", 1, 10, 3)
user_ids_to_recommend = st.multiselect(
    "Select User IDs",
    unique_users[:50],  # Displaying the first 50 users for selection
    default=list(unique_users[:num_users])
)

if user_ids_to_recommend:
    # Perform collaborative filtering on selected users
    st.subheader("Generating Recommendations...")

    # Create a Surprise Reader object, specifying the rating scale
    reader = Reader(rating_scale=(1, 5))

    # Load the data into a Surprise Dataset
    dataset = Dataset.load_from_df(data[['user_id', 'business_id', 'rating']], reader)

    # Split the data into training and testing sets
    trainset, testset = train_test_split(dataset, test_size=0.25, random_state=42)

    # Train an SVD model
    algo = SVD()
    algo.fit(trainset)

    # Make predictions on the test set and evaluate the model
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    st.write(f"Model RMSE: {rmse:.4f}")

    # Function to get collaborative filtering recommendations
    def get_collaborative_recommendations(user_ids, algo=algo, data=data):
        recommendations = {}
        for user_id in user_ids:
            business_ids = data['business_id'].unique()
            preds = [algo.predict(user_id, business_id) for business_id in business_ids]
            preds.sort(key=lambda x: x.est, reverse=True)
            top_business_ids = [pred.iid for pred in preds[:10]]
            top_recommendations = data[data['business_id'].isin(top_business_ids)][['business_id', 'name']].drop_duplicates()
            recommendations[user_id] = top_recommendations
        return recommendations

    # Get recommendations for the selected users
    recommendations = get_collaborative_recommendations(user_ids_to_recommend)

    # Display the recommendations
    for user_id, recs in recommendations.items():
        st.write(f"Collaborative Filtering Recommendations for User {user_id}:")
        if not recs.empty:
            st.table(recs)
        else:
            st.write("No recommendations found.")
else:
    st.info("Please select at least one user to get recommendations.")

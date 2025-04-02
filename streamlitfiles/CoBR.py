import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

st.title("Content Based Filtering Methodology")

@st.cache_data
def load_business_data():
    business = pd.read_csv("data/business.csv")
    review = pd.read_csv("data/review.csv")
    data = pd.merge(left=review, right=business, how='left', on='business_id')
    return data

data = load_business_data()

data.rename(columns={'stars_x': 'rating', 'stars_y': 'b/s_rating'}, inplace=True)

data.fillna({'address': 'Not-Available', 
             'attributes': 'Not-Available', 
             'categories': 'Not-Available', 
             'hours': 'Not-Available'}, inplace=True)

data['location'] = data[['city', 'state', 'address']].apply(
    lambda x: f"State:{x['state']}, City:{x['city']}, Address:{x['address']} ", axis=1)

# Drop the original columns after combining
data.drop(columns=['state', 'city', 'address'], inplace=True)

# Generate user ID mapping
ids = data[['user_id']].drop_duplicates().reset_index(drop=True).copy()
ids = ids.reset_index()
data = pd.merge(data, ids, how='left', on='user_id').drop('user_id', axis=1).rename(columns={'index': 'user_id'})

# Adjust user ID to start from 1
data['user_id'] = data['user_id'] + 1

# Filter only restaurants
data = data[data['categories'].str.contains('Restaurants', na=False)].reset_index(drop=True)

# Drop irrelevant columns
cols_to_drop = ['review_id', 'useful', 'postal_code', 'funny', 'cool', 'is_open', 'date']
data.drop(columns=cols_to_drop, inplace=True)

# Ensure required columns are present
data = data[['business_id', 'name', 'categories', 'attributes', 'location', 'rating', 'user_id', 'text']]

# Safely evaluate attributes column
def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return {}

data['categories'] = data['categories'].apply(lambda x: [] if x == 'Not-Available' else x.split(', '))
data['attributes'] = data['attributes'].apply(safe_literal_eval)

# Extract price range from attributes
data['price_range'] = data['attributes'].apply(lambda x: int(x.get('RestaurantsPriceRange2', 0)) if isinstance(x, dict) else 0)

# User input for state choice
st.subheader('Select State for Recommendations')
state_choice = st.text_input('Enter the state abbreviation (e.g., PA, AZ, CA):').strip().upper()


pa_businesses = data[data['location'].str.contains(f'State:{state_choice}', case=False, na=False)]
st.write(f"Found {len(pa_businesses)} businesses in {state_choice}.")

# User input for price range
st.subheader('Select Price Range')
price_range = st.slider('Select price range (1: Lowest, 4: Highest)', min_value=1, max_value=4, value=(1, 4))

 # Filter businesses by price range
filtered_businesses = pa_businesses[(pa_businesses['price_range'] >= price_range[0]) & 
                                            (pa_businesses['price_range'] <= price_range[1])]


# Combine text features for content-based filtering
filtered_businesses['combined_features'] = filtered_businesses['categories'].astype(str) + ' ' + filtered_businesses['text']

# Vectorize text data
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filtered_businesses['combined_features'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@st.cache_data
def get_recommendations(query, cosine_sim=cosine_sim):
    try:
        idx = filtered_businesses[filtered_businesses['name'].str.contains(query, case=False, na=False)].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Exclude the business itself
        business_indices = [i[0] for i in sim_scores]
        return filtered_businesses['name'].iloc[business_indices].tolist()
    except IndexError:
        return ['No recommendations found for the given input.']


recommendations = get_recommendations(state_choice)
st.write('Top Recommended Restaurants:')
for rec in recommendations:
    st.write(f'- {rec}')
else:
    st.warning("Please enter a valid state abbreviation.")


# # Create a TF-IDF vectorizer to convert text data into numerical vectors
# tfidf = TfidfVectorizer(stop_words='english')

# # Combine relevant text features (categories and text) for content-based filtering
# pa_businesses.loc[:, 'combined_features'] = pa_businesses['categories'].astype(str) + ' ' + pa_businesses['text']


# # Fit and transform the combined features
# tfidf_matrix = tfidf.fit_transform(pa_businesses['combined_features'])

# # Compute the cosine similarity matrix
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# # Function to get recommendations based on business ID
# def get_recommendations(business_id, cosine_sim=cosine_sim):
#     # Get the index of the business
#     idx = pa_businesses[pa_businesses['business_id'] == business_id].index[0]

#     # Get the pairwise similarity scores
#     sim_scores = list(enumerate(cosine_sim[idx]))

#     # Sort the businesses based on similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the scores of the 10 most similar businesses
#     sim_scores = sim_scores[1:11]  # Exclude the business itself

#     # Get the business indices
#     business_indices = [i[0] for i in sim_scores]

#     # Return the top 10 most similar businesses
#     return pa_businesses['name'].iloc[business_indices]

# # Example usage
# business_id_to_recommend = pa_businesses['business_id'].iloc[0]
# recommendations = get_recommendations(business_id_to_recommend)
# st.write(f"Recommendations for business ID {business_id_to_recommend}")
# st.dataframe(recommendations)




# Not DEBUG YET
# # Normalize the ratings
# scaler = MinMaxScaler()
# pa_businesses['normalized_stars'] = scaler.fit_transform(pa_businesses[['rating']])
# pa_businesses['normalized_reviews'] = scaler.fit_transform(pa_businesses[['b/s_rating']])


# # Calculate the weighted score

# # Weight the cosine similarity
# weighted_sim = cosine_sim * (pa_businesses['normalized_stars'].values[:, None] * pa_businesses['normalized_reviews'].values[:, None])


# # Update indices mapping
# indices = pd.Series(pa_businesses.index, index=pa_businesses['name']).drop_duplicates()

# ## Function to get weighted recommendations
# def get_weighted_recommendations(name, weighted_sim=weighted_sim):
#     try:
#         idx = indices[name]
#         sim_scores = list(enumerate(weighted_sim[idx]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#         sim_scores = sim_scores[1:11]  # Top 10 recommendations
#         business_indices = [i[0] for i in sim_scores]
#         return pa_businesses['name'].iloc[business_indices].tolist()
#     except KeyError:
#         return [f"Restaurant '{name}' not found."]

# # User Input and Display Recommendations
# name = st.text_input("Enter a restaurant name for recommendations:")

# if st.button("Get Recommendations"):
#     if name:
#         recommendations = get_weighted_recommendations(name)
#         st.write("Here are the top recommendations:")
#         for rec in recommendations:
#             st.write(f"- {rec}")
#     else:
#         st.warning("Please enter a restaurant name.")


# def evaluate_recommendations(y_true, y_pred):
#     precision = precision_score(y_true, y_pred, average='weighted')
#     recall = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')
#     return precision, recall, f1

# # Mean Average Precision (MAP)
# def mean_average_precision(recommended, relevant):
#     ap = 0.0
#     hits = 0
#     for i, item in enumerate(recommended):
#         if item in relevant:
#             hits += 1
#             ap += hits / (i + 1)
#     return ap / len(relevant)

# # Example evaluation (assuming you have ground truth data)
# y_true = [1, 0, 1, 0, 1]  # Replace with actual relevant items
# y_pred = [1, 1, 0, 0, 1]  # Replace with actual recommended items

# precision, recall, f1 = evaluate_recommendations(y_true, y_pred)

# # Display accuracy results in Streamlit
# st.write("Evaluation Metrics:")
# accuracyresult = pd.DataFrame({
#     'Precision': [precision], 
#     'Recall': [recall], 
#     'F1 Score': [f1]
# })
# st.dataframe(accuracyresult)

# st.write(get_weighted_recommendations('Restaurant Name'))

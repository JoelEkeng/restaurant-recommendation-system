# Importing all the necessary libraries

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections


st.title("Resturant Recommendation System")

st.write("Welcome to the Resturant Recommendation System.This system uses the data of the resturants and the users to recommend resturants to the users.")
st.write("1. Content Based Filtering: This system recommends resturants based on the similarity of the resturants.")
st.write("2. Collaborative Filtering: This system recommends resturants based on the similarity of the users.")
st.write("3. Hybrid Filtering: This system recommends resturants based on the similarity of the resturants and the users.")


st.title("Data Visualizations")


business= pd.read_csv("data/business.csv")
# previewing the datasetdata.loc[ data.categories.str.contains('Restaurants')]
business.head(10)

review=pd.read_csv("data/review.csv")
# previewing the dataset
review.head(10)

tab1, tab2 = st.tabs(["Business", "Reviews"])
tab1.dataframe(business, height=450, use_container_width=True)
tab2.dataframe(review, height=450, use_container_width=True)


st.title("Merging the two datasets on the basis of business_id")
mergeData=pd.merge(left=review , right=business, how='left', on='business_id')
st.dataframe(mergeData.head(10))


st.title("Informative Description on the Data set.")
st.dataframe(mergeData.describe())


# Data Cleaning
st.title("Data Cleaning")
mergeData.rename(columns={'stars_x':'rating', 'stars_y':'b/s_rating'}, inplace=True)
# checking for columns with missing values
mergeData.isna().sum()
# Imputing missing values in the address attributes categories and hours columns with "Not-Available"
mergeData.fillna({'address': 'Not-Available', 
            'attributes': 'Not-Available', 
            'categories': 'Not-Available', 
            'hours': 'Not-Available'}, inplace=True)
# checking for duplicated columns
st.write("Duplicates: ",mergeData.duplicated().sum())

# combining the address columns
mergeData['location']=mergeData[['city','state','address']]\
            .apply( lambda x: f"State:{x['state']}, City:{x['city']}, Address:{x['address']} ", axis=1)

# then we drop the combined columns
mergeData.drop(columns=['state', 'city','address'], axis=1, inplace=True)



# converting the user_id into intergers

# selecting only the unique user ids as a dataframe
ids=mergeData[['user_id']].drop_duplicates('user_id').reset_index(drop=True).copy()
ids=ids.reset_index()

# merging the ids dataframe with our original dataframe using the user id column as primary key
# renaming the index column to represent the user ids
mergeData=pd.merge(mergeData,ids, how='left', on='user_id').drop('user_id', axis=1).rename(columns={'index':'user_id'})

def add(x):
    """ adds 1 to the existing user id"""
    y=x+1
    return y
mergeData.user_id=mergeData.user_id.apply(add ) 
st.write("Table after cleaning the Merged DataSet")
st.dataframe(mergeData.head())


# creating a function to extract the price values
def Price(val):
    """
    The function takes in a dictionary as input and extracts the price in the 'RestaurantsPriceRange2' key, else returns a '0'
    if the value if 'Not-Available'
    """
    try:
        p = eval(val)['RestaurantsPriceRange2']    
        return int(p)                              
    except:
        return 0                                   
    
# applying the function to the attributes column
mergeData['price']=mergeData.attributes.apply(Price)

# previewing the column
st.write("Price Column after cleaning")
st.table(mergeData[['price']].head())


# selecting only the restaurants
mergeData=mergeData.loc[ mergeData.categories.str.contains('Restaurants')].copy().reset_index(drop=True)
# droping irrelevant columns
cols=['review_id', 'useful','postal_code','funny', 'cool', 'is_open', 'date']
mergeData.drop(columns=cols, axis=1, inplace=True)  


column1 = 'rating'
column2 = 'b/s_rating'


st.title("Histogram of User ratings and Business ratings")

st.subheader("Histograms")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))  
 
sns.countplot(data=mergeData, x=column1 ,ax=axes[0] , color='tab:blue')
axes[0].set_xlabel("User ratings")
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Histogram of User ratings')

sns.countplot(data=mergeData, x=column2 ,ax=axes[1] ,color='tab:red')
axes[1].set_xlabel("Business ratings")
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Histogram of Business ratings')


plt.tight_layout()
st.pyplot(fig)


st.title("Correlation between User ratings and Business ratings")
df_1 = mergeData[['rating', 'b/s_rating', 'review_count']]
st.dataframe(df_1.corr())


st.title("Top 10 Restaurant Categories");
categories=[ cat for category in mergeData.drop_duplicates('business_id').categories for cat in category.split(',')]

categories=collections.Counter(categories)
common=categories.most_common(12)

fig, ax=plt.subplots(figsize=(8,5))
x=[ i[0] for i in common[2:]]
y=[i[1] for i in common[2:]]
sns.barplot(x=x, y=y, color='tab:blue', ax=ax)
ax.set_xlabel("Categories")
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Counts")
ax.set_ylim([300,900])
ax.bar_label( ax.containers[0], padding=3, fmt='{:,}');

st.pyplot(fig)


st.title("City Counts");

location= mergeData.drop_duplicates('business_id')[['location']]
city=location.location.apply(lambda x: x.split(',')[1].replace("City:",'')) 
city=collections.Counter(city)
city=city.most_common(10)

y=[i[0] for i in city]
x=[i[1] for i in city]

fig, ax=plt.subplots(figsize=(7,5))
sns.barplot( y=y, x=x, color='tab:blue')
ax.set_ylabel("City Names")
ax.set_xlabel("Count")
ax.bar_label( ax.containers[0],padding=3, fmt='{:,}');
st.pyplot(fig)


st.title("State Counts");
states=location.location.apply(lambda x: x.split(',')[0].replace("State:",'')) 
states=collections.Counter(states)
states=states.most_common(10)

y=[i[0] for i in states]
x=[i[1] for i in states]


fig, ax=plt.subplots(figsize=(8,4))
sns.barplot( y=x,x=y , color='tab:blue')
ax.set_ylabel("States")
ax.set_xlabel("Count")
ax.bar_label( ax.containers[0],padding=3, fmt='{:,}');
st.pyplot(fig)



# most popular restaurants
st.title("Most Popular Restaurants");

index=mergeData.drop_duplicates(subset='business_id').sort_values(by=['review_count','b/s_rating'],ascending=False)[:10].index

fig, ax=plt.subplots(figsize=(7,5))
sns.barplot(data=mergeData.loc[index], x="review_count", y='name', color='tab:blue')
ax.set_ylabel("Resturant Names")
ax.set_xlabel("Review Counts")
ax.set_xlim([1500,4600])
ax.bar_label( ax.containers[0],padding=3, fmt='{:,}');

st.pyplot(fig)


# Data preprocessing: Drop duplicate businesses and extract prices
st.title("Business Rating vs. Price Range")

df = mergeData.drop_duplicates('business_id')


print("Number of businesses without price: ", sum([1 for i in df.price.values if i == 0]))

# Create a box plot
fig, ax = plt.subplots(figsize=(16, 6))
sns.boxplot(x='b/s_rating', y='price', data=df, ax=ax,color='tab:blue')
ax.set_xlabel("Business Rating")
ax.set_ylabel("Price")
ax.set_yticks([0, 1, 2, 3, 4, 5])

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title("Business Rating vs. Price Range")

st.pyplot(fig)


# 2D histogram of business rating against number of reviews
st.title("Business Rating agaisnt Number of Reviews in 2D Histogram")
fig, ax = plt.subplots(figsize=(16, 5))

# Create a 2D histogram
hist = ax.hist2d(mergeData['review_count'], mergeData['b/s_rating'], bins=25)

ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Rating")
ax.set_title('Business Rating Against Number of Reviews (2D Histogram)')

# Add a colorbar to indicate the count per bin
cb = plt.colorbar(hist[3], ax=ax)
cb.set_label('Counts')

st.pyplot(fig)

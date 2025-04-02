import streamlit as st

# Define the pages
main_page = st.Page("streamlitfiles/main.py", title="Exploartory Data Analysis")
Contentpage = st.Page("streamlitfiles/CoBR.py", title="Content Based Recommendation")
Collaborativepage = st.Page("streamlitfiles/CBR.py", title="Collaborative Filtering")
Hybridpage = st.Page("streamlitfiles/HRR.py", title="Hybrid Filtering")


# Set up navigation
pg = st.navigation([main_page, Contentpage, Collaborativepage, Hybridpage])

# Run the selected page
pg.run()
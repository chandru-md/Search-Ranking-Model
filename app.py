import streamlit as st
from pipelines.search_pipeline import search

# Page config
st.set_page_config(page_title="Search Ranking System", layout="centered")

# Title
st.title("🔍 Search Ranking System")
st.write("Enter a query to get ranked results using ML")

# Input box
query = st.text_input("Enter your search query:")

# Button
if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a query")
    else:
        results = search(query)

        st.subheader("Top Results")

        for i, (doc, score) in enumerate(results, start=1):
            st.markdown(f"""
            ### {i}. {doc}
            Score: {float(score):.2f}
            """)
            st.markdown("---")
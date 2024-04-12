import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to compute cosine similarity
def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_sim


# Streamlit app
def calculating_simularity():
    # Inputs
    st.header("Text Input")
    text1 = st.text_input("Enter Text 1", "Text 1")
    text2 = st.text_input("Enter Text 2", "Text 2")

    # Compute cosine similarity
    cosine_sim = compute_cosine_similarity(text1, text2)
    st.subheader("Cosine Similarity")
    st.write(cosine_sim)

st.set_page_config(page_title="Cosine Simularity Calculation", page_icon="🧮")
st.markdown("# Cosine Simularity")
st.sidebar.header("Cosine Simularity")
st.write(
    """Some text... Bla Bla Bla... 2"""
)

calculating_simularity()

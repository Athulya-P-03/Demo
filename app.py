import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# LOAD TRAINED FILES
# -----------------------------
@st.cache_data
def load_movies():
    return pd.read_csv("movies_cleaned.csv")  # From Jupyter Notebook

movies = load_movies()

@st.cache_resource
def load_similarity():
    with open("similarity.pkl", "rb") as f:
        return pickle.load(f)

similarity_matrix = load_similarity()


# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend_movie(movie_name):

    movie_name = movie_name.lower().strip()

    if movie_name not in movies["title_clean"].values:
        return None

    idx = movies[movies["title_clean"] == movie_name].index[0]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in scores[1:11]]

    results = movies.iloc[top_indices][["title", "genres", "imdbId"]].copy()
    results["watch_link"] = results["imdbId"].apply(
        lambda x: f"https://www.imdb.com/title/tt{int(x):07d}/"
        if not np.isnan(x) else "N/A"
    )
    return results


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎬 Movie Recommendation System")

user_movie = st.text_input("Enter Movie Name")

if st.button("Get Recommendations"):
    if user_movie.strip() == "":
        st.warning("Please enter a movie name!")
    else:
        recs = recommend_movie(user_movie)

        if recs is None:
            st.error("Movie not found!")
        else:
            st.subheader("Recommended Movies:")
            for idx, row in recs.iterrows():
                col1, col2 = st.columns([3,1])
                with col1:
                    st.write(f"**{row['title']}**")
                    st.write(f"{row['genres']}")
                with col2:
                    st.markdown(
                        f'<a href="{row["watch_link"]}" target="_blank">'
                        f'<button style="padding:8px;">Play </button></a>',
                        unsafe_allow_html=True,
                    )
                st.markdown("---")

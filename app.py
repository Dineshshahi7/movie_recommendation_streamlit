import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Load dataset
# --------------------------
# Use the uploaded movies.csv path
DF_PATH = "/mnt/data/movies.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(r"C:\\Users\\Swastikpc\\Movie_Recommendation_System\\movies.csv")
    return df

@st.cache_data
def build_similarity(df):
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        df[feature] = df[feature].fillna('')

    combined_features = (
        df['genres'].astype(str) + ' ' +
        df['keywords'].astype(str) + ' ' +
        df['tagline'].astype(str) + ' ' +
        df['cast'].astype(str) + ' ' +
        df['director'].astype(str)
    )

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    return similarity, vectorizer

try:
    df = load_data(DF_PATH)
except FileNotFoundError:
    st.error(f"movies.csv not found at {DF_PATH}. Make sure the file is in that path.")
    st.stop()

similarity, _ = build_similarity(df)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie and get similar movies using content-based filtering.")

movie_name = st.text_input("Enter Your Favorite Movie:")

if st.button("Recommend"):
    if not movie_name or movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        list_of_all_titles = df['title'].astype(str).tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1, cutoff=0.5)

        if len(find_close_match) == 0:
            st.error("Movie not found. Try another movie or check spelling.")
        else:
            close_match = find_close_match[0]
            # Handle case where index column might be present or not
            if 'index' in df.columns:
                index_of_the_movie = int(df[df.title == close_match]['index'].values[0])
            else:
                index_of_the_movie = int(df[df.title == close_match].index[0])

            similarity_score = list(enumerate(similarity[index_of_the_movie]))
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

            st.subheader("Recommended Movies ✔️")

            i = 1
            for movie in sorted_similar_movies:
                index = movie[0]
                title_from_index = df.iloc[index]['title']

                if i <= 10:
                    st.write(f"{i}. {title_from_index}")
                    i += 1

# Footer / tips
st.markdown("---")
st.caption("Tip: If your dataset is large, the first run may take a few seconds to compute the TF-IDF and similarity matrix.")

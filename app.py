import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="MovieAddict",
    page_icon="🎬",
    layout="wide"
)

# ==================================================
# LOAD ARTIFACTS
# ==================================================
@st.cache_resource
def load_artifacts():
    movies = pickle.load(open("movies.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    X = pickle.load(open("count_matrix.pkl", "rb"))
    return movies, vectorizer, X

movies, vectorizer, X = load_artifacts()

# ==================================================
# CORE RECOMMENDATION FUNCTIONS
# ==================================================
def recommend_similar_movies(title, top_n=8):
    title = title.lower().strip()
    idx = movies[movies["title_clean"] == title].index[0]
    scores = cosine_similarity(X[idx], X).flatten()
    similar_indices = np.argsort(scores)[::-1][1: top_n + 1]
    return movies.iloc[similar_indices]["title"].tolist()


def search_by_description(query, top_n=8):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, X).flatten()
    idx = scores.argsort()[::-1][:top_n]
    return movies.iloc[idx]["title"].tolist()


def search_by_genre(genre, top_n=8):
    mask = movies["genres"].str.contains(genre, case=False, na=False)
    return movies[mask]["title"].head(top_n).tolist()


def popular_movies(top_n=8):
    if "popularity" in movies.columns:
        return (
            movies.sort_values("popularity", ascending=False)
            ["title"]
            .head(top_n)
            .tolist()
        )
    return movies["title"].head(top_n).tolist()


def top_movies_by_genre(genre, top_n=10, sort_col="popularity"):
    mask = movies["genres"].str.contains(genre, case=False, na=False)

    if sort_col in movies.columns:
        return (
            movies[mask]
            .sort_values(sort_col, ascending=False)["title"]
            .head(top_n)
            .tolist()
        )

    return movies[mask]["title"].head(top_n).tolist()

# ==================================================
# STYLING
# ==================================================
st.markdown(
    """
    <style>
    body { background-color:#0f0f0f; }
    .card {
        background:#181818;
        padding:10px;
        border-radius:8px;
        margin-bottom:8px;
        font-size:14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# HEADER
# ==================================================
st.markdown("## 🎬 MoViE-AdDict")
st.caption("A content-based movie recommender built using NLP & cosine similarity")
st.caption("Internship Project")

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.markdown("### 📂 Browse")

mode = st.sidebar.radio(
    "Mode",
    [
        "Home (Personalised Rows)",
        "Similar Movies (Content-Based)",
        "Search (Description / Genre)",
        "Popular & Trending",
        "Top Movies by Genre",
    ],
)

top_n = st.sidebar.slider("Results", 5, 15, 8)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Team:** Golden Dawn  
    **Creators:**  
    - Abhishek Kumar  
    - Aditya Raj  
    - Adarsh Yadav  

    GitHub:  
    https://github.com/abhishekkumar177/movie-recommender-streamlit
    """
)

# ==================================================
# MAIN CONTENT
# ==================================================
if mode == "Home (Personalised Rows)":
    st.subheader("Continue Watching")
    sample = movies["title"].sample(top_n, random_state=1).tolist()
    for m in sample:
        st.markdown(f"<div class='card'>🎬 {m}</div>", unsafe_allow_html=True)

    st.subheader("Action Picks")
    for m in search_by_genre("Action", top_n):
        st.markdown(f"<div class='card'>🎬 {m}</div>", unsafe_allow_html=True)

elif mode == "Similar Movies (Content-Based)":
    movie = st.selectbox("Select a movie", movies["title"].values)
    if st.button("Recommend"):
        with st.spinner("Finding similar movies..."):
            for m in recommend_similar_movies(movie, top_n):
                st.markdown(f"<div class='card'>🎬 {m}</div>", unsafe_allow_html=True)

elif mode == "Search (Description / Genre)":
    query = st.text_input("Search by description or genre")
    search_type = st.selectbox("Search type", ["Description", "Genre"])

    if query:
        with st.spinner("Searching..."):
            if search_type == "Description":
                results = search_by_description(query, top_n)
            else:
                results = search_by_genre(query, top_n)

        for m in results:
            st.markdown(f"<div class='card'>🎬 {m}</div>", unsafe_allow_html=True)

elif mode == "Top Movies by Genre":
    st.subheader("🎭 Top Movies by Genre")

    genre_list = [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Drama", "Family", "Fantasy", "Horror", "Mystery",
        "Romance", "Science Fiction", "Thriller", "War", "Western"
    ]

    selected_genre = st.selectbox("Choose a genre", genre_list)
    st.markdown(f"### 🔥 Top {top_n} {selected_genre} Movies")

    results = top_movies_by_genre(selected_genre, top_n)

    if results:
        for m in results:
            st.markdown(f"<div class='card'>🎬 {m}</div>", unsafe_allow_html=True)
    else:
        st.warning("No movies found for this genre.")

else:
    st.subheader("Popular & Trending")
    for m in popular_movies(top_n):
        st.markdown(f"<div class='card'>🎬 {m}</div>", unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("Built with CountVectorizer & Cosine Similarity • Streamlit")

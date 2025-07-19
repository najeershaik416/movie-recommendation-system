# app.py

import streamlit as st
import pandas as pd
from recommender import MovieRecommender

# --- Page Configuration ---
st.set_page_config(
    page_title="CineSuggest Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- Caching the Recommender Object ---
@st.cache_resource
def load_recommender():
    """Load the MovieRecommender and cache it."""
    try:
        recommender = MovieRecommender(
            movies_path='movie.csv',
            ratings_path='rating.csv'
        )
        return recommender
    except FileNotFoundError:
        st.error("Error: Dataset files not found. Please ensure 'movie.csv' and 'rating.csv' are in a 'data' subfolder.")
        return None

# --- Load Data ---
recommender = load_recommender()

# --- Main App ---
if recommender:
    st.title("🎬 CineSuggest: Your Personal Movie Guide")
    st.markdown("Discover movies tailored to your taste. Select a recommendation type from the sidebar to get started.")

    # --- Sidebar ---
    st.sidebar.header("Recommendation Options")
    recommendation_type = st.sidebar.radio(
        "Choose your recommendation style:",
        (
            "1. For First-Time Visitors (Popular Movies)",
            "2. For Registered Users (History-Based)",
            "3. Custom Filter (Rule-Based)",
            "4. Find Similar Movies (Query-Based)",
            "5. AI-Powered Search (Natural Language)"
        ),
        key="recommendation_type"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("The AI-Powered Search uses the DeepSeek model via OpenRouter to understand your requests.")

    # --- Display Recommendations Based on Choice ---
    
    if "1." in recommendation_type:
        st.header("🌟 Top Picks for Everyone")
        st.markdown("These are the most popular and highly-rated movies on the platform. A great place to start!")
        if st.button("Show Popular Movies"):
            with st.spinner("Finding the most popular movies..."):
                recommendations = recommender.recommend_for_fresh_users(top_n=12)
                st.dataframe(recommendations, use_container_width=True, hide_index=True)

    elif "2." in recommendation_type:
        st.header("👤 Recommendations Just for You")
        st.markdown("Based on movies you've watched and rated highly (4+ stars).")
        user_id = st.number_input("Enter your User ID (e.g., 1 to 138493)", min_value=1, max_value=138493, value=1, step=1)
        if st.button(f"Get Recommendations for User {user_id}"):
            with st.spinner("Analyzing your watch history..."):
                recommendations = recommender.recommend_for_existing_users(user_id=user_id, top_n=12)
                if recommendations is not None and not recommendations.empty:
                    st.success(f"Here are some movies you might like, User {user_id}:")
                    st.dataframe(recommendations, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"Could not generate recommendations for User {user_id}. They may have no highly-rated movies in their history.")

    elif "3." in recommendation_type:
        st.header("🔧 Custom Filtered Search")
        st.markdown("Fine-tune your search for popular movies with specific rules.")
        col1, col2 = st.columns(2)
        with col1:
            min_year = st.slider("Show movies released after year:", min_value=1900, max_value=2024, value=2000)
        with col2:
            exclude_genres = st.multiselect("Exclude these genres:", options=recommender.genres)
        if st.button("Find Movies with These Rules"):
            with st.spinner("Applying your custom filters..."):
                recommendations = recommender.recommend_rule_based(exclude_genres=exclude_genres, min_year=min_year, top_n=12)
                st.dataframe(recommendations, use_container_width=True, hide_index=True)

    elif "4." in recommendation_type:
        st.header("🔍 Find Movies Like...")
        st.markdown("Tell us a movie you like, and we'll find similar ones. You can even ask for a twist!")
        all_movie_titles = recommender.get_all_movies()
        movie_title = st.selectbox("Select a movie you like:", options=all_movie_titles)
        more_of_genre = st.selectbox("Optional: I want it to be more...", options=[None] + recommender.genres, format_func=lambda x: "No preference" if x is None else x)
        if st.button(f"Find movies like '{movie_title}'"):
            with st.spinner(f"Finding movies similar to '{movie_title}'..."):
                recommendations = recommender.recommend_query_based(movie_title=movie_title, top_n=12, more_of_genre=more_of_genre)
                if isinstance(recommendations, pd.DataFrame):
                    st.dataframe(recommendations, use_container_width=True, hide_index=True)
                else:
                    st.error(recommendations)

    elif "5." in recommendation_type:
        st.header("🤖 AI-Powered Natural Language Search")
        st.markdown("Just tell us what you're looking for in plain English!")
        default_query = "movie like Interstellar but more romantic"
        user_query = st.text_input("Your request:", value=default_query)
        if st.button("✨ Find Movies with AI"):
            if "DEEPSEEK_API_KEY" not in st.secrets or not st.secrets["DEEPSEEK_API_KEY"]:
                st.error("API key not found. Please add your DeepSeek/OpenRouter API key to your Streamlit secrets (.streamlit/secrets.toml).")
            else:
                api_key = st.secrets["DEEPSEEK_API_KEY"]
                with st.spinner("🧠 Asking the AI for recommendations... this may take a moment."):
                    recommendations, explanation = recommender.recommend_natural_language(user_query, api_key)
                if recommendations is not None:
                    st.success("Here's what I found for you!")
                    st.info(f"🤖 *AI Explanation:* {explanation}")
                    st.dataframe(recommendations, use_container_width=True, hide_index=True)
                else:
                    st.error(explanation)
else:
    st.error("Recommender system could not be initialized. Please check the file paths and try again.")
# recommender.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
import json

class MovieRecommender:
    """
    A class to handle movie data loading and provide various types of recommendations,
    including AI-powered query parsing.
    """
    def __init__(self, movies_path, ratings_path):
        """
        Initializes the recommender, loads data, and precomputes necessary matrices.
        
        Args:
            movies_path (str): Path to the movies.csv file.
            ratings_path (str): Path to the ratings.csv file.
        """
        self.movies_df = None
        self.ratings_df = None
        self.cosine_sim = None
        self.indices = None
        self.genres = None
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        self._load_data(movies_path, ratings_path)
        self._preprocess_data()
        self._build_similarity_matrix()

    def _load_data(self, movies_path, ratings_path):
        """Loads movie and rating data from CSV files."""
        print("Loading data...")
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
        print("Data loaded successfully.")

    def _preprocess_data(self):
        """Preprocesses the movie data."""
        print("Preprocessing data...")
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
        self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce').fillna(0).astype(int)
        self.movies_df['title_clean'] = self.movies_df['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)$', '', x).strip())
        
        # Create a lowercased version of title for matching
        self.movies_df['title_lower'] = self.movies_df['title_clean'].str.lower()
        
        self.movies_df['genres'] = self.movies_df['genres'].str.replace('|', ' ', regex=False)
        
        all_genres = set()
        self.movies_df['genres'].str.split(' ').apply(all_genres.update)
        self.genres = sorted([g for g in all_genres if g and g != '(no genres listed)'])
        print("Preprocessing complete.")

    def _build_similarity_matrix(self):
        """Builds a TF-IDF matrix for genres and computes the cosine similarity."""
        print("Building similarity matrix...")
        self.movies_df = self.movies_df.reset_index(drop=True)
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['genres'])
        
        self.cosine_sim = cosine_similarity(tfidf_matrix)
        
        # Create a mapping from lowercased title to index
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title_lower'])
        print("Similarity matrix built successfully.")
        
    def _call_llm_api(self, prompt, api_key):
        """Helper function to call the LLM API."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful movie assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {e}")
            return None

    def _parse_query_with_llm(self, user_query, api_key):
        """Uses an LLM to parse a natural language query into structured JSON."""
        prompt = f"""
        You are a movie assistant. Extract the key elements from the following query: "{user_query}"
        Analyze the query to identify the main reference movie and any modifications like genre additions or tonal shifts.
        Infer a list of relevant genres. If the user says "more romantic", 'Romance' should be a key genre. If "less horror", 'Horror' should be an excluded genre.
        Return a JSON object with the following structure. Do NOT add any text before or after the JSON object.
        {{
          "reference_movie": "The main movie title mentioned",
          "add_genres": ["List of genres to prioritize"],
          "exclude_genres": ["List of genres to avoid"],
          "explanation_context": "A very brief summary of the user's request, e.g., 'more romantic' or 'with a space theme'"
        }}
        """
        response_json = self._call_llm_api(prompt, api_key)
        if response_json:
            try:
                content = response_json["choices"][0]["message"]["content"]
                cleaned_content = re.sub(r'json\n|\n', '', content).strip()
                return json.loads(cleaned_content)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Failed to parse LLM response: {e}\nContent: {content}")
                return None
        return None

    def _generate_explanation_with_llm(self, movie_list, parsed_data, api_key):
        """Generates a brief explanation for the recommendations."""
        titles = ", ".join(movie_list)
        prompt = f"""
        A user wanted recommendations for movies like "{parsed_data.get('reference_movie', 'a specific movie')}" with the preference "{parsed_data.get('explanation_context', 'no specific preference')}".
        Based on this, you recommended the following movies: {titles}.
        Write a short, friendly, and insightful explanation (2-3 sentences) for why these movies are a good match for the user's request.
        """
        response_json = self._call_llm_api(prompt, api_key)
        if response_json:
            try:
                return response_json["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError):
                return "Could not generate an explanation at this time."
        return "Could not generate an explanation at this time."

    def get_all_movies(self):
        """Returns a sorted list of all movie titles."""
        return self.movies_df['title_clean'].sort_values().unique().tolist()

    def recommend_for_fresh_users(self, top_n=10):
        """Recommends movies based on popularity."""
        rating_counts = self.ratings_df.groupby('movieId').agg(rating_count=('rating', 'count'), avg_rating=('rating', 'mean')).reset_index()
        min_ratings = 50 
        popular_movies = rating_counts[rating_counts['rating_count'] >= min_ratings]
        popular_movies = popular_movies.merge(self.movies_df[['movieId', 'title_clean', 'genres', 'year']], on='movieId')
        return popular_movies.sort_values('avg_rating', ascending=False).head(top_n)

    def recommend_for_existing_users(self, user_id, top_n=10):
        """Recommends movies based on a user's high-rated watch history."""
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        if user_ratings.empty: return None
        
        liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId']
        if liked_movies.empty: return None

        # Get the indices of the liked movies from the main movies_df
        liked_indices = self.movies_df[self.movies_df['movieId'].isin(liked_movies)].index.tolist()
        if not liked_indices: return None

        avg_sim_scores = self.cosine_sim[liked_indices].mean(axis=0)
        sim_scores_series = pd.Series(avg_sim_scores, index=self.movies_df.index)
        
        top_indices = sim_scores_series.sort_values(ascending=False).index
        
        recommended_movies = self.movies_df.loc[top_indices]
        watched_movie_ids = user_ratings['movieId'].unique()
        unwatched_recs = recommended_movies[~recommended_movies['movieId'].isin(watched_movie_ids)]
        
        return unwatched_recs[['title_clean', 'genres', 'year']].head(top_n)

    def recommend_rule_based(self, exclude_genres=None, min_year=None, top_n=10):
        """Recommends popular movies based on filtering rules."""
        filtered_movies = self.movies_df.copy()
        if min_year:
            filtered_movies = filtered_movies[filtered_movies['year'] >= min_year]
        if exclude_genres:
            for genre in exclude_genres:
                filtered_movies = filtered_movies[~filtered_movies['genres'].str.contains(genre, case=False, na=False)]
        
        return self.recommend_for_fresh_users(top_n=500).merge(filtered_movies[['movieId']], on='movieId').head(top_n)

    def recommend_query_based(self, movie_title, top_n=10, more_of_genre=None):
        """Recommends movies similar to a given movie, with an optional genre boost."""
        if movie_title.lower() not in self.indices:
            return f"Movie '{movie_title}' not found in the dataset."
        
        idx = self.indices[movie_title.lower()]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n*5] # Larger pool for re-ranking

        candidate_indices = [i[0] for i in sim_scores]
        result_df = self.movies_df.iloc[candidate_indices].copy()
        result_df['similarity_score'] = [s[1] for s in sim_scores]

        if more_of_genre and more_of_genre in self.genres:
            boost_factor = 1.5
            result_df['final_score'] = result_df.apply(lambda row: row['similarity_score'] * boost_factor if more_of_genre in row['genres'] else row['similarity_score'], axis=1)
            result_df = result_df.sort_values('final_score', ascending=False)
        else:
            result_df = result_df.sort_values('similarity_score', ascending=False)

        return result_df[['title_clean', 'genres', 'year']].head(top_n)

    def recommend_natural_language(self, user_query, api_key, top_n=10):
        """Orchestrates the natural language recommendation process."""
        parsed_data = self._parse_query_with_llm(user_query, api_key)
        if not parsed_data: return None, "Sorry, I couldn't understand your request. Please try rephrasing it."

        ref_movie_title = parsed_data.get("reference_movie")
        if not ref_movie_title or ref_movie_title.lower() not in self.indices:
            return None, f"Sorry, the movie '{ref_movie_title}' was not found in our database."
        
        idx = self.indices[ref_movie_title.lower()]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        add_genres = [g.lower() for g in parsed_data.get("add_genres", [])]
        exclude_genres = [g.lower() for g in parsed_data.get("exclude_genres", [])]
        
        candidate_movies = []
        for i, score in sim_scores[1:]:
            movie_data = self.movies_df.iloc[i]
            movie_genres_lower = movie_data['genres'].lower()
            
            if any(ex_g in movie_genres_lower for ex_g in exclude_genres): continue
            
            boost = 1.0
            for add_g in add_genres:
                if add_g in movie_genres_lower: boost += 0.5 
            
            final_score = score * boost
            candidate_movies.append((i, final_score))

        candidate_movies = sorted(candidate_movies, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in candidate_movies[:top_n]]
        recommendations = self.movies_df.iloc[top_indices][['title_clean', 'genres', 'year']]
        
        if recommendations.empty: return None, "I couldn't find any movies matching your specific criteria."

        explanation = self._generate_explanation_with_llm(recommendations['title_clean'].tolist(), parsed_data, api_key)
        return recommendations, explanation
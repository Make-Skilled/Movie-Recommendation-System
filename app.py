from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import pandas as pd
from model import MovieRecommender
from flask_cors import CORS
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load the model
try:
    logger.info("Loading model and data...")
    if not os.path.exists('movie_recommender.pkl'):
        logger.error("Model file not found. Please run model.py first.")
        raise FileNotFoundError("movie_recommender.pkl not found")
    
    with open('movie_recommender.pkl', 'rb') as f:
        recommender = pickle.load(f)
    
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Mock user data (replace with database in production)
user_data = {
    'watched_movies': ['The Dark Knight', 'Inception', 'The Matrix'],
    'favorite_movies': ['The Dark Knight', 'Inception'],
    'watchlist': ['The Shawshank Redemption', 'Pulp Fiction'],
    'genre_preferences': {
        'Action': 85,
        'Drama': 70,
        'Sci-Fi': 65,
        'Thriller': 60
    }
}

def format_movie_data(movie):
    """Format movie data for API response"""
    try:
        return {
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'rating': float(movie['rating']) if 'rating' in movie and pd.notna(movie['rating']) else None,
            'review': movie['review'] if 'review' in movie and pd.notna(movie['review']) else None,
            'poster_url': f'https://via.placeholder.com/100?text={movie["title"][:2]}'  # Mock poster
        }
    except Exception as e:
        logger.error(f"Error formatting movie data: {str(e)}")
        return None

@app.route('/')
def home():
    try:
        # Get popular movies for the homepage
        popular_movies = recommender.get_popular_movies(10)
        highest_rated = recommender.get_highest_rated_movies(10)
        
        # Convert to list of dictionaries for template
        popular_list = []
        for title, (rating, num_ratings) in popular_movies.iterrows():
            popular_list.append({
                'title': title,
                'rating': float(rating),
                'num_ratings': int(num_ratings)
            })
            
        highest_rated_list = []
        for title, (rating, num_ratings) in highest_rated.iterrows():
            highest_rated_list.append({
                'title': title,
                'rating': float(rating),
                'num_ratings': int(num_ratings)
            })
        
        return render_template('index.html', 
                             popular_movies=popular_list,
                             highest_rated=highest_rated_list)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        return render_template('index.html', 
                             popular_movies=[],
                             highest_rated=[],
                             error="Error loading movies")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        movie_name = request.form.get('movie_name')
        logger.info(f"Getting recommendations for movie: {movie_name}")
        
        if not movie_name:
            return render_template('recommendations.html', 
                                 movie_name='',
                                 recommendations=[],
                                 error='Please provide a movie name')
        
        # Get recommendations
        recommendations = recommender.get_recommendations(movie_name)
        
        if recommendations is None or recommendations.empty:
            return render_template('recommendations.html',
                                 movie_name=movie_name,
                                 recommendations=[],
                                 error='Movie not found')
        
        # Format recommendations for display
        formatted_recommendations = []
        for _, row in recommendations.iterrows():
            formatted_recommendations.append({
                'title': str(row['title']),  # Ensure title is a string
                'correlation': float(row['Correlation']),
                'type': str(row['type'])  # Ensure type is a string
            })
        
        logger.info(f"Found {len(formatted_recommendations)} recommendations")
        return render_template('recommendations.html',
                             movie_name=movie_name,
                             recommendations=formatted_recommendations)
                             
    except Exception as e:
        logger.error(f"Error in recommend route: {str(e)}")
        return render_template('recommendations.html',
                             movie_name=movie_name if 'movie_name' in locals() else '',
                             recommendations=[],
                             error=f"Error getting recommendations: {str(e)}")

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    try:
        return send_from_directory('static', path)
    except Exception as e:
        logger.error(f"Error serving static file {path}: {str(e)}")
        return str(e), 404

@app.route('/api/movies', methods=['GET'])
def get_all_movies():
    try:
        # Return list of all movies with basic info
        movies_list = []
        for title, (rating, num_ratings) in recommender.ratings.iterrows():
            movies_list.append({
                'title': title,
                'rating': float(rating),
                'num_ratings': int(num_ratings)
            })
        return jsonify({'movies': movies_list})
    
    except Exception as e:
        logger.error(f"Error getting all movies: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    try:
        movie = recommender.ratings[recommender.ratings['movieId'] == movie_id]
        if movie.empty:
            return jsonify({'error': 'Movie not found'}), 404
        
        return jsonify(format_movie_data(movie.iloc[0]))
    
    except Exception as e:
        logger.error(f"Error getting movie details: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/stats', methods=['GET'])
def get_user_stats():
    try:
        stats = {
            'total_movies': len(recommender.ratings),
            'watched_movies': len(user_data['watched_movies']),
            'favorite_movies': len(user_data['favorite_movies']),
            'watchlist_movies': len(user_data['watchlist'])
        }
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/recommendations', methods=['GET'])
def get_user_recommendations():
    try:
        # Get recommendations based on user's favorite movies
        favorite_movie = user_data['favorite_movies'][0]
        recommendations = recommender.get_recommendations(favorite_movie)
        
        formatted_recommendations = [
            format_movie_data(movie) for _, movie in recommendations.iterrows()
        ]
        
        return jsonify({'recommendations': formatted_recommendations})
    
    except Exception as e:
        logger.error(f"Error getting user recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/recently-watched', methods=['GET'])
def get_recently_watched():
    try:
        # Get recently watched movies with details
        recently_watched = []
        for movie_title in user_data['watched_movies']:
            movie = recommender.ratings[recommender.ratings['title'] == movie_title]
            if not movie.empty:
                recently_watched.append(format_movie_data(movie.iloc[0]))
        
        return jsonify({'movies': recently_watched})
    
    except Exception as e:
        logger.error(f"Error getting recently watched movies: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/genre-preferences', methods=['GET'])
def get_genre_preferences():
    try:
        preferences = [
            {'genre': genre, 'percentage': percentage}
            for genre, percentage in user_data['genre_preferences'].items()
        ]
        return jsonify({'preferences': preferences})
    
    except Exception as e:
        logger.error(f"Error getting genre preferences: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_movies():
    try:
        query = request.args.get('q', '').lower()
        if not query:
            return jsonify({'results': []})
        
        # Search movies by title
        matching_movies = recommender.ratings[
            recommender.ratings.index.str.lower().str.contains(query)
        ].head(5)
        
        results = []
        for title, (rating, num_ratings) in matching_movies.iterrows():
            results.append({
                'title': title,
                'rating': float(rating),
                'num_ratings': int(num_ratings)
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"Error searching movies: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
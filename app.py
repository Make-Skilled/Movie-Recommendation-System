from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import pandas as pd
from model import MovieRecommender
from telugu_knn_model import TeluguMovieRecommender
from flask_cors import CORS
import numpy as np
import os
import logging
import sqlite3
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load the models
try:
    logger.info("Loading models and data...")
    if not os.path.exists('movie_recommender.pkl'):
        logger.error("Model file not found. Please run model.py first.")
        raise FileNotFoundError("movie_recommender.pkl not found")
    
    with open('movie_recommender.pkl', 'rb') as f:
        recommender = pickle.load(f)
    
    # Initialize and load Telugu movie recommender
    telugu_recommender = TeluguMovieRecommender()
    if not os.path.exists('telugu_knn_model.pkl'):
        logger.info("Training Telugu movie KNN model...")
        from telugu_knn_model import train_and_save_model
        train_and_save_model()
    telugu_recommender.load_model()
    
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
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

@app.route('/api/telugu/recommend', methods=['POST'])
def telugu_recommend():
    """API endpoint for KNN-based Telugu movie recommendations"""
    try:
        data = request.get_json()
        movie_title = data.get('movie_title')
        n_recommendations = data.get('n_recommendations', 5)
        
        if not movie_title:
            return jsonify({'error': 'Movie title is required'}), 400
        
        recommendations = telugu_recommender.get_recommendations(
            movie_title, 
            n_recommendations=n_recommendations
        )
        
        if recommendations is None:
            return jsonify({'error': 'Movie not found'}), 404
        
        return jsonify({
            'movie_title': movie_title,
            'recommendations': recommendations
        })
    
    except Exception as e:
        logger.error(f"Error in telugu_recommend API: {str(e)}")
        return jsonify({'error': str(e)}), 500

def init_db():
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    with open('schema.sql') as f:
        c.executescript(f.read())
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect('movies.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/review', methods=['POST'])
def add_review():
    try:
        data = request.json
        movie_title = data.get('movie_title')
        user_name = data.get('user_name')
        rating = float(data.get('rating'))
        comment = data.get('comment', '')

        if not all([movie_title, user_name, rating]):
            return jsonify({'error': 'Missing required fields'}), 400

        if not 0 <= rating <= 10:
            return jsonify({'error': 'Rating must be between 0 and 10'}), 400

        conn = get_db()
        c = conn.cursor()
        c.execute('''
            INSERT INTO reviews (movie_title, user_name, rating, comment)
            VALUES (?, ?, ?, ?)
        ''', (movie_title, user_name, rating, comment))
        conn.commit()
        conn.close()

        return jsonify({'message': 'Review added successfully'}), 201
    except Exception as e:
        logging.error(f"Error adding review: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/reviews/<movie_title>', methods=['GET'])
def get_reviews(movie_title):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('''
            SELECT user_name, rating, comment, created_at
            FROM reviews
            WHERE movie_title = ?
            ORDER BY created_at DESC
        ''', (movie_title,))
        reviews = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify(reviews)
    except Exception as e:
        logging.error(f"Error fetching reviews: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/custom-recommend', methods=['GET', 'POST'])
def custom_recommend():
    try:
        df = pd.read_csv('telugu_movies.csv')
        directors = sorted(df['director'].dropna().unique())
        genres = sorted(set(g for genre in df['genre'].dropna() for g in genre.split('/')))
        years = sorted(df['year'].dropna().unique(), reverse=True)
        results = []
        selected_director = request.form.get('director') if request.method == 'POST' else ''
        selected_genre = request.form.get('genre') if request.method == 'POST' else ''
        selected_year = request.form.get('year') if request.method == 'POST' else ''
        min_rating = request.form.get('min_rating') if request.method == 'POST' else ''
        movie_title = request.form.get('movie_title') if request.method == 'POST' else ''
        error = ''
        knn_recommendations = []

        if request.method == 'POST':
            if movie_title:
                # Get KNN-based recommendations
                knn_recommendations = telugu_recommender.get_recommendations(movie_title)
                if knn_recommendations is None:
                    error = f"Movie '{movie_title}' not found"
            else:
                # Get filtered results
                filtered = df.copy()
                if selected_director:
                    filtered = filtered[filtered['director'] == selected_director]
                if selected_genre:
                    filtered = filtered[filtered['genre'].str.contains(selected_genre, na=False)]
                if selected_year:
                    try:
                        year_val = int(selected_year)
                        filtered = filtered[filtered['year'] == year_val]
                    except ValueError:
                        error = 'Invalid year value.'
                if min_rating:
                    try:
                        min_rating_val = float(min_rating)
                        filtered = filtered[filtered['rating'] >= min_rating_val]
                    except ValueError:
                        error = 'Invalid rating value.'
                
                # Sort results by rating (highest first)
                filtered = filtered.sort_values('rating', ascending=False)
                results = filtered.to_dict('records')

        # Get reviews for each movie
        conn = get_db()
        c = conn.cursor()
        for movie in results:
            c.execute('''
                SELECT AVG(rating) as avg_rating, COUNT(*) as review_count
                FROM reviews
                WHERE movie_title = ?
            ''', (movie['title'],))
            review_stats = c.fetchone()
            movie['user_rating'] = round(review_stats['avg_rating'], 1) if review_stats['avg_rating'] else None
            movie['review_count'] = review_stats['review_count']
        conn.close()

        return render_template(
            'custom_recommend.html',
            directors=directors,
            genres=genres,
            years=years,
            results=results,
            knn_recommendations=knn_recommendations,
            selected_director=selected_director,
            selected_genre=selected_genre,
            selected_year=selected_year,
            min_rating=min_rating,
            movie_title=movie_title,
            error=error
        )
    except Exception as e:
        logger.error(f"Error in custom_recommend route: {str(e)}")
        return render_template(
            'custom_recommend.html',
            directors=[],
            genres=[],
            years=[],
            results=[],
            knn_recommendations=[],
            error=f"Error loading recommendations: {str(e)}"
        )

if __name__ == '__main__':
    init_db()
    app.run(debug=True) 
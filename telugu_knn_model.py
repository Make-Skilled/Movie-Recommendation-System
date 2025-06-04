import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeluguMovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.knn_model = None
        self.mlb = MultiLabelBinarizer()
        self.feature_matrix = None
        
    def load_data(self, csv_path='telugu_movies.csv'):
        """Load and preprocess the movie data"""
        try:
            self.movies_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.movies_df)} movies from {csv_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def prepare_features(self):
        """Prepare features for KNN model"""
        try:
            # Convert genres to binary features
            genres = self.movies_df['genre'].str.split('/')
            genre_matrix = self.mlb.fit_transform(genres)
            
            # Create feature matrix
            features = []
            for idx, row in self.movies_df.iterrows():
                feature_vector = []
                # Add genre features
                feature_vector.extend(genre_matrix[idx])
                # Add rating (normalized)
                feature_vector.append(row['rating'] / 10.0)
                # Add year (normalized)
                feature_vector.append((row['year'] - 2000) / 24.0)  # Assuming years range from 2000-2024
                features.append(feature_vector)
            
            self.feature_matrix = np.array(features)
            logger.info("Features prepared successfully")
            return True
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return False

    def train_model(self, n_neighbors=5):
        """Train the KNN model"""
        try:
            self.knn_model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='cosine',
                algorithm='brute'
            )
            self.knn_model.fit(self.feature_matrix)
            logger.info("KNN model trained successfully")
            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

    def get_recommendations(self, movie_title, n_recommendations=5):
        """Get movie recommendations based on a movie title"""
        try:
            # Find the movie index
            movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index
            if len(movie_idx) == 0:
                logger.warning(f"Movie '{movie_title}' not found")
                return None
            
            movie_idx = movie_idx[0]
            
            # Get nearest neighbors
            distances, indices = self.knn_model.kneighbors(
                self.feature_matrix[movie_idx].reshape(1, -1),
                n_neighbors=n_recommendations + 1
            )
            
            # Remove the movie itself from recommendations
            indices = indices[0][1:]
            distances = distances[0][1:]
            
            # Prepare recommendations
            recommendations = []
            for idx, distance in zip(indices, distances):
                movie = self.movies_df.iloc[idx]
                recommendations.append({
                    'title': movie['title'],
                    'director': movie['director'],
                    'genre': movie['genre'],
                    'rating': movie['rating'],
                    'year': movie['year'],
                    'comments': movie['comments'],
                    'similarity_score': 1 - distance  # Convert distance to similarity score
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return None

    def save_model(self, model_path='telugu_knn_model.pkl'):
        """Save the trained model"""
        try:
            model_data = {
                'knn_model': self.knn_model,
                'mlb': self.mlb,
                'feature_matrix': self.feature_matrix,
                'movies_df': self.movies_df
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, model_path='telugu_knn_model.pkl'):
        """Load a trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.knn_model = model_data['knn_model']
            self.mlb = model_data['mlb']
            self.feature_matrix = model_data['feature_matrix']
            self.movies_df = model_data['movies_df']
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

def train_and_save_model():
    """Train and save the KNN model"""
    recommender = TeluguMovieRecommender()
    if recommender.load_data():
        if recommender.prepare_features():
            if recommender.train_model():
                recommender.save_model()
                return True
    return False

if __name__ == '__main__':
    train_and_save_model() 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the movie data"""
    try:
        print("Loading movies data...")
        # Load the movies dataset
        if not os.path.exists('movies.csv'):
            raise FileNotFoundError("movies.csv not found")
        
        movies_df = pd.read_csv('movies.csv')
        print(f"Loaded {len(movies_df)} movies")
        print("Sample of movies:", movies_df['title'].head())
        
        # Load the ratings dataset
        if not os.path.exists('reviews.csv'):
            raise FileNotFoundError("reviews.csv not found")
            
        ratings_df = pd.read_csv('reviews.csv', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        print(f"Loaded {len(ratings_df)} ratings")
        
        # Merge movies and ratings
        merged_df = pd.merge(movies_df, ratings_df, on='item_id')
        print(f"Merged data shape: {merged_df.shape}")
        print("Sample of merged data:", merged_df[['item_id', 'title', 'rating']].head())
        
        return merged_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

class MovieRecommender:
    def __init__(self):
        self.ratings = None
        self.moviemat = None
        self.movie_titles = None
        self.data = None
        self.tfidf = None
        self.tfidf_matrix = None
        
    def load_data(self, ratings_path, movies_path):
        """Load and prepare the data"""
        print("\nLoading data in MovieRecommender...")
        # Load ratings data
        column_names = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(ratings_path, sep='\t', names=column_names)
        print(f"Loaded {len(df)} ratings")
        
        # Load movie titles
        self.movie_titles = pd.read_csv(movies_path)
        print(f"Loaded {len(self.movie_titles)} movies")
        print("Sample movie titles:", self.movie_titles['title'].head())
        
        # Merge data
        self.data = pd.merge(df, self.movie_titles, on='item_id')
        print(f"Merged data shape: {self.data.shape}")
        
        # Create ratings dataframe
        self.ratings = pd.DataFrame(self.data.groupby('title')['rating'].mean())
        self.ratings['num_of_ratings'] = pd.DataFrame(self.data.groupby('title')['rating'].count())
        print(f"Created ratings for {len(self.ratings)} unique movies")
        
        # Create movie matrix
        self.moviemat = self.data.pivot_table(index='user_id', columns='title', values='rating')
        print(f"Created movie matrix with {self.moviemat.shape[0]} users and {self.moviemat.shape[1]} movies")
        
        # Create TF-IDF matrix for content-based recommendations
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.movie_titles['title'])
        print("Created TF-IDF matrix for content-based recommendations")
        
    def find_movie_title(self, search_title):
        """Find the exact movie title from the database"""
        search_title = search_title.lower()
        print(f"\nSearching for movie: {search_title}")
        
        # Try exact match first
        exact_matches = self.movie_titles[self.movie_titles['title'].str.lower() == search_title]
        if not exact_matches.empty:
            print(f"Found exact match: {exact_matches.iloc[0]['title']}")
            return exact_matches.iloc[0]['title']
            
        # Try partial match
        partial_matches = self.movie_titles[self.movie_titles['title'].str.lower().str.contains(search_title)]
        if not partial_matches.empty:
            print(f"Found partial match: {partial_matches.iloc[0]['title']}")
            return partial_matches.iloc[0]['title']
            
        print("No matches found")
        return None
        
    def get_recommendations(self, movie_name, min_ratings=100, n_recommendations=10):
        """Get movie recommendations using both collaborative and content-based filtering"""
        print(f"\nGetting recommendations for: {movie_name}")
        
        # Find the exact movie title
        exact_title = self.find_movie_title(movie_name)
        if exact_title is None:
            print(f"Movie '{movie_name}' not found in database")
            return None
            
        if exact_title not in self.moviemat.columns:
            print(f"Movie '{exact_title}' has no ratings data")
            return None
            
        print(f"Found movie in database: {exact_title}")
        
        # Collaborative filtering recommendations
        print("Computing collaborative filtering recommendations...")
        movie_user_ratings = self.moviemat[exact_title]
        similar_movies = self.moviemat.corrwith(movie_user_ratings)
        corr_movie = pd.DataFrame(similar_movies, columns=['Correlation'])
        corr_movie.dropna(inplace=True)
        corr_movie = corr_movie.join(self.ratings['num_of_ratings'])
        collab_recommendations = corr_movie[corr_movie['num_of_ratings'] > min_ratings].sort_values('Correlation', ascending=False)
        print(f"Found {len(collab_recommendations)} collaborative recommendations")
        
        # Content-based recommendations
        print("Computing content-based recommendations...")
        movie_idx = self.movie_titles[self.movie_titles['title'] == exact_title].index[0]
        movie_vector = self.tfidf_matrix[movie_idx]
        content_similarities = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        content_indices = content_similarities.argsort()[::-1][1:n_recommendations+1]
        content_recommendations = self.movie_titles.iloc[content_indices]
        print(f"Found {len(content_recommendations)} content-based recommendations")
        
        # Combine both types of recommendations
        combined_recommendations = pd.DataFrame()
        if not collab_recommendations.empty:
            # Reset index to make title a column
            collab_recommendations = collab_recommendations.reset_index()
            collab_recommendations = collab_recommendations.rename(columns={'index': 'title'})
            collab_recommendations = collab_recommendations.head(n_recommendations)
            collab_recommendations['type'] = 'collaborative'
        
        content_df = pd.DataFrame({
            'title': content_recommendations['title'].values,
            'Correlation': content_similarities[content_indices],
            'type': 'content-based'
        })
        
        combined_recommendations = pd.concat([combined_recommendations, content_df])
        final_recommendations = combined_recommendations.sort_values('Correlation', ascending=False).head(n_recommendations)
        print(f"Returning {len(final_recommendations)} final recommendations")
        
        # Ensure all titles are strings and not NaN
        final_recommendations['title'] = final_recommendations['title'].fillna('Unknown Movie')
        final_recommendations['title'] = final_recommendations['title'].astype(str)
        
        return final_recommendations
        
    def get_popular_movies(self, n=10):
        """Get most popular movies based on number of ratings"""
        return self.ratings.sort_values('num_of_ratings', ascending=False).head(n)
        
    def get_highest_rated_movies(self, n=10, min_ratings=100):
        """Get highest rated movies with minimum number of ratings"""
        return self.ratings[self.ratings['num_of_ratings'] > min_ratings].sort_values('rating', ascending=False).head(n)

def create_soup(x):
    # Combine title and review for better recommendations
    review = x['review'] if 'review' in x and pd.notna(x['review']) else ''
    return f"{x['title']} {review}"

def build_model():
    try:
        print("\nBuilding recommendation model...")
        # Load the data
        movies_df = load_data()
        
        # Initialize and train the recommender
        recommender = MovieRecommender()
        recommender.load_data('reviews.csv', 'movies.csv')
        
        # Save the model
        print("\nSaving model...")
        with open('movie_recommender.pkl', 'wb') as f:
            pickle.dump(recommender, f)
        
        print("Model saved successfully!")
        print(f"Total movies processed: {len(movies_df)}")
        
    except Exception as e:
        print(f"Error building model: {str(e)}")
        raise

def get_recommendations(title, cosine_sim, indices, movies_df, n_recommendations=5):
    try:
        # Get the index of the movie that matches the title
        idx = indices[title]
        
        # Get the pairwise similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the n most similar movies
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Get the movie details
        recommendations = movies_df.iloc[movie_indices][['item_id', 'title']]
        
        # Add rating if available
        if 'rating' in movies_df.columns:
            recommendations['rating'] = movies_df.iloc[movie_indices]['rating']
        
        return recommendations
    
    except KeyError:
        print(f"Movie '{title}' not found in the database.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    build_model() 
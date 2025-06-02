# Movie Recommendation System

A movie recommendation system built with Python Flask, using collaborative filtering to suggest movies based on user ratings.

## Features

- Movie recommendations based on user ratings
- Popular movies listing
- Highest rated movies listing
- Clean and responsive UI using Bootstrap and Tailwind CSS

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the following data files in your project directory:
- `file.tsv` - Movie ratings data
- `Movie_Id_Titles.csv` - Movie titles data

3. Run the Flask application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py` - Flask application and routes
- `model.py` - Movie recommendation model
- `templates/` - HTML templates
- `static/` - Static files (CSS, images)
- `requirements.txt` - Project dependencies

## Usage

1. On the homepage, you'll see popular movies and highest-rated movies
2. Enter a movie name in the search box to get personalized recommendations
3. The system will show you similar movies based on user ratings
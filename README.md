# MovieMate - Movie Recommendation System

MovieMate is a sophisticated movie recommendation system that uses collaborative filtering and content-based filtering to provide personalized movie suggestions. The system features a modern, responsive web interface built with Flask, HTML, and Tailwind CSS.

## Features

- **Smart Recommendations**: Get personalized movie suggestions based on your preferences
- **Dual Recommendation Engine**:
  - Collaborative Filtering: Based on user behavior and preferences
  - Content-Based Filtering: Based on movie features and similarities
- **Popular Movies**: Discover trending and highly-rated movies
- **Modern UI**: Beautiful and responsive interface with smooth animations
- **Mobile-Friendly**: Fully responsive design that works on all devices

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, Tailwind CSS
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Icons**: Font Awesome
- **Styling**: Custom CSS animations and gradients

## Project Structure

```
Movie-Recommendation-System/
├── app.py                 # Flask application
├── model.py              # Recommendation system model
├── static/               # Static files
│   └── styles.css        # Custom CSS styles
├── templates/            # HTML templates
│   ├── index.html        # Home page
│   └── recommendations.html  # Recommendations page
├── movies.csv            # Movie dataset
├── ratings.csv           # User ratings dataset
└── README.md            # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Home Page**:
   - Enter a movie name in the search box
   - Click "Get Recommendations" to receive personalized suggestions
   - Browse popular and top-rated movies

2. **Recommendations Page**:
   - View personalized movie recommendations
   - See recommendation types (collaborative/content-based)
   - View similarity scores
   - Search for another movie

## Features in Detail

### Smart Recommendations
- Uses both collaborative and content-based filtering
- Provides similarity scores for each recommendation
- Combines multiple recommendation strategies for better results

### Modern UI Features
- Smooth animations and transitions
- Responsive design for all screen sizes
- Beautiful gradient backgrounds
- Interactive movie cards
- Mobile-friendly navigation

### Data Processing
- Efficient handling of large datasets
- Real-time recommendation generation
- Optimized for performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Movie data sourced from [MovieLens](https://movielens.org/)
- Icons provided by [Font Awesome](https://fontawesome.com/)
- UI framework by [Tailwind CSS](https://tailwindcss.com/)
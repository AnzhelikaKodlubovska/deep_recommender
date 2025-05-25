import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/ratings.csv"): 
    column_names = ["userId", "movieId", "rating", "timestamp"]
    
    # Явно вказуємо, що розділювач — табуляція, і додаємо назви колонок
    df = pd.read_csv(path, sep="\t", names=column_names)

    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    df['user'] = df['userId'].map(user2idx)
    df['movie'] = df['movieId'].map(movie2idx)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, val_df, user2idx, movie2idx

def load_movie_titles(movies_path="data/movies.csv"):
    df = pd.read_csv(movies_path, sep="|", encoding="latin1", header=None)
    
    df.columns = [
        "movie_id", "title", "release_date", "video_release_date", "url", 
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", 
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    movie_titles = dict(zip(df['movie_id'], df['title']))
    return movie_titles



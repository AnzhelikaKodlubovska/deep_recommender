import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# Імпортуємо RecommenderNet з model.py та load_data з data_loader.py
# Переконайтеся, що файл model.py існує і містить визначення RecommenderNet
from model import RecommenderNet
from data_loader import load_data # Припускаємо, що load_data вже змінена для ratings.csv

class MovieDataset(Dataset):
    def __init__(self, ratings_df, user2idx, movie2idx):
        self.users = ratings_df['userId'].map(user2idx).values
        self.movies = ratings_df['movieId'].map(movie2idx).values
        self.ratings = ratings_df['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'movie': torch.tensor(self.movies[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float32)
        }

def train_model(model, train_loader, val_loader, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['user'], batch['movie']).squeeze() # Додаємо .squeeze()
            loss = criterion(outputs, batch['rating'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                outputs = model(batch['user'], batch['movie']).squeeze() # Додаємо .squeeze()
                loss = criterion(outputs, batch['rating'])
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

# ====================================================================
# Додайте цей блок коду в кінці файлу train.py
# ====================================================================

if __name__ == "__main__":
    print("Loading data...")
    # Завантажуємо дані за допомогою функції з data_loader.py
    train_df, val_df, user2idx, movie2idx = load_data()

    print(f"Number of users: {len(user2idx)}")
    print(f"Number of movies: {len(movie2idx)}")
    print(f"Number of training ratings: {len(train_df)}")
    print(f"Number of validation ratings: {len(val_df)}")

    # Створюємо датасети та даталоадери
    train_dataset = MovieDataset(train_df, user2idx, movie2idx)
    val_dataset = MovieDataset(val_df, user2idx, movie2idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Ініціалізуємо модель
    n_users = len(user2idx)
    n_movies = len(movie2idx)
    model = RecommenderNet(n_users, n_movies)

    print("Starting model training...")
    # Навчаємо модель
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10) # Збільшив епохи до 10 для кращого навчання

    print("Training complete.")

    # Побудова графіка втрат (необов'язково, але корисно)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


    # ====================================================================
    # Ось найважливіша частина: збереження навченої моделі!
    # ====================================================================
    model_dir = "model"
    model_path = os.path.join(model_dir, "recommender_model.pth")

    # Створюємо папку 'model', якщо вона не існує
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # Зберігаємо стан моделі
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")
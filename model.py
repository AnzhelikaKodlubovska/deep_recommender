import torch
import torch.nn as nn

class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=50):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embedding_dim)
        self.movie_embed = nn.Embedding(n_movies, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, user_ids, movie_ids):
        user_vec = self.user_embed(user_ids)
        movie_vec = self.movie_embed(movie_ids)
        x = torch.cat([user_vec, movie_vec], dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze()

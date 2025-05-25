import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QComboBox, QListWidget
)
from data_loader import load_data, load_movie_titles
from model import RecommenderNet


def load_model(model_path, n_users, n_movies):
    model = RecommenderNet(n_users, n_movies)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def recommend_top_movies(user_id, model, movie_ids, user2idx, movie2idx, top_n=5):
    user_idx = torch.tensor([user2idx[user_id]] * len(movie_ids))
    movie_idxs = torch.tensor([movie2idx[m] for m in movie_ids])
    with torch.no_grad():
        preds = model(user_idx, movie_idxs)
    top_indices = preds.argsort(descending=True)[:top_n]
    return [movie_ids[i] for i in top_indices]


class RecommenderApp(QWidget):
    def __init__(self):
        super().__init__()
        with open("styles.css", "r") as styleFile:
            self.setStyleSheet(styleFile.read())
        self.setWindowTitle("🎬 Movie Recommender")

        self.train_df, _, self.user2idx, self.movie2idx = load_data()
        self.movie_titles = load_movie_titles()
        self.movie_ids = self.train_df['movieId'].unique()

        self.model = load_model("model/recommender_model.pth",
                                len(self.user2idx), len(self.movie2idx))

        self.layout = QVBoxLayout()

        self.label = QLabel("Оберіть користувача (ID):")
        self.combo = QComboBox()
        self.combo.addItems([str(uid) for uid in self.user2idx.keys()])

        self.button = QPushButton("Рекомендувати фільми")
        self.button.clicked.connect(self.show_recommendations)

        self.list_widget = QListWidget()

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combo)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.list_widget)

        self.setLayout(self.layout)

    def show_recommendations(self):
        self.list_widget.clear()
        user_id = int(self.combo.currentText())
        recommended_ids = recommend_top_movies(
            user_id, self.model, self.movie_ids, self.user2idx, self.movie2idx
        )
        for mid in recommended_ids:
            title = self.movie_titles.get(mid, f"Movie ID {mid}")
            self.list_widget.addItem(title)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecommenderApp()
    window.show()
    sys.exit(app.exec_())

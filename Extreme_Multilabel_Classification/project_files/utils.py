from pathlib import Path
import pickle


def load_data(path: str = "./data"):
    with open(Path(path) / "train_features_sparse.pickle", "rb") as f:
        train_features = pickle.load(f)
    with open(Path(path) / "train_labels_sparse.pickle", "rb") as f:
        train_labels = pickle.load(f)
    with open(Path(path) / "dev_features_sparse.pickle", "rb") as f:
        dev_features = pickle.load(f)
    with open(Path(path) / "dev_labels_sparse.pickle", "rb") as f:
        dev_labels = pickle.load(f)
    return train_features, train_labels, dev_features, dev_labels
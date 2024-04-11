import numpy as np

from src import metrics

n_classes = 1000
emb_dim = 16
n_samples = 10000

y_true = np.random.randint(0, n_classes, size=(n_samples,))
embeddings = np.random.randn(n_samples, emb_dim)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

metrics.eval_metrics(y_true, embeddings, verbose=True)

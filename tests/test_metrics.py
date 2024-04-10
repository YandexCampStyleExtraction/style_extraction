import numpy as np

from src import metrics

n_classes = 10
emb_dim = 32
n_samples = 1000

y_true = np.random.randint(0, n_classes + 1, size=(n_samples,))
embeddings = np.random.random((n_samples, emb_dim))

# print(y_true[:5])
# print(embeddings[:5])

fprs = [0.5, 0.2, 0.01]
for fpr in fprs:
    print(f"TPR@FPR={fpr}: {metrics.tpr_at_fpr(y_true, embeddings, fpr)}")

symmetries = [None, 'jeffreys', 'jensen-shannon']
for symmetry in symmetries:
    print(f"KL-Divergence ({symmetry=}): {metrics.kl_div(y_true, embeddings, symmetry=symmetry)}")

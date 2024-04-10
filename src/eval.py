from src import metrics
from loguru import logger
import numpy as np


def get_metrics(model, dataloader):
    pred = []
    labels = []
    for i, batch in enumerate(dataloader):
        # batch = batch.to(device)
        labels.append(batch.pop('labels'))
        pred.append(model(**batch))

    pred = np.array(pred).ravel()
    labels = np.array(labels).ravel()

    # tpr_at_fpr
    fprs = [0.5, 0.2, 0.01]
    for fpr in fprs:
        logger.info(f"TPR@FPR={fpr}: {metrics.tpr_at_fpr(labels, pred, fpr)}")

    # same_sim
    logger.info(f"Same_sim: {metrics.get_same_similarities(labels, pred)}")

    # false_sim
    logger.info(f"False_sim: {metrics.get_false_similarities(labels, pred)}")

    logger.info(f"kl_div: {metrics.kl_div(labels, pred)}")

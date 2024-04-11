from models.losses import TripletMarginLoss, ContrastiveLoss


AVAILABLE_SSL_LOSSES = {
    'contrastive': ContrastiveLoss,
    'triplet': TripletMarginLoss
}

AVAILABLE_CLS_LOSSES = {
    'arcface',
    'sphereface',
    'cosface',
}

import numpy as np


def scoreToPercentage(score, seuil, minseuil):
    score = np.maximum(np.minimum(score, 0), minseuil)
    percentage = np.where(score > seuil, np.round((1 - score/seuil*0.5)*100), np.round((0.5 - (score-seuil)/(-500-seuil)*0.5)*100))
    return percentage

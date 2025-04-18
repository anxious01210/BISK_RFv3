# face_recognition/utils.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def is_same_person(embedding1, embedding2, threshold=0.45):
    similarity = cosine_similarity(
        [embedding1],
        [embedding2]
    )[0][0]
    return similarity >= threshold, similarity

# extras/test_face.py
# If you ever run scripts from subfolders (like inside extras/) and want to import local modules cleanly, you can add the project root to PYTHONPATH: PYTHONPATH=. python extras/test_face.py
#  Or, add this to the top of test_face.py:
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from face_recognition.embedder import get_face_embeddings
from face_recognition.utils import is_same_person

img1 = cv2.imread("person1.jpg")
img2 = cv2.imread("person1_duplicate.jpg")

face1 = get_face_embeddings(img1)[0]
face2 = get_face_embeddings(img2)[0]

same, score = is_same_person(face1.embedding, face2.embedding)
print(f"Same person: {same}, Similarity score: {score}")

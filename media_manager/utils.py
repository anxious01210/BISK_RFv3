import os
from django.conf import settings

def safe_join(base, *paths):
    final_path = os.path.abspath(os.path.join(base, *paths))
    if not final_path.startswith(os.path.abspath(base)):
        raise ValueError("Path traversal detected.")
    return final_path

def is_image_file(filename):
    ext = filename.lower().split('.')[-1]
    return ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']

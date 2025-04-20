# attendance/models.py
from django.db import models
from django.utils.html import format_html

class Student(models.Model):
    h_code = models.CharField(max_length=20, unique=True)
    full_name = models.CharField(max_length=100)
    # optional: class_name, grade, etc.

    def __str__(self):
        return f"{self.h_code} - {self.full_name}"

class FaceImage(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='face_images')
    image_path = models.CharField(max_length=255)  # You can switch to ImageField + django-filer later if needed
    embedding_path = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.h_code} - {self.image_path}"

    def image_tag(self):
        return format_html('<img src="{}" style="height: 100px;" />', self.image_path)
    image_tag.short_description = 'Preview'

# attendance/models.py
from django.conf import settings
from django.db import models
from django.utils.html import format_html

class Student(models.Model):
    h_code = models.CharField(max_length=20, unique=True)
    full_name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
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
        if self.image_path:
            url = settings.MEDIA_URL + self.image_path
            return format_html(
                '<a href="{}" target="_blank"><img src="{}" style="height: 100px;" /></a>',
                url, url
            )
        return "No image"

    image_tag.short_description = 'Preview'

class Period(models.Model):
    name = models.CharField(max_length=50)
    start_time = models.TimeField()
    end_time = models.TimeField()
    is_active = models.BooleanField(default=True)  # To enable/disable this period

    def __str__(self):
        return f"{self.name} ({self.start_time} - {self.end_time})"

class Camera(models.Model):
    name = models.CharField(max_length=100)
    url = models.CharField(max_length=255, help_text="URL of the camera main-stream, ex. rtsp://admin:examplepass123!@192.168.1.100:554/Streaming/Channels/101/")
    location = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
    stream_start_time = models.TimeField(null=True, blank=True)
    stream_end_time = models.TimeField(null=True, blank=True)

    def __str__(self):
        return self.name

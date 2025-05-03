from django.db import models

class Upload(models.Model):
    MEDIA_TYPE_CHOICES = (
        ('image', 'Image'),
        ('video', 'Video'),
    )
    file = models.FileField(upload_to='face_inspector_uploads/')
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.media_type} uploaded on {self.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}"

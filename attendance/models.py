# attendance/models.py
from django.conf import settings
from django.db import models
from django.utils.html import format_html
from django.utils import timezone
from multiselectfield import MultiSelectField

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


class AttendanceLog(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    period = models.ForeignKey(Period, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.SET_NULL, null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ('student', 'period')  # prevent duplicate logs per period

    def __str__(self):
        return f"{self.student.h_code} @ {self.period.name} on {self.timestamp.astimezone(timezone.get_current_timezone()).strftime('%Y-%m-%d %H:%M:%S')}"


WEEKDAYS = (
    ('Mon', 'Monday'),
    ('Tue', 'Tuesday'),
    ('Wed', 'Wednesday'),
    ('Thu', 'Thursday'),
    ('Fri', 'Friday'),
    ('Sat', 'Saturday'),
    ('Sun', 'Sunday'),
)

class RecognitionSchedule(models.Model):
    name = models.CharField(max_length=100)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="schedules", default=1 )
    weekdays = MultiSelectField(choices=WEEKDAYS)
    start_time = models.TimeField()
    end_time = models.TimeField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} - {self.camera.name} ({', '.join(self.weekdays)} {self.start_time}-{self.end_time})"

    @property
    def weekday_values(self):
        day_map = {
            "Mon": 0, "Tue": 1, "Wed": 2,
            "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6
        }
        return [day_map[day] for day in self.weekdays if day in day_map]

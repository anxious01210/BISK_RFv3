# attendance/models.py
from django.conf import settings
from django.db import models
from django.utils.html import format_html
from django.utils import timezone
from multiselectfield import MultiSelectField
# from django.utils.timezone import localtime, make_aware
import os
from django.utils.timezone import now
import logging

logger = logging.getLogger(__name__)


def attendance_crop_path(instance, filename):
    today = now().date()
    h_code = getattr(instance.student, 'h_code', None)
    camera_name = getattr(instance.camera, 'name', None)
    timestamp = now().strftime("%Y%m%d_%H%M%S")

    if not h_code or not camera_name:
        raise ValueError("Missing student.h_code or camera.name in attendance_crop_path")

    name = f"{h_code}_{camera_name}_{timestamp}.jpg"
    path = os.path.join("attendance_crops", today.strftime("%Y/%m/%d"), name)

    logger.debug(f"[UPLOAD_PATH] -> {path}")
    return path


# def attendance_crop_path(instance, filename):
#     today = now().date()
#     h_code = instance.student.h_code
#     camera_name = instance.camera.name
#     timestamp = now().strftime("%Y%m%d_%H%M%S")
#
#     name = f"{h_code}_{camera_name}_{timestamp}.jpg"
#     path = os.path.join("attendance_crops", today.strftime("%Y/%m/%d"), name)
#
#     # Log it for investigation
#     logger.warning(f"[DEBUG][UPLOAD_PATH] Raw filename: {filename}")
#     logger.warning(f"[DEBUG][UPLOAD_PATH] Computed path: {path}")
#
#     return path

# import datetime
#
# import os
# from datetime import datetime
#
# def attendance_crop_path(instance, filename):
#     today = datetime.now()
#     return os.path.join(
#         "attendance_crops",
#         str(today.year),
#         f"{today.month:02}",
#         f"{today.day:02}",
#         filename
#     )
#

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
    attendance_window_seconds = models.PositiveIntegerField(
        default=5,
        help_text="Time window in seconds to update attendance if a better match is found. Default is 15 seconds."
    )

    def __str__(self):
        return f"{self.name} ({self.start_time} - {self.end_time})"
        # try:
        #     now_date = timezone.now().date()
        #     start_dt = make_aware(datetime.datetime.combine(now_date, self.start_time))
        #     end_dt = make_aware(datetime.datetime.combine(now_date, self.end_time))
        #     return f"{self.name} ({localtime(start_dt).time()} - {localtime(end_dt).time()})"
        # except Exception:
        #     return f"{self.name} ({self.start_time} - {self.end_time})"


class Camera(models.Model):
    name = models.CharField(max_length=100)
    url = models.CharField(max_length=255, help_text="URL of the camera main-stream, ex. rtsp://admin:examplepass123!@192.168.1.100:554/Streaming/Channels/101/")
    location = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    stream_start_time = models.TimeField(null=True, blank=True)
    stream_end_time = models.TimeField(null=True, blank=True)

    def __str__(self):
        return self.name

# def attendance_crop_path(instance, filename):
#     now = timezone.now()
#     return f"attendance_crops/{now.year}/{now.month:02}/{now.day:02}/{filename}"

class AttendanceLog(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    # cropped_face = models.ImageField(upload_to='attendance_crops/', blank=True, null=True)
    cropped_face = models.ImageField(upload_to=attendance_crop_path, blank=True, null=True)
    period = models.ForeignKey(Period, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.SET_NULL, null=True, blank=True)
    match_score = models.FloatField(null=True, blank=True, help_text="Cosine similarity score of the recognition match.")
    timestamp = models.DateTimeField()
    date = models.DateField() # for the default in makemigration use __import__('datetime').date.today()

    class Meta:
        unique_together = ('student', 'period', 'date')  # prevent duplicate logs per period

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
    cameras = models.ManyToManyField(Camera, related_name="schedules")
    weekdays = MultiSelectField(choices=WEEKDAYS)
    start_time = models.TimeField()
    end_time = models.TimeField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        # return f"{self.name} - {self.cameras.name} ({', '.join(self.weekdays)} {self.start_time}-{self.end_time})"
        return f"{self.name} - ({', '.join(self.weekdays)} {self.start_time}-{self.end_time})"

    @property
    def weekday_values(self):
        day_map = {
            "Mon": 0, "Tue": 1, "Wed": 2,
            "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6
        }
        return [day_map[day] for day in self.weekdays if day in day_map]

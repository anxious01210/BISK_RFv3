from django.contrib import admin
from django.urls import path
from django.utils.safestring import mark_safe
from django.http import HttpResponse
from import_export import resources
from import_export.admin import ImportExportModelAdmin
import os

from .models import Student, FaceImage, Period, Camera, AttendanceLog, RecognitionSchedule

### --- Resources ---

class StudentResource(resources.ModelResource):
    class Meta:
        model = Student

class FaceImageResource(resources.ModelResource):
    class Meta:
        model = FaceImage

class PeriodResource(resources.ModelResource):
    class Meta:
        model = Period

class CameraResource(resources.ModelResource):
    class Meta:
        model = Camera

class AttendanceLogResource(resources.ModelResource):
    class Meta:
        model = AttendanceLog

### --- Admin Classes ---

@admin.register(Student)
class StudentAdmin(ImportExportModelAdmin):
    resource_class = StudentResource
    list_display = ('h_code', 'full_name', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('h_code', 'full_name')

@admin.register(FaceImage)
class FaceImageAdmin(ImportExportModelAdmin):
    resource_class = FaceImageResource
    list_display = ('student', 'image_tag', 'embedding_path', 'uploaded_at')
    readonly_fields = ('image_tag',)
    search_fields = ('student__h_code', 'image_path')

@admin.register(Period)
class PeriodAdmin(ImportExportModelAdmin):
    resource_class = PeriodResource
    list_display = ('name', 'start_time', 'end_time', 'is_active', 'attendance_window_seconds')
    list_filter = ('is_active',)
    search_fields = ('name',)
    ordering = ('start_time',)

# @admin.register(Camera)
# class CameraAdmin(ImportExportModelAdmin):
#     resource_class = CameraResource
#     list_display = ('name', 'location', 'is_active', 'stream_start_time', 'stream_end_time')
#     list_filter = ('is_active',)
#
#     def get_urls(self):
#         urls = super().get_urls()
#         custom_urls = [
#             path('log-viewer/', self.admin_site.admin_view(self.log_viewer), name='camera-log-viewer')
#         ]
#         return custom_urls + urls
#
#     def log_viewer(self, request):
#         log_dir = os.path.join(os.path.dirname(__file__), '../logs/attendance')
#         if not os.path.exists(log_dir):
#             return HttpResponse("Log directory not found.", content_type='text/plain')
#
#         files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
#         content = "<h2>Camera Logs</h2><ul>"
#         for f in sorted(files):
#             with open(os.path.join(log_dir, f), 'r') as log_file:
#                 lines = log_file.readlines()[-30:]  # Show last 30 lines
#                 content += f"<li><strong>{f}</strong><pre>{''.join(lines)}</pre></li><br>"
#         content += "</ul>"
#         return HttpResponse(content)


# @admin.register(Camera)
# class CameraAdmin(ImportExportModelAdmin):
#     resource_class = CameraResource
#     list_display = ('name', 'location', 'is_active', 'stream_start_time', 'stream_end_time', 'view_logs_button')
#     list_filter = ('is_active',)
#
#     def get_urls(self):
#         urls = super().get_urls()
#         custom_urls = [
#             path('log-viewer/', self.admin_site.admin_view(self.log_viewer), name='camera-log-viewer'),
#         ]
#         return custom_urls + urls
#
#     def log_viewer(self, request):
#         log_dir = os.path.join(os.path.dirname(__file__), '../logs/attendance')
#         if not os.path.exists(log_dir):
#             return HttpResponse("Log directory not found.", content_type='text/plain')
#
#         files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
#         content = "<h2>Camera Logs</h2><ul>"
#         for f in sorted(files):
#             with open(os.path.join(log_dir, f), 'r') as log_file:
#                 lines = log_file.readlines()[-30:]  # Show last 30 lines
#                 content += f"<li><strong>{f}</strong><pre>{''.join(lines)}</pre></li><br>"
#         content += "</ul>"
#         return HttpResponse(content)
#
#     def view_logs_button(self, obj):
#         return mark_safe(
#             f'<a class="button" href="/admin/attendance/camera/log-viewer/" target="_blank" '
#             f'style="padding:4px 8px; background-color:#28a745; color:white; border-radius:5px; text-decoration:none;">'
#             f'📄 View Logs</a>'
#         )
#
#     view_logs_button.short_description = 'Logs'
#     view_logs_button.allow_tags = True


@admin.register(Camera)
class CameraAdmin(ImportExportModelAdmin):
    resource_class = CameraResource
    list_display = ('name', 'location', 'is_active', 'stream_start_time', 'stream_end_time', 'view_logs_button')
    list_filter = ('is_active',)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('log-viewer/<int:camera_id>/', self.admin_site.admin_view(self.log_viewer), name='camera-log-viewer'),
        ]
        return custom_urls + urls

    def log_viewer(self, request, camera_id):
        log_dir = os.path.join(os.path.dirname(__file__), '../logs/attendance')
        if not os.path.exists(log_dir):
            return HttpResponse("Log directory not found.", content_type='text/plain')

        try:
            camera = Camera.objects.get(id=camera_id)
        except Camera.DoesNotExist:
            return HttpResponse("Camera not found.", content_type='text/plain')

        # Now filter logs related to this camera name
        files = [f for f in os.listdir(log_dir) if f.endswith('.log') and camera.name in f]
        if not files:
            return HttpResponse(f"No logs found for camera: {camera.name}", content_type='text/plain')

        content = f"<h2>Logs for Camera: {camera.name}</h2><ul>"
        for f in sorted(files):
            with open(os.path.join(log_dir, f), 'r') as log_file:
                lines = log_file.readlines()[-30:]  # Last 30 lines
                content += f"<li><strong>{f}</strong><pre>{''.join(lines)}</pre></li><br>"
        content += "</ul>"
        return HttpResponse(content)

    def view_logs_button(self, obj):
        return mark_safe(
            f'<a class="button" href="/admin/attendance/camera/log-viewer/{obj.id}/" target="_blank" '
            f'style="padding:4px 8px; background-color:#28a745; color:white; border-radius:5px; text-decoration:none;">'
            f'📄 View {obj.name} Logs</a>'
        )

    view_logs_button.short_description = 'Logs'
    view_logs_button.allow_tags = True



@admin.register(AttendanceLog)
class AttendanceLogAdmin(ImportExportModelAdmin):
    resource_class = AttendanceLogResource
    list_display = ('student', 'period', 'camera', 'timestamp', 'date', 'colored_match_score')
    list_filter = ('period', 'camera', 'date')
    search_fields = ('student__h_code', 'student__full_name', 'camera__name', 'date')

    def colored_match_score(self, obj):
        try:
            score = float(obj.match_score)
        except (ValueError, TypeError):
            return mark_safe("<span style='color: gray;'>-</span>")

        # New coloring based on cosine similarity (0 - 1)
        if score >= 0.8:
            color = 'green'
        elif score >= 0.6:
            color = 'orange'
        else:
            color = 'red'

        return mark_safe(f"<span style='color: {color};'>%.2f</span>" % score)

    colored_match_score.short_description = mark_safe(
        "Match Score<br><small><span style='color:green;'>Green ≥ 0.80</span>"
        "<span style='color:orange;'>Orange ≥ 0.60</span>"
        "<span style='color:red;'>Red &lt; 0.60</span></small>"
    )

@admin.register(RecognitionSchedule)
class RecognitionScheduleAdmin(ImportExportModelAdmin):
    list_display = ('name', 'get_weekdays_display', 'start_time', 'end_time', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('name',)

    def get_weekdays_display(self, obj):
        return ", ".join(obj.weekdays)
    get_weekdays_display.short_description = "Weekdays"

# attendance/admin.py

from django.contrib import admin
from django.urls import path, reverse
from django.utils.safestring import mark_safe
from django.http import HttpResponse, FileResponse
from import_export import resources
from import_export.admin import ImportExportModelAdmin
from django.shortcuts import get_object_or_404
from django.conf import settings

import os
from datetime import date

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

@admin.register(Camera)
class CameraAdmin(ImportExportModelAdmin):
    resource_class = CameraResource
    list_display = ('name', 'location', 'is_active', 'stream_start_time', 'stream_end_time', 'view_logs_button')
    list_filter = ('is_active',)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('<int:camera_id>/log-viewer/', self.admin_site.admin_view(self.log_viewer), name='camera-log-viewer'),
        ]
        return custom_urls + urls

    def view_logs_button(self, obj):
        return mark_safe(
            f'<a class="button" href="{reverse("admin:camera-log-viewer", args=[obj.id])}" target="_blank" '
            f'style="padding:4px 8px; background-color:#28a745; color:white; border-radius:5px; text-decoration:none;">'
            f'📄 View Logs</a>'
        )

    view_logs_button.short_description = 'Logs'
    view_logs_button.allow_tags = True

    # def log_viewer(self, request, camera_id):
    #     camera = get_object_or_404(Camera, id=camera_id)
    #
    #     log_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'attendance')
    #     today_str = date.today().strftime('%Y-%m-%d')
    #     log_filename = f"{camera.name}_{today_str}.log"
    #     log_path = os.path.join(log_dir, log_filename)
    #
    #     if not os.path.exists(log_path):
    #         return HttpResponse(f"Log file not found: {log_filename}", content_type='text/plain')
    #
    #     # === CONFIG: Number of lines ===
    #     show_last_n_lines = 300  # or set to None for all lines
    #
    #     with open(log_path, 'r', encoding='utf-8') as log_file:
    #         lines = log_file.readlines()
    #         if show_last_n_lines is not None:
    #             lines = lines[-show_last_n_lines:]
    #
    #     content = f"""
    #         <h2>📄 Logs for Camera: {camera.name}</h2>
    #         <a href="/media/logs/attendance/{log_filename}" download
    #            style="display:inline-block;margin-bottom:10px;padding:6px 12px;background-color:#007bff;color:white;border-radius:5px;text-decoration:none;">
    #            ⬇️ Download Log
    #         </a>
    #         <pre style="white-space: pre-wrap; font-size: 13px; background:#f8f8f8; padding:10px;">{''.join(lines)}</pre>
    #     """
    #     return HttpResponse(content)
    def log_viewer(self, request, camera_id):
        camera = get_object_or_404(Camera, id=camera_id)

        log_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'attendance')
        today_str = date.today().strftime('%Y-%m-%d')
        log_filename = f"{camera.name}_{today_str}.log"
        log_path = os.path.join(log_dir, log_filename)

        if not os.path.exists(log_path):
            return HttpResponse(f"Log file not found: {log_filename}", content_type='text/plain')

        # No need to read file content here, frontend will fetch it dynamically
        log_url = f"/media/logs/attendance/{log_filename}"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>📄 Logs for Camera: {camera.name}</title>
            <meta charset="utf-8">
            <script>
                function loadLogs() {{
                    fetch('{log_url}')
                        .then(response => response.text())
                        .then(data => {{
                            var logArea = document.getElementById('log-content');
                            logArea.textContent = data;
                            logArea.scrollTop = logArea.scrollHeight;  // Auto-scroll to bottom
                        }})
                        .catch(err => {{
                            console.error('Error loading logs:', err);
                        }});
                }}
                setInterval(loadLogs, 1000);  // Refresh every 1 seconds
                window.onload = loadLogs;  // Load immediately
            </script>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                pre {{
                    white-space: pre-wrap;
                    font-size: 13px;
                    background: #f8f8f8;
                    padding: 10px;
                    height: 80vh;
                    overflow-y: scroll;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                }}
                a.download-button {{
                    display: inline-block;
                    margin-bottom: 10px;
                    padding: 6px 12px;
                    background-color: #007bff;
                    color: white;
                    border-radius: 5px;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <h2>📄 Logs for Camera: {camera.name}</h2>
            <a class="download-button" href="{log_url}" download>⬇️ Download Log</a><br><br>
            <pre id="log-content">Loading...</pre>
        </body>
        </html>
        """
        return HttpResponse(html)


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

from django.contrib import admin
from import_export import resources
from import_export.admin import ImportExportModelAdmin
from django.utils.html import format_html
from django.utils.safestring import mark_safe
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
    list_display = ('name', 'start_time', 'end_time', 'is_active')
    list_filter = ('is_active',)

@admin.register(Camera)
class CameraAdmin(ImportExportModelAdmin):
    resource_class = CameraResource
    list_display = ('name', 'location', 'is_active', 'stream_start_time', 'stream_end_time')
    list_filter = ('is_active',)

@admin.register(AttendanceLog)
class AttendanceLogAdmin(ImportExportModelAdmin):
    resource_class = AttendanceLogResource
    list_display = ('student', 'period', 'camera', 'timestamp', 'colored_match_score')
    list_filter = ('period', 'camera')
    search_fields = ('student__h_code',)

    def colored_match_score(self, obj):
        try:
            score = float(obj.match_score)
        except (ValueError, TypeError):
            return mark_safe("<span style='color: gray;'>-</span>")

        color = (
            'green' if score >= 250 else
            'orange' if score >= 220 else
            'red'
        )
        return mark_safe(f"<span style='color: {color};'>%.2f</span>" % score)

    colored_match_score.short_description = mark_safe(
        # "Match Score [ <span style='color:orange;'>Orange</span> (220–250) &lt; <span style='color:green;'>Green</span> (~300–350) ]"
        "Match Score<span style='color:green; padding:0 2px;''>250–350</span><span style='color:orange; padding: 0 2px;'> 220–249</span>"
    )

@admin.register(RecognitionSchedule)
class RecognitionScheduleAdmin(ImportExportModelAdmin):
    list_display = ('name', 'get_weekdays_display', 'start_time', 'end_time', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('name',)

    def get_weekdays_display(self, obj):
        return ", ".join(obj.weekdays)
    get_weekdays_display.short_description = "Weekdays"

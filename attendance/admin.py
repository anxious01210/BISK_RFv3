from django.contrib import admin
from import_export import resources
from import_export.admin import ImportExportModelAdmin
from .models import Student, FaceImage, Period, Camera

# -------------------------------
# Resources for Import-Export
# -------------------------------

class StudentResource(resources.ModelResource):
    class Meta:
        model = Student
        fields = ('id', 'h_code', 'full_name', 'is_active')

class FaceImageResource(resources.ModelResource):
    class Meta:
        model = FaceImage
        fields = ('id', 'student__h_code', 'image_path', 'embedding_path', 'uploaded_at')

class PeriodResource(resources.ModelResource):
    class Meta:
        model = Period
        fields = ('id', 'name', 'start_time', 'end_time', 'is_active')

class CameraResource(resources.ModelResource):
    class Meta:
        model = Camera
        fields = ('id', 'name', 'stream_url', 'location', 'is_active', 'start_time', 'end_time')


# -------------------------------
# Admin Configurations
# -------------------------------

@admin.register(Student)
class StudentAdmin(ImportExportModelAdmin):
    list_display = ('h_code', 'full_name', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('h_code', 'full_name')
    resource_class = StudentResource

@admin.register(FaceImage)
class FaceImageAdmin(ImportExportModelAdmin):
    list_display = ('student', 'image_tag', 'embedding_path', 'uploaded_at')
    readonly_fields = ('image_tag',)
    search_fields = ('student__h_code',)
    resource_class = FaceImageResource

@admin.register(Period)
class PeriodAdmin(ImportExportModelAdmin):
    list_display = ('name', 'start_time', 'end_time', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('name',)
    resource_class = PeriodResource

@admin.register(Camera)
class CameraAdmin(ImportExportModelAdmin):
    list_display = ('name', 'location', 'is_active', 'stream_start_time', 'stream_end_time')
    list_filter = ('is_active',)
    search_fields = ('name', 'location')
    resource_class = CameraResource

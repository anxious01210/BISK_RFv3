from django.contrib import admin
from import_export import resources
from import_export.admin import ImportExportModelAdmin

from .models import Student, FaceImage


# 1. Import/export resource for Student
class StudentResource(resources.ModelResource):
    class Meta:
        model = Student
        fields = ('id', 'h_code', 'full_name')  # Add any other fields here
        export_order = ('id', 'h_code', 'full_name')


# 2. Admin for Student
@admin.register(Student)
class StudentAdmin(ImportExportModelAdmin):
    resource_class = StudentResource
    list_display = ('h_code', 'full_name')
    search_fields = ('h_code', 'full_name')
    list_per_page = 50
    ordering = ('h_code',)


# 3. Admin for FaceImage
@admin.register(FaceImage)
class FaceImageAdmin(admin.ModelAdmin):
    list_display = ('student', 'image_tag', 'image_path', 'embedding_path', 'uploaded_at')
    list_filter = ('uploaded_at',)
    search_fields = ('student__h_code', 'student__full_name', 'image_path')
    readonly_fields = ('image_tag',)
    ordering = ('-uploaded_at',)
    list_per_page = 50

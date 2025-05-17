# file_manager/urls.py
from django.urls import path
from . import views

app_name = "file_manager"

urlpatterns = [
    path('', views.explorer_view, name='explorer'),
    path('list-folder/', views.list_folder_contents, name='list_folder'),
    path("analyze-folders/", views.analyze_folders, name="analyze_folders"),
    path("upload/", views.upload_files, name="upload_files"),
]
    # path("upload/", views.upload_file, name="upload_file"),

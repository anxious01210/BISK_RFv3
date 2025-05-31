# file_manager/urls.py
from django.urls import path
from . import views

app_name = "file_manager"

urlpatterns = [
    path('', views.explorer_view, name='explorer'),
    path('list-folder/', views.list_folder_contents, name='list_folder'),
    path("analyze-folders/", views.analyze_folders, name="analyze_folders"),
    path("upload/", views.upload_files, name="upload_files"),
    path("run-embeddings/", views.run_embeddings_script, name="run_embeddings_script"),
    path("run-sort-faces/", views.run_sort_faces_script, name="run_sort_faces_script"),
    path('stream-sort-faces/<uuid:job_id>/', views.stream_sort_faces_logs, name='stream_sort_faces_logs'),
    # path("stop-sort-faces/<uuid:job_id>/", views.stop_sort_faces_job, name="stop_sort_faces_job"),
    # path('stream-sort-faces-output/<uuid:job_id>/', views.stream_sort_faces_output, name='stream_sort_faces_output'),
    # path("stream-sort-faces/<uuid:job_id>/", views.stream_sort_faces_logs, name="stream_sort_faces_logs"),
    # path("stream-sort-faces/<uuid:job_id>/", views.stream_sort_faces_output, name="stream_sort_faces_output"),
]
# path("upload/", views.upload_file, name="upload_file"),
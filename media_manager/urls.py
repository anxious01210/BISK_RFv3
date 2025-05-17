from django.urls import path
from . import views

app_name = "media_manager"

urlpatterns = [
    path('', views.browser_view, name='browser'),
    path('upload/', views.upload_files, name='upload_files'),
    path('create-folder/', views.create_folder, name='create_folder'),
    path('delete/', views.delete_files, name='delete_files'),
    path('rename/', views.rename_file, name='rename_file'),
    path('move/', views.move_files, name='move_files'),
    path('download/', views.download_files, name='download_files'),
    path('folder-tree/', views.folder_tree, name='folder_tree'),
]

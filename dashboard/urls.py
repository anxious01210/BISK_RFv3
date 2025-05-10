# dashboard/urls.py
from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('system_stats/', views.system_stats_view, name='system_stats'),
]






# # dashboard/urls.py
# from django.urls import path
# from . import views
#
# app_name = 'dashboard'
#
# urlpatterns = [
#     path('', views.dashboard_view, name='dashboard'),
#     path('system_stats/', views.system_stats_view, name='system_stats'),
# ]
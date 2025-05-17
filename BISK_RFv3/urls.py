"""
URL configuration for BISK_RFv3 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from filebrowser.sites import site as filebrowser_site
urlpatterns = [
    path('admin/filebrowser/', filebrowser_site.urls),
    path('admin/', admin.site.urls),
    path('dashboard/', include('dashboard.urls', namespace='dashboard')),
    path('media-manager/', include('media_manager.urls', namespace='media_manager')),
    path('file-manager/', include('file_manager.urls', namespace='file_manager')),
    path("__reload__/", include("django_browser_reload.urls")),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Modify Site Header
admin.site.site_header = 'BISK Admin Panel'
# Modify Site Title
admin.site.site_title = "BISK site admin "
# Modify Site Index Title
admin.site.index_title = "BISK Portal"

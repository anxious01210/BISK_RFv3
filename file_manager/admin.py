from django.contrib import admin
from django.http import HttpResponseRedirect
from django.urls import reverse

class FileManagerAdminView(admin.ModelAdmin):
    def changelist_view(self, request, extra_context=None):
        return HttpResponseRedirect(reverse('file_manager:explorer'))

# Dummy model just to create the link
from django.db import models
class FileManagerLink(models.Model):
    class Meta:
        verbose_name = "📁 File Manager"
        verbose_name_plural = "📁 File Manager"
        managed = False  # Don't create table

admin.site.register(FileManagerLink, FileManagerAdminView)

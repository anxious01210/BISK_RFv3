from django.contrib import admin
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.db import models

class DashboardLink(models.Model):
    class Meta:
        verbose_name = "📊 Dashboard"
        verbose_name_plural = "📊 Dashboard"
        managed = False

class DashboardAdmin(admin.ModelAdmin):
    def changelist_view(self, request, extra_context=None):
        return HttpResponseRedirect(reverse("dashboard:dashboard"))

admin.site.register(DashboardLink, DashboardAdmin)

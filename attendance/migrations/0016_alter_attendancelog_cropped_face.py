# Generated by Django 5.2 on 2025-05-01 15:08

import attendance.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0015_attendancelog_cropped_face'),
    ]

    operations = [
        migrations.AlterField(
            model_name='attendancelog',
            name='cropped_face',
            field=models.ImageField(blank=True, null=True, upload_to=attendance.models.attendance_crop_path),
        ),
    ]

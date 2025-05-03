import os
from django.conf import settings
from django.contrib import admin
from django.utils.html import format_html
from .models import Upload
from .face_utils import process_uploaded_media
from django.utils.safestring import mark_safe

class UploadAdmin(admin.ModelAdmin):
    list_display = ['file', 'media_type', 'uploaded_at', 'view_faces']
    readonly_fields = ['view_faces', 'media_type']

    def save_model(self, request, obj, form, change):
        ext = os.path.splitext(obj.file.name)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            obj.media_type = 'video'
        elif ext in ['.jpg', '.jpeg', '.png']:
            obj.media_type = 'image'
        else:
            obj.media_type = 'unknown'
        super().save_model(request, obj, form, change)
        process_uploaded_media(obj.file.path, obj.media_type)


    def view_faces(self, obj):
        base_name = os.path.splitext(os.path.basename(obj.file.name))[0]
        processed_dir = os.path.join(settings.MEDIA_ROOT, 'face_inspector_processed', base_name)

        if not os.path.exists(processed_dir):
            return "No processed faces found."

        imgs = []
        for f in sorted(os.listdir(processed_dir)):
            if not f.lower().endswith(".jpg"):
                continue

            img_path = f'/media/face_inspector_processed/{base_name}/{f}'
            filename = f.replace('.jpg', '')

            h_code = "Unknown"
            name = "Not recognized"
            score = None
            border_color = "#6666cc"  # default purple for unknowns

            if '__' in filename and '_score=' in filename:
                try:
                    h_code, rest = filename.split('__', 1)
                    base_part, score_str = rest.rsplit('_score=', 1)
                    name = base_part.rsplit('_', 1)[0] if '_' in base_part else base_part
                    score = float(score_str)

                    if h_code.lower() == "unknown":
                        border_color = "#6666cc"  # purple for unknowns
                    elif score >= 0.80:
                        border_color = 'green'
                    elif score >= 0.50:
                        border_color = 'orange'
                    else:
                        border_color = 'red'

                except Exception as e:
                    print(f"⚠️ Failed to parse file: {f} → {e}")
            else:
                # keep h_code = Unknown, name = Not recognized, border = purple
                pass

            caption = f"<b>{h_code}</b><br><small>{name}"
            if score is not None:
                caption += f"<br>score={score:.2f}"
            else:
                caption += "<br>not recognized"
            caption += "</small>"

            imgs.append(f"""
                <div style='text-align:center;margin:10px'>
                    <a href='{img_path}' target='_blank'>
                        <img src='{img_path}' width='100' style='border: 3px solid {border_color}; border-radius:4px; padding: 4px; margin-bottom: 2px;'>
                    </a><br>
                    {caption}
                </div>
            """)

        return mark_safe(
            f"<div style='display:grid;grid-template-columns:repeat(6, 1fr);gap:10px'>{''.join(imgs)}</div>"
        )

    view_faces.short_description = "Face Matches"


admin.site.register(Upload, UploadAdmin)








# import os
# from django.conf import settings
# from django.contrib import admin
# from django.utils.html import format_html
# from .models import Upload
# from .face_utils import process_uploaded_media
# from django.utils.safestring import mark_safe
#
# class UploadAdmin(admin.ModelAdmin):
#     list_display = ['file', 'media_type', 'uploaded_at', 'view_faces']
#     readonly_fields = ['view_faces', 'media_type']
#
#     def save_model(self, request, obj, form, change):
#         ext = os.path.splitext(obj.file.name)[1].lower()
#         if ext in ['.mp4', '.avi', '.mov', '.mkv']:
#             obj.media_type = 'video'
#         elif ext in ['.jpg', '.jpeg', '.png']:
#             obj.media_type = 'image'
#         else:
#             obj.media_type = 'unknown'
#         super().save_model(request, obj, form, change)
#         process_uploaded_media(obj.file.path, obj.media_type)
#
#     def view_faces(self, obj):
#         base_name = os.path.splitext(os.path.basename(obj.file.name))[0]
#         processed_dir = os.path.join(settings.MEDIA_ROOT, 'face_inspector_processed', base_name)
#
#         if not os.path.exists(processed_dir):
#             return "No processed faces found."
#
#         imgs = []
#         for f in sorted(os.listdir(processed_dir)):
#             if not f.lower().endswith(".jpg"):
#                 continue
#
#             img_path = f'/media/face_inspector_processed/{base_name}/{f}'
#             filename = f.replace('.jpg', '')
#
#             try:
#                 h_code, rest = filename.split('__', 1)
#                 score = 0.0
#
#                 if '_score=' in rest:
#                     base_part, score_str = rest.rsplit('_score=', 1)
#                     name = base_part.rsplit('_', 1)[0] if '_' in base_part else base_part
#                     score = float(score_str)
#                 else:
#                     name = rest
#
#                 if score >= 0.80:
#                     border_color = 'green'
#                 elif score >= 0.50:
#                     border_color = 'orange'
#                 else:
#                     border_color = 'red'
#
#                 imgs.append(f"""
#                     <div style='text-align:center;margin:10px'>
#                         <a href='{img_path}' target='_blank'>
#                             <img src='{img_path}' width='100' style='border: 3px solid {border_color}; border-radius:4px; padding: 4px; margin: 4px;'>
#                         </a><br>
#                         <b>{h_code}</b><br>
#                         <small>{name}<br>score={score:.2f}</small>
#                     </div>
#                 """)
#             except Exception as e:
#                 print(f"⚠️ Failed to parse filename: {f} — {e}")
#                 imgs.append(f"""
#                     <div style='text-align:center;margin:10px'>
#                         <a href='{img_path}' target='_blank'>
#                             <img src='{img_path}' width='100' style='border: 3px solid #888888; border-radius:4px; padding: 4px; margin: 4px;'>
#                         </a><br>
#                         <b>Unknown</b><br>
#                         <small>Not recognized</small>
#                     </div>
#                 """)
#
#         if not imgs:
#             return "No matched face images found."
#
#         return mark_safe(
#             f"<div style='display:grid;grid-template-columns:repeat(6, 1fr);gap:10px'>{''.join(imgs)}</div>"
#         )
#
#     view_faces.short_description = "Face Matches"
#
# admin.site.register(Upload, UploadAdmin)






# import os
# from django.conf import settings
# from django.contrib import admin
# from django.utils.html import format_html
# from .models import Upload
# from .face_utils import process_uploaded_media
# from django.utils.safestring import mark_safe
# from django.conf import settings
#
# class UploadAdmin(admin.ModelAdmin):
#     list_display = ['file', 'media_type', 'uploaded_at', 'view_faces']
#     readonly_fields = ['view_faces', 'media_type']
#
#     def save_model(self, request, obj, form, change):
#         ext = os.path.splitext(obj.file.name)[1].lower()
#         if ext in ['.mp4', '.avi', '.mov', '.mkv']:
#             obj.media_type = 'video'
#         elif ext in ['.jpg', '.jpeg', '.png']:
#             obj.media_type = 'image'
#         else:
#             obj.media_type = 'unknown'
#         super().save_model(request, obj, form, change)
#         process_uploaded_media(obj.file.path, obj.media_type)
#
#
#     def view_faces(self, obj):
#         base_name = os.path.splitext(os.path.basename(obj.file.name))[0]
#         processed_dir = os.path.join(settings.MEDIA_ROOT, 'face_inspector_processed', base_name)
#
#         if not os.path.exists(processed_dir):
#             return "No processed faces found."
#
#         imgs = []
#         for f in sorted(os.listdir(processed_dir)):
#             if not f.lower().endswith(".jpg"):
#                 continue
#
#             img_path = f'/media/face_inspector_processed/{base_name}/{f}'
#             filename = f.replace('.jpg', '')
#
#             try:
#                 h_code, rest = filename.split('__', 1)
#                 score = 0.0
#
#                 if '_score=' in rest:
#                     base_part, score_str = rest.rsplit('_score=', 1)
#                     name = base_part.rsplit('_', 1)[0] if '_' in base_part else base_part
#                     score = float(score_str)
#                 else:
#                     name = rest
#
#                 if score >= 0.80:
#                     border_color = 'green'
#                 elif score >= 0.50:
#                     border_color = 'orange'
#                 else:
#                     border_color = 'red'
#
#                 imgs.append(f"""
#                     <div style='text-align:center;margin:10px'>
#                         <a href='{img_path}' target='_blank'>
#                             <img src='{img_path}' width='100' style='border: 3px solid {border_color}; border-radius:4px; padding: 4px; margin: 4px;'>
#                         </a><br>
#                         <b>{h_code}</b><br>
#                         <small>{name}<br>score={score:.2f}</small>
#                     </div>
#                 """)
#             except Exception as e:
#                 print(f"⚠️ Failed to parse filename: {f} — {e}")
#                 imgs.append(f"""
#                     <div style='text-align:center;margin:10px'>
#                         <a href='{img_path}' target='_blank'>
#                             <img src='{img_path}' width='100' style='border: 3px solid gray; border-radius:4px; padding: 4px; margin: 4px;'>
#                         </a><br>
#                         <b>Unknown</b><br>
#                         <small>Not recognized</small>
#                     </div>
#                 """)
#
#         if not imgs:
#             return "No matched face images found."
#
#         return mark_safe(
#             f"<div style='display:grid;grid-template-columns:repeat(6, 1fr);gap:10px'>{''.join(imgs)}</div>"
#         )
#
# admin.site.register(Upload, UploadAdmin)






# import os
# from django.conf import settings
# from django.contrib import admin
# from django.utils.html import format_html
# from .models import Upload
# from .face_utils import process_uploaded_media
#
# class UploadAdmin(admin.ModelAdmin):
#     list_display = ['file', 'media_type', 'uploaded_at', 'view_faces']
#     readonly_fields = ['view_faces', 'media_type']
#
#     # Automatic media_type
#     def save_model(self, request, obj, form, change):
#         ext = os.path.splitext(obj.file.name)[1].lower()
#         if ext in ['.mp4', '.avi', '.mov', '.mkv']:
#             obj.media_type = 'video'
#         elif ext in ['.jpg', '.jpeg', '.png']:
#             obj.media_type = 'image'
#         else:
#             obj.media_type = 'unknown'
#         super().save_model(request, obj, form, change)
#         process_uploaded_media(obj.file.path, obj.media_type)
#
#     # Updated display method inside UploadAdmin
#     def view_faces(self, obj):
#         processed_dir = obj.file.path.replace('face_inspector_uploads', 'face_inspector_processed')
#         if not os.path.exists(processed_dir):
#             return "No processed faces found."
#
#         imgs = []
#         for f in sorted(os.listdir(processed_dir)):
#             img_path = f'/media/face_inspector_processed/{os.path.basename(processed_dir)}/{f}'
#             filename = f.replace('.jpg', '')
#
#             parts = filename.split('__')
#             if len(parts) == 2:
#                 h_code, name_part = parts
#                 if '_' in name_part:
#                     name, idx = name_part.rsplit('_', 1)
#                 else:
#                     name, idx = name_part, ''
#                 if '_score=' in name:
#                     name, score_str = name.split('_score=')
#                     score = float(score_str)
#                 else:
#                     score = 0.0
#                 # Color logic
#                 if score >= 0.80:
#                     border_color = 'green'
#                 elif score >= 0.50:
#                     border_color = 'orange'
#                 else:
#                     border_color = 'red'
#
#                 imgs.append(f"""
#                 <div style='text-align:center;margin:10px'>
#                     <a href='{img_path}' target='_blank'>
#                         <img src='{img_path}' width='100' style='border: 2px solid {border_color}; border-radius:4px'>
#                     </a><br>
#                     <b>{h_code}</b><br>
#                     <small>{name.strip()}<br>score={score:.2f}</small>
#                 </div>
#                 """)
#             else:
#                 imgs.append(f"""
#                 <div style='text-align:center;margin:10px'>
#                     <a href='{img_path}' target='_blank'>
#                         <img src='{img_path}' width='100' style='border: 2px solid gray; border-radius:4px'>
#                     </a><br>
#                     <b>Unknown</b><br>
#                     <small>Not recognized</small>
#                 </div>
#                 """)
#
#         return format_html("<div style='display:grid;grid-template-columns:repeat(6, 1fr);gap:10px'>{}</div>",
#                            ''.join(imgs))
#
#         if not imgs:
#             return "No matched face images found."
#
#         html_grid = "<div style='display:grid;grid-template-columns:repeat(6, 1fr);gap:15px'>{}</div>".format(
#             ''.join(imgs))
#         return format_html(html_grid)
#
#     # def view_faces(self, obj):
#     #     # Derive processed folder path
#     #     base_name = os.path.splitext(os.path.basename(obj.file.name))[0]
#     #     processed_dir = os.path.join(settings.MEDIA_ROOT, 'face_inspector_processed', base_name)
#     #
#     #     if not os.path.exists(processed_dir):
#     #         return "No processed faces found."
#     #
#     #     imgs = []
#     #     for f in sorted(os.listdir(processed_dir)):
#     #         if f.lower().endswith('.jpg'):
#     #             try:
#     #                 h_code, name = f.split('__')[0], f.split('__')[1].replace('.jpg', '')
#     #             except IndexError:
#     #                 h_code, name = "Unknown", "Unknown"
#     #
#     #             img_url = os.path.join(settings.MEDIA_URL, 'face_inspector_processed', base_name, f)
#     #             imgs.append(f"""
#     #                 <div style='text-align:center;margin:10px'>
#     #                     <img src='{img_url}' width='100'><br>
#     #                     <b>{h_code}</b><br>
#     #                     <small>{name}</small>
#     #                 </div>
#     #             """)
#     #
#     #     if not imgs:
#     #         return "No matched face images found."
#     #
#     #     return format_html("<div style='display:grid;grid-template-columns:repeat(6, 1fr);gap:10px'>{}</div>", ''.join(imgs))
#
#     view_faces.short_description = "Matched Faces (Grid)"
#
# admin.site.register(Upload, UploadAdmin)




# import os
# from django.contrib import admin
# from django.utils.html import format_html
# from .models import Upload
# from .face_utils import process_uploaded_media  # to be created
#
# class UploadAdmin(admin.ModelAdmin):
#     list_display = ['file', 'media_type', 'uploaded_at', 'view_faces']
#     readonly_fields = ['view_faces']
#
#     def save_model(self, request, obj, form, change):
#         super().save_model(request, obj, form, change)
#         process_uploaded_media(obj.file.path, obj.media_type)  # main processing
#
#     def view_faces(self, obj):
#         processed_dir = obj.file.path.replace('face_inspector_uploads', 'face_inspector_processed')
#         if not os.path.exists(processed_dir):
#             return "No processed faces found."
#
#         imgs = []
#         for f in sorted(os.listdir(processed_dir)):
#             if f.endswith('.jpg'):
#                 h_code, name = f.split('__')[0], f.split('__')[1].replace('.jpg', '')
#                 img_path = f'/media/face_inspector_processed/{os.path.basename(processed_dir)}/{f}'
#                 imgs.append(f"""
#                 <div style='text-align:center;margin:10px'>
#                     <img src='{img_path}' width='100'><br>
#                     <b>{h_code}</b><br>
#                     <small>{name}</small>
#                 </div>
#                 """)
#         return format_html("<div style='display:grid;grid-template-columns:repeat(6, 1fr);gap:10px'>{}</div>", ''.join(imgs))
#
#     view_faces.allow_tags = True
#     view_faces.short_description = "Matched Faces (Grid)"
#
# admin.site.register(Upload, UploadAdmin)

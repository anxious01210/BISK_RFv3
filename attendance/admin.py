# attendance/admin.py

from django.contrib import admin
from django.urls import path, reverse
from django.utils.safestring import mark_safe
from django.http import HttpResponse, FileResponse
from import_export import resources
from import_export.admin import ImportExportModelAdmin
from django.shortcuts import get_object_or_404
from django.conf import settings

import os
from datetime import date

from .models import Student, FaceImage, Period, Camera, AttendanceLog, RecognitionSchedule

### --- Resources ---

class StudentResource(resources.ModelResource):
    class Meta:
        model = Student

class FaceImageResource(resources.ModelResource):
    class Meta:
        model = FaceImage

class PeriodResource(resources.ModelResource):
    class Meta:
        model = Period

class CameraResource(resources.ModelResource):
    class Meta:
        model = Camera

class AttendanceLogResource(resources.ModelResource):
    class Meta:
        model = AttendanceLog

### --- Admin Classes ---

@admin.register(Student)
class StudentAdmin(ImportExportModelAdmin):
    resource_class = StudentResource
    list_display = ('h_code', 'full_name', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('h_code', 'full_name')

@admin.register(FaceImage)
class FaceImageAdmin(ImportExportModelAdmin):
    resource_class = FaceImageResource
    list_display = ('student', 'image_tag', 'embedding_path', 'uploaded_at')
    readonly_fields = ('image_tag',)
    search_fields = ('student__h_code', 'image_path')

@admin.register(Period)
class PeriodAdmin(ImportExportModelAdmin):
    resource_class = PeriodResource
    list_display = ('name', 'start_time', 'end_time', 'is_active', 'attendance_window_seconds')
    list_filter = ('is_active',)
    search_fields = ('name',)
    ordering = ('start_time',)

@admin.register(Camera)
class CameraAdmin(ImportExportModelAdmin):
    resource_class = CameraResource
    list_display = ('name', 'location', 'is_active', 'stream_start_time', 'stream_end_time', 'view_logs_button')
    list_filter = ('is_active',)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('<int:camera_id>/log-viewer/', self.admin_site.admin_view(self.log_viewer), name='camera-log-viewer'),
        ]
        return custom_urls + urls

    def view_logs_button(self, obj):
        return mark_safe(
            f'<a class="button" href="{reverse("admin:camera-log-viewer", args=[obj.id])}" target="_blank" '
            f'style="padding:4px 8px; background-color:#28a745; color:white; border-radius:5px; text-decoration:none;">'
            f'📄 View Logs</a>'
        )

    view_logs_button.short_description = 'Logs'
    view_logs_button.allow_tags = True


    # Theis version (I love it)
    def log_viewer(self, request, camera_id):
        from django.utils.html import escape
        camera = get_object_or_404(Camera, id=camera_id)

        log_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'attendance')

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>📄 Logs for Camera: {camera.name}</title>
            <meta charset="utf-8">
            <script>
                var refreshInterval = 5000;
                var autoScrollEnabled = true;
                var refreshTimer = null;
                var selectedDate = (new Date()).toISOString().slice(0,10);
                var darkMode = false;

                function escapeHTML(str) {{
                    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                }}

                function colorize(line) {{
                    if (line.includes('✅') || line.includes('🟢') || line.includes('🔁')) {{
                        return '<span style="color: limegreen;">' + escapeHTML(line) + '</span>';
                    }} else if (line.includes('❌') || line.includes('⚠️')) {{
                        return '<span style="color: tomato;">' + escapeHTML(line) + '</span>';
                    }} else if (line.includes('🛑') || line.includes('WARNING')) {{
                        return '<span style="color: orange;">' + escapeHTML(line) + '</span>';
                    }} else {{
                        return '<span style="color: inherit;">' + escapeHTML(line) + '</span>';
                    }}
                }}

                function getLogUrl() {{
                    return `/media/logs/attendance/{camera.name}_` + selectedDate + '.log';
                }}

                function loadLogs() {{
                    fetch(getLogUrl())
                        .then(response => {{
                            if (!response.ok) throw new Error('Not found');
                            return response.text();
                        }})
                        .then(data => {{
                            var lines = data.split('\\n');
                            var filtered = lines;

                            document.getElementById('total-lines').innerText = lines.length;

                            var query = document.getElementById('search-input').value.toLowerCase();
                            if (query) {{
                                filtered = lines.filter(line => line.toLowerCase().includes(query));
                            }}

                            document.getElementById('filtered-lines').innerText = filtered.length;

                            var coloredLines = filtered.map(colorize);
                            var logArea = document.getElementById('log-content');
                            logArea.style.opacity = 0.3;  // smooth fade start
                            setTimeout(() => {{
                                logArea.innerHTML = coloredLines.join('<br>');
                                logArea.style.opacity = 1;  // fade in
                                if (autoScrollEnabled) {{
                                    logArea.scrollTop = logArea.scrollHeight;
                                }}
                            }}, 100);
                        }})
                        .catch(err => {{
                            document.getElementById('log-content').innerHTML = '<i>No log found for selected date.</i>';
                            document.getElementById('total-lines').innerText = 0;
                            document.getElementById('filtered-lines').innerText = 0;
                        }});
                }}

                function handleSearch() {{
                    loadLogs();
                }}

                function updateRefreshInterval() {{
                    var ms = parseInt(document.getElementById('refresh-input').value);
                    if (isNaN(ms) || ms <= 0) {{
                        ms = 5000;
                        document.getElementById('refresh-input').value = 5000;
                    }}
                    refreshInterval = ms;
                    clearInterval(refreshTimer);
                    refreshTimer = setInterval(loadLogs, refreshInterval);
                }}

                function toggleAutoScroll() {{
                    autoScrollEnabled = !autoScrollEnabled;
                    var button = document.getElementById('toggle-scroll');
                    button.innerText = autoScrollEnabled ? 'Auto-Scroll: ON' : 'Auto-Scroll: OFF';
                }}

                function changeDate() {{
                    selectedDate = document.getElementById('date-picker').value;
                    loadLogs();
                }}

                function toggleDarkMode() {{
                    darkMode = !darkMode;
                    if (darkMode) {{
                        document.body.classList.add('dark-mode');
                    }} else {{
                        document.body.classList.remove('dark-mode');
                    }}
                }}

                window.onload = function() {{
                    document.getElementById('date-picker').value = selectedDate;
                    loadLogs();
                    refreshTimer = setInterval(loadLogs, refreshInterval);
                }};
            </script>

            <style>
                body {{
                    font-family: Arial, sans-serif;
                    padding: 20px;
                    transition: background-color 0.3s, color 0.3s;
                }}
                .dark-mode {{
                    background-color: #121212;
                    color: #e0e0e0;
                }}
                .dark-mode pre {{
                    background-color: #1e1e1e;
                }}
                #top-bar {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }}
                #top-left {{
                    display: flex;
                    align-items: center;
                    gap: 15px;
                }}
                #controls {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    flex-wrap: wrap;
                    margin-bottom: 20px;
                }}
                #search-input {{
                    width: 400px;
                    padding: 6px;
                }}
                #refresh-input {{
                    width: 120px;
                    padding: 6px;
                }}
                input, button, select {{
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 6px;
                }}
                pre {{
                    white-space: pre-wrap;
                    font-size: 13px;
                    background: #f8f8f8;
                    padding: 10px;
                    height: 80vh;
                    overflow-y: scroll;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    transition: opacity 0.5s;
                }}
                .dark-mode pre::-webkit-scrollbar-thumb {{
                    background-color: #555;
                }}
                .dark-mode pre::-webkit-scrollbar-track {{
                    background: #222;
                }}
                .dark-mode-button {{
                    background: #444;
                    color: #fff;
                    padding: 8px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                }}
                .download-button {{
                    padding: 6px 12px;
                    background-color: #007bff;
                    color: white;
                    border-radius: 5px;
                    text-decoration: none;
                }}
                .counter {{
                    font-size: 13px;
                    margin-left: 10px;
                }}
            </style>
        </head>
        <body>
            <div id="top-bar">
                <div id="top-left">
                    <a id="download-link" href="/media/logs/attendance/{camera.name}_"+(new Date()).toISOString().slice(0,10)+".log" download class="download-button">⬇️ Download Log</a>
                    <h2>📄 Logs for Camera: {camera.name}</h2>
                </div>
                <div class="dark-mode-button" onclick="toggleDarkMode()">🌓</div>
            </div>

            <div id="controls">
                🔎 <input type="text" id="search-input" placeholder="Search..." oninput="handleSearch()">
                📅 <input type="date" id="date-picker" onchange="changeDate()">
                🔄 <input type="number" id="refresh-input" value="5000" onchange="updateRefreshInterval()"> ms
                <button id="toggle-scroll" onclick="toggleAutoScroll()">Auto-Scroll: ON</button>
                <span class="counter">📈 Filtered: <span id="filtered-lines">0</span></span>
                <span class="counter">📏 Total: <span id="total-lines">0</span></span>
            </div>

            <pre id="log-content">Loading...</pre>

        </body>
        </html>
        """
        return HttpResponse(html)


    # The last version (with some good features)
    # def log_viewer(self, request, camera_id):
    #     from .models import Camera
    #     from datetime import date
    #     import os
    #     from django.conf import settings
    #     from django.shortcuts import get_object_or_404
    #     from django.http import HttpResponse
    #
    #     camera = get_object_or_404(Camera, id=camera_id)
    #     today_str = date.today().strftime('%Y-%m-%d')
    #     default_log_filename = f"{camera.name}_{today_str}.log"
    #
    #     html = f"""
    #         <head>
    #             <title>Logs - {camera.name}</title>
    #             <meta charset="utf-8">
    #             <style>
    #                 body {{
    #                     font-family: Arial, sans-serif;
    #                     margin: 20px;
    #                     background-color: white;
    #                     color: black;
    #                     transition: background-color 0.3s, color 0.3s;
    #                 }}
    #                 .dark-mode {{
    #                     background-color: #121212;
    #                     color: #e0e0e0;
    #                 }}
    #                 #top-bar {{
    #                     display: flex;
    #                     justify-content: space-between;
    #                     align-items: center;
    #                     margin-bottom: 10px;
    #                 }}
    #                 #controls {{
    #                     display: flex;
    #                     gap: 10px;
    #                     margin-bottom: 10px;
    #                     flex-wrap: wrap;
    #                 }}
    #                 input[type="text"], input[type="number"], input[type="date"] {{
    #                     padding: 5px;
    #                     border-radius: 5px;
    #                     border: 1px solid #ccc;
    #                 }}
    #                 button {{
    #                     padding: 6px 12px;
    #                     border-radius: 5px;
    #                     border: none;
    #                     cursor: pointer;
    #                 }}
    #                 #download-btn {{
    #                     background: white;
    #                     color: #007bff;
    #                     # border: 1px solid #007bff;
    #                     font-weight: bold;
    #                     text-decoration: none;
    #                 }}
    #                 #download-btn:hover {{
    #                     background-color: #007bff;
    #                     color: white;
    #                 }}
    #                 #refresh-btn {{
    #                     background-color: #28a745;
    #                     color: white;
    #                 }}
    #                 #toggle-dark {{
    #                     background-color: #333;
    #                     color: white;
    #                 }}
    #                 pre {{
    #                     background: #f8f8f8;
    #                     padding: 10px;
    #                     white-space: pre-wrap;
    #                     height: 600px;
    #                     overflow-y: scroll;
    #                     border-radius: 10px;
    #                     font-size: 13px;
    #                 }}
    #                 .dark-mode pre {{
    #                     background: #1e1e1e;
    #                 }}
    #                 #topBtn {{
    #                     display: none;
    #                     position: fixed;
    #                     bottom: 30px;
    #                     right: 30px;
    #                     z-index: 99;
    #                     font-size: 16px;
    #                     border: none;
    #                     outline: none;
    #                     background-color: #007bff;
    #                     color: white;
    #                     cursor: pointer;
    #                     padding: 10px 15px;
    #                     border-radius: 50%;
    #                     box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    #                     transition: background-color 0.3s;
    #                 }}
    #                 #topBtn:hover {{
    #                     background-color: #0056b3;
    #                 }}
    #             </style>
    #         </head>
    #         <body>
    #
    #             <div id="top-bar">
    #                 <div>
    #                     <a href="/media/logs/attendance/{default_log_filename}" id="download-btn" download>⬇️ Download Log</a>
    #                     <strong style="margin-left: 15px;">Logs for Camera: {camera.name}</strong>
    #                 </div>
    #                 <button id="toggle-dark" onclick="toggleDark()">🌗</button>
    #             </div>
    #
    #             <div id="controls">
    #                 <input type="text" id="search" placeholder="🔍 Search logs..." style="width:300px;">
    #                 <input type="date" id="date-picker" value="{today_str}">
    #                 <input type="number" id="refresh-interval" placeholder="🔁 Refresh ms" style="width:120px;" value="5000">
    #                 <label><input type="checkbox" id="auto-scroll" checked> Auto-scroll</label>
    #                 <span id="line-stats" style="margin-left: auto;">Lines Loaded: 0 | Total: 0</span>
    #             </div>
    #
    #             <pre id="log-content">Loading...</pre>
    #
    #             <button onclick="scrollToTop()" id="topBtn" title="Go to top">⬆️</button>
    #
    #             <script>
    #                 let autoScroll = true;
    #                 let refreshInterval = 5000;
    #                 let timer;
    #
    #                 document.getElementById('auto-scroll').addEventListener('change', function() {{
    #                     autoScroll = this.checked;
    #                 }});
    #
    #                 document.getElementById('refresh-interval').addEventListener('change', function() {{
    #                     refreshInterval = parseInt(this.value) || 5000;
    #                     clearInterval(timer);
    #                     timer = setInterval(fetchLogs, refreshInterval);
    #                 }});
    #
    #                 document.getElementById('date-picker').addEventListener('change', fetchLogs);
    #
    #                 function fetchLogs() {{
    #                     const dateValue = document.getElementById('date-picker').value;
    #                     const cameraName = "{camera.name}";
    #                     const logPath = `/media/logs/attendance/${{cameraName}}_${{dateValue}}.log`;
    #
    #                     fetch(logPath)
    #                         .then(response => {{
    #                             if (!response.ok) {{
    #                                 throw new Error("Log not found.");
    #                             }}
    #                             return response.text();
    #                         }})
    #                         .then(data => {{
    #                             const searchText = document.getElementById('search').value.toLowerCase();
    #                             const allLines = data.split("\\n");
    #                             const filtered = allLines.filter(line => line.toLowerCase().includes(searchText));
    #
    #                             // Highlight colorized lines
    #                             const coloredContent = filtered.map(line => {{
    #                                 if (line.includes("ERROR") || line.includes("❌")) return `<span style='color:red;'>${{line}}</span>`;
    #                                 if (line.includes("WARNING") || line.includes("⚠️")) return `<span style='color:orange;'>${{line}}</span>`;
    #                                 return `<span style='color:green;'>${{line}}</span>`;
    #                             }}).join("<br>");
    #
    #                             document.getElementById('log-content').innerHTML = coloredContent;
    #                             document.getElementById('line-stats').innerText = `Lines Loaded: ${{filtered.length}} | Total: ${{allLines.length}}`;
    #
    #                             if (autoScroll) {{
    #                                 const logBox = document.getElementById('log-content');
    #                                 logBox.scrollTop = logBox.scrollHeight;
    #                             }}
    #                         }})
    #                         .catch(error => {{
    #                             document.getElementById('log-content').innerHTML = "<span style='color:red;'>No logs found for this date.</span>";
    #                             document.getElementById('line-stats').innerText = `Lines Loaded: 0 | Total: 0`;
    #                         }});
    #                 }}
    #
    #                 document.getElementById('search').addEventListener('input', fetchLogs);
    #
    #                 function toggleDark() {{
    #                     document.body.classList.toggle('dark-mode');
    #                 }}
    #
    #                 function scrollToTop() {{
    #                     window.scrollTo({{ top: 0, behavior: 'smooth' }});
    #                 }}
    #
    #                 window.onscroll = function() {{
    #                     const btn = document.getElementById("topBtn");
    #                     if (document.body.scrollTop > 300 || document.documentElement.scrollTop > 300) {{
    #                         btn.style.display = "block";
    #                     }} else {{
    #                         btn.style.display = "none";
    #                     }}
    #                 }};
    #
    #                 fetchLogs();
    #                 timer = setInterval(fetchLogs, refreshInterval);
    #             </script>
    #
    #         </body>
    #     """
    #
    #     return HttpResponse(html)


@admin.register(AttendanceLog)
class AttendanceLogAdmin(ImportExportModelAdmin):
    resource_class = AttendanceLogResource
    list_display = ('student', 'period', 'camera', 'timestamp', 'date', 'colored_match_score')
    list_filter = ('period', 'camera', 'date')
    search_fields = ('student__h_code', 'student__full_name', 'camera__name', 'date')

    def colored_match_score(self, obj):
        try:
            score = float(obj.match_score)
        except (ValueError, TypeError):
            return mark_safe("<span style='color: gray;'>-</span>")

        # New coloring based on cosine similarity (0 - 1)
        if score >= 0.8:
            color = 'green'
        elif score >= 0.6:
            color = 'orange'
        else:
            color = 'red'

        return mark_safe(f"<span style='color: {color};'>%.2f</span>" % score)

    colored_match_score.short_description = mark_safe(
        "Match Score<br><small><span style='color:green;'>Green ≥ 0.80</span>"
        "<span style='color:orange;'>Orange ≥ 0.60</span>"
        "<span style='color:red;'>Red &lt; 0.60</span></small>"
    )

@admin.register(RecognitionSchedule)
class RecognitionScheduleAdmin(ImportExportModelAdmin):
    list_display = ('name', 'get_weekdays_display', 'start_time', 'end_time', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('name',)

    def get_weekdays_display(self, obj):
        return ", ".join(obj.weekdays)
    get_weekdays_display.short_description = "Weekdays"

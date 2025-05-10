# dashboard/utils.py
import psutil
import subprocess
import platform


def get_gpu_details():
    details = {
        "name": "Unknown",
        "power_draw": "-",
        "memory_clock": "-",
        "graphics_clock": "-",
        "temperature": "-",
        "memory_used": 0,
        "memory_total": 0,
    }
    try:
        query = [
            'nvidia-smi',
            '--query-gpu=name,power.draw,clocks.mem,clocks.gr,temperature.gpu,memory.used,memory.total',
            '--format=csv,nounits,noheader'
        ]
        output = subprocess.check_output(query).decode().strip()
        if output:
            parts = [p.strip() for p in output.split(',')]
            (details["name"], details["power_draw"], details["memory_clock"],
             details["graphics_clock"], details["temperature"],
             used, total) = parts[:7]
            details["memory_used"] = int(used)
            details["memory_total"] = int(total)
    except Exception:
        pass
    return details


def get_system_stats():
    cpu_model = platform.processor()
    if not cpu_model:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        break
        except Exception:
            cpu_model = "Unknown"

    cpu = {
        'usage': psutil.cpu_percent(interval=1),
        'model': cpu_model,
        'cores': psutil.cpu_count(logical=True)
    }
    ram = {
        'usage': psutil.virtual_memory().percent,
        'total': round(psutil.virtual_memory().total / (1024 ** 3), 2)  # in GB
    }
    gpu = get_gpu_details()
    return {
        'cpu': cpu,
        'ram': ram,
        'gpu': gpu
    }



# # dashboard/utils.py
# import psutil
# import subprocess
#
#
# def get_gpu_details():
#     details = {
#         "name": "Unknown",
#         "power_draw": "-",
#         "memory_clock": "-",
#         "graphics_clock": "-",
#         "temperature": "-",
#         "memory_used": 0,
#         "memory_total": 0,
#     }
#     try:
#         query = [
#             'nvidia-smi',
#             '--query-gpu=name,power.draw,clocks.mem,clocks.gr,temperature.gpu,memory.used,memory.total',
#             '--format=csv,nounits,noheader'
#         ]
#         output = subprocess.check_output(query).decode().strip()
#         if output:
#             parts = [p.strip() for p in output.split(',')]
#             (details["name"], details["power_draw"], details["memory_clock"],
#              details["graphics_clock"], details["temperature"],
#              used, total) = parts[:7]
#             details["memory_used"] = int(used)
#             details["memory_total"] = int(total)
#     except Exception:
#         pass
#     return details
#
#
# def get_system_stats():
#     cpu = psutil.cpu_percent(interval=1)
#     ram = psutil.virtual_memory().percent
#     gpu = get_gpu_details()
#     return {
#         'cpu': cpu,
#         'ram': ram,
#         'gpu': gpu
#     }
#

# # dashboard/utils.py
# import psutil
# import subprocess
#
# def get_system_stats():
#     cpu = psutil.cpu_percent(interval=1)
#     ram = psutil.virtual_memory().percent
#     gpu_used, gpu_total = 0, 0
#     try:
#         output = subprocess.check_output([
#             'nvidia-smi', '--query-gpu=memory.used,memory.total',
#             '--format=csv,nounits,noheader'
#         ]).decode().strip()
#         if output:
#             gpu_used, gpu_total = map(int, output.split(','))
#     except Exception:
#         pass
#     return {
#         'cpu': cpu,
#         'ram': ram,
#         'gpu': {
#             'used': gpu_used,
#             'total': gpu_total
#         }
#     }
#

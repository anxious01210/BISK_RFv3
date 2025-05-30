# script_runner.py

# This dictionary is used to hold buffers for streaming script output per job_id
job_output_buffers = {}

# Dictionary to keep track of currently running subprocesses
running_processes = {}
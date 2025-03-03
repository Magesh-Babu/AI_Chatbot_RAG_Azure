# Gunicorn configuration file
import multiprocessing

max_requests = 500
max_requests_jitter = 25

log_file = "-"

bind = "0.0.0.0:8000"

worker_class = "uvicorn.workers.UvicornWorker"
workers = 1
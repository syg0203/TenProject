# gunicorn.conf.py
from datetime import datetime
import os

wsgi_app = "src.main:app"
bind = "0.0.0.0:2030"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 0
keepalive = 30

print(f'app ready on {bind}')
# gunicorn.conf.py
from datetime import datetime
import os

wsgi_app = "src.main:app"
bind = "0.0.0.0:2030"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 30 # 리턴이 일정 시간동안 안오면 restart
keepalive = 30
max_requests = 5 # worker가 정해진 횟수만큼 요청받은후 restart
reload = False # reload when code change

accesslog = "./logs/access/access_.log"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

errorlog = "./logs/error/error.log"

#logconfig = "./logs/uvicorn_log.ini"
loglevel = "info"

print(f'app ready on {bind}')
import sys
import os
from dotenv import load_dotenv

root_path = os.getcwd()
sys.path.insert(1, root_path)

load_dotenv()

class Configs:
    SLACK_URL = os.environ.get("SLACK_URL")
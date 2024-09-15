import requests
import sys
import os

root_path = os.getcwd()
sys.path.insert(1,root_path)

import traceback
from modules.core.configs import Configs

class common:
    def exception_func(e, yg=True,status_code=500):
        text = (
            f"*Summary : * {str(e)}\n\n"
            f"*Details*\n```{traceback.format_exc()}```"
            )
        common.logging_alert(text, yg=yg)

    def logging_alert(text,yg=True):
        url = Configs.SLACK_URL
        person = "bruce"
        if yg:
            person = "YG"
        icon_emoji = f":{person}:"
        username = f"{person}_Alertmanager"
        header = {"Content-type":"application/json"}
        attachments = [
            {
                "pretext":"*Error Alert*",
                "title":f"Fix User : {person}",
                "color":"#f54242",
                "text":text,
                "mrkdwn_in":["text","pretext"]
                }
            ]

        data = {
            "icon_emoji":icon_emoji,
            "username":username,
            "attachments":attachments,
        }
        return requests.post(url, headers=header, json=data, verify=False)
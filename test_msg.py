
from slack_sdk import WebClient

myToken = "xoxb-1702501177444-1689568357894-oVvUyKlVTQhKJAqHaWmzxj9x"
slack = WebClient(myToken)
slack.chat_postMessage(channel = "#stock", text = "aws test")

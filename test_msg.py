import requests 
def send_message_to_slack(text): 
    url = "https://hooks.slack.com/services/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" 
    payload = { "text" : text } 
    requests.post(url, json=payload)

send_message_to_slack("AWS Send Message Using Python")

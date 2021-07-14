def incomingWebhookMessage(message):
    slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
    slack.notify(text=message, channel="#auto_trade",username="bot",mrkdwn=True)

incomingWebhookMessage("aws testss_please")

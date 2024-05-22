import requests

TELEGRAM_TOKEN = "6451639539:AAGAKKSqYumqiBlGVCYcmZI8T84r55Ko0_k"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
WEBHOOK_URL = "https://prime-hermit-ethical.ngrok-free.app/webhook" 

response = requests.post(TELEGRAM_API_URL, data={"url": WEBHOOK_URL})
print(response.json())
import requests

response=requests.post(
    "http://localhost:8000/hepha/invoke",
    json={'input':{'topic':"What is the weather in london"}})

print(response)
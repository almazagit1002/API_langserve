import requests

# URL of the FastAPI server
url = "http://localhost:8000/Hepha_api"

# String to be modified
input_string = "weather in london"
url_with_param = f"{url}?input_string={input_string}"


response = requests.post(url_with_param)

# Check if the request was successful
if response.status_code == 200:
    modified_string = response.text
    print("Answer:", modified_string)
else:
    print("Error:", response.text)

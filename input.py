import requests

# test api
url = "http://localhost:5000/search"
files = {"image": open("testimage/fox.jpeg", "rb")}
response = requests.post(url, files=files)

print(response.status_code)
print(response.json())

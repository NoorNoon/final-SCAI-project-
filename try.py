import requests

url = "http://127.0.0.1:8000/predict"
files = {'images': open(r'C:\Users\zsaff\PycharmProjects\PythonProject2\Facial Data.v2i.yolov8\test\images\Emotions.jpg', 'rb')}  # غيّري الاسم

response = requests.post(url, files=files)

print(response.json())

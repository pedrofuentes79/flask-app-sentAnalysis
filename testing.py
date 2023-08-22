import requests

url = 'http://localhost:5000/predict/api'
data = {'input_sentence': "it came with less amount of food than expected, I could not find it tasty"}

response = requests.post(url, json=data)
response_data = response.json()

print("Input Sentence:", response_data['input_sentence'])
print("Sentiment:", response_data['sentiment'])
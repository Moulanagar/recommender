import requests

url = "https://recommender-5-inq4.onrender.com/recommend"
data = {
    "client_id": 1,
    "required_skills": ["python", "flask"],
    "budget": 200
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())

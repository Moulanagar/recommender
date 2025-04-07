
# 💼 Freelancer Recommender System

A hybrid recommendation system that suggests the top freelancers for clients based on project requirements like skills, budget, and timeline. Built using a combination of **Collaborative Filtering** and **Content-Based Filtering**, and deployed via a Flask API.

---

## 📊 Dataset Overview

### 📁 Freelancer Dataset
Simulated data representing freelancer portfolios with the following structure:
```json
{
  "freelancer_id": 284,
  "name": "Freelancer 284",
  "experience_years": 14,
  "hourly_rate": 148,
  "skills": ["Git", "GraphQL", "GitHub", "React", "MongoDB", "C++", "TensorFlow", "NumPy"]
}
```

### 📁 Interaction Dataset
Simulated data capturing past interactions between clients and freelancers, used to support collaborative filtering.

---

## 🧠 Model Selection & Training Process

### ✅ Hybrid Model Design

- **Collaborative Filtering:**  
  - Algorithm: Singular Value Decomposition (SVD)
  - Library: `surprise`  
  - Learns preferences based on historical client-freelancer ratings.
  
- **Content-Based Filtering:**  
  - Skills match using **TF-IDF Vectorizer** and **Cosine Similarity**
  - Budget and experience filters

### 🧮 Final Score Calculation

```
Final Score = 0.6 × Content-Based Score + 0.4 × Collaborative Filtering Score
```

### 🏋️ Training

```bash
python train_model.py
```

This script trains and saves the collaborative model (`model.pkl`) using the interaction dataset.

---

## 🚀 API Functionality

### 🔗 Endpoint

```http
POST /recommend_web
```

### 🔧 Input (HTML Form)

- `client_id`: Numeric ID of the client
- `budget`: Budget for the project
- `timeline`: Project timeline (in days)
- `required_skills`: List of required skills

### 📤 Output

- Returns top 5 freelancer recommendations based on hybrid model.
- Displays name, skills, experience, and hourly rate.  
*(Final score is used internally and not shown.)*

---

## 🌐 Deployment

### Hosted On

- **Render**

### Deployment Steps

1. Push code to GitHub.
2. Connect GitHub repo to Render.
3. Set up environment with:
   - Python 3.10+
   - `requirements.txt`
4. Add build/start commands:
   ```bash
   gunicorn app:app
   ```

---

## 🧪 How to Test the API

### Local Testing

1. Clone the repo:
   ```bash
   git clone https://github.com/Moulanagar/recommender.git
   cd recommender
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. Open browser and go to:
   ```
   http://localhost:5000
   ```

5. Fill in form and submit to view recommendations.

---

## 📎 Links

- 🔗 GitHub Repository: https://github.com/Moulanagar/recommender
- 🌍 Hosted API: https://recommender-14.onrender.com

---

## 🏁 Conclusion

This project demonstrates how a hybrid recommender system can intelligently match clients to the most suitable freelancers by combining historical interaction data and semantic skill matching.

Many improovements can be done and model can be improoved based on availability of data.
---

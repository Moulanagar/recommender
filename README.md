# Freelancer Recommender System

A hybrid recommendation system built using Collaborative Filtering and Content-Based Filtering to suggest top freelancers based on client preferences.

---

## 🔍 Model Selection & Training

- **Model:** Singular Value Decomposition (SVD) for collaborative filtering.
- **Input Features:** Skills, experience, hourly rate, client preferences.
- **Output:** List of top freelancer recommendations for a given client with matched skills and budget.
- **Hybrid Logic:** Combines SVD-predicted ratings with a similarity match based on skills and budget.

### Training the model
```bash
python train_model.py

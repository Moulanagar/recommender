from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

app = Flask(__name__)

# Load data (no pre-trained model anymore)
with open("model1 (3).pkl", "rb") as f:
    data = pickle.load(f)

freelancers_df = data["freelancers_df"]
interactions = data["interactions"]

# -------------------------
# Content-based scoring
# -------------------------
def compute_content_score(job_input, df):
    df = df.copy()
    df['skills_str'] = df['skills'].apply(lambda x: ' '.join(x))
    job_skills_str = ' '.join(job_input['required_skills'])

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['skills_str'].tolist() + [job_skills_str])
    similarity = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])
    df['similarity'] = similarity

    df['exp_score'] = df['experience_years'] / df['experience_years'].max()
    df['rating_score'] = df['avg_rating'] / 5
    df['job_score'] = df['jobs_completed'] / df['jobs_completed'].max()

    daily_rate = df['hourly_rate'] * 8 * 3
    df['budget_score'] = job_input['budget'] / (daily_rate + 1)
    df['budget_score'] = df['budget_score'].apply(lambda x: 1 if x >= 1 else x)

    df['content_score'] = (
        0.4 * df['similarity'] +
        0.2 * df['exp_score'] +
        0.2 * df['rating_score'] +
        0.1 * df['job_score'] +
        0.1 * df['budget_score']
    )
    return df

# -------------------------
# Collaborative Filtering (dynamic SVD training)
# -------------------------
def get_cf_scores(client_id, df, interactions):
    df = df.copy()
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(interactions[["client_id", "freelancer_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    model = SVD()
    model.fit(trainset)

    cf_scores = []
    for _, row in df.iterrows():
        prediction = model.predict(client_id, row['freelancer_id'])
        cf_scores.append(prediction.est)

    df['cf_score'] = cf_scores
    return df

# -------------------------
# API Endpoint
# -------------------------
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    required_fields = ['client_id', 'required_skills', 'budget']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing fields'}), 400

    freelancers = compute_content_score(data, freelancers_df)
    freelancers = get_cf_scores(data['client_id'], freelancers, interactions)

    freelancers['final_score'] = (
        0.6 * freelancers['content_score'] +
        0.4 * freelancers['cf_score']
    )
    top5 = freelancers.sort_values(by='final_score', ascending=False).head(5)

    return jsonify(top5[['freelancer_id', 'name', 'skills', 'final_score']].to_dict(orient='records'))

# -------------------------
# Health check
# -------------------------
@app.route('/')
def home():
    return "Freelancer Recommendation System is running!"

# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, request, jsonify, render_template
from surprise import Dataset, Reader, SVD
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load the data (includes freelancers_df and interactions)
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

freelancers_df = data["freelancers_df"]
interactions = data["interactions"]

# ---------- Recommender Functions ----------

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

    # Update budget score using timeline
    daily_rate = df['hourly_rate'] * 8
    estimated_cost = daily_rate * job_input['timeline']
    df['budget_score'] = job_input['budget'] / (estimated_cost + 1)
    df['budget_score'] = df['budget_score'].apply(lambda x: 1 if x >= 1 else x)

    df['content_score'] = (
        0.4 * df['similarity'] +
        0.2 * df['exp_score'] +
        0.2 * df['rating_score'] +
        0.1 * df['job_score'] +
        0.1 * df['budget_score']
    )
    return df

def get_cf_scores(client_id, df, interactions):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(interactions[["client_id", "freelancer_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)

    cf_scores = []
    for _, row in df.iterrows():
        prediction = model.predict(client_id, row['freelancer_id'])
        cf_scores.append(prediction.est)
    df['cf_score'] = cf_scores
    return df

def hybrid_recommendation(job_input, freelancers_df, interactions):
    freelancers_df = compute_content_score(job_input, freelancers_df)
    freelancers_df = get_cf_scores(job_input["client_id"], freelancers_df, interactions)

    freelancers_df['final_score'] = (
        0.6 * freelancers_df['content_score'] + 0.4 * freelancers_df['cf_score']
    )

    avg_daily_rate = freelancers_df['hourly_rate'].mean() * 8 * job_input['timeline']
    if job_input['budget'] < avg_daily_rate * 0.5:
        return [{
            "name": "Budget too low",
            "skills": [],
            "experience_years": 0,
            "hourly_rate": 0,
            "final_score": 0,
            "message": (
                f"Your budget ${job_input['budget']} may be too low "
                f"for freelancers with the required skills and timeline. "
                f"Consider increasing it above ${int(avg_daily_rate * 0.5)}."
            )
        }]
    
    top5 = freelancers_df.sort_values(by='final_score', ascending=False).head(5)
    return top5[['freelancer_id', 'name', 'skills', 'experience_years', 'hourly_rate', 'final_score']].to_dict(orient='records')

# ---------- Routes ----------

@app.route('/')
def home():
    return render_template('index.html', recommendations=None)

@app.route('/recommend_web', methods=['POST'])
def recommend_web():
    client_id = int(request.form.get('client_id'))
    budget = float(request.form.get('budget'))
    skills = request.form.getlist('required_skills')
    timeline = int(request.form.get('timeline'))  # NEW timeline input

    job_input = {
        'client_id': client_id,
        'budget': budget,
        'required_skills': skills,
        'timeline': timeline
    }

    recommendations = hybrid_recommendation(job_input, freelancers_df, interactions)
    return render_template('index.html', recommendations=recommendations)

# ---------- Run ----------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

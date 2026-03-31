# PMRT Predictor — Web App

A Flask web application that accepts a CSV dataset, validates encoding,
runs a logistic regression model, reports missing values, and returns predictions.

---

## Project Structure

```
pmrt_predictor/
├── app.py               # Flask backend + model logic
├── templates/
│   └── index.html       # Frontend UI
├── requirements.txt
├── Procfile             # For Render / Railway deployment
└── README.md
```

---

## Running Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:5000
```

---

## Free Hosting — Recommended: Render.com

**Render** offers a free tier for web services (Flask included) with no credit card required.

### Steps

1. Push this folder to a GitHub repository.
2. Go to https://render.com and sign up (free).
3. Click **New → Web Service** and connect your GitHub repo.
4. Set the following:
   - **Runtime**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app`
5. Click **Create Web Service**.

Render will build and deploy automatically. You get a public URL like
`https://pmrt-predictor.onrender.com`.

> ⚠️ **Free tier note**: The service "sleeps" after 15 minutes of inactivity.
> The first request after sleep takes ~30 seconds to wake up. This is normal.

---

## Alternative Free Hosts

| Service | Notes |
|---|---|
| **Railway.app** | 500 free hours/month; faster cold starts than Render |
| **Fly.io** | Free tier with more control; requires CLI setup |
| **PythonAnywhere** | Easy Flask hosting; free tier has outbound network limits |

---

## CSV Format

Your uploaded CSV must have these exact column headers:

| Column | Accepted Values |
|---|---|
| Age at diagnosis | Numeric (years) |
| Maximum dimension/size of pre-operative tumor on any imaging modality (mm) | Numeric |
| Maximum size of pre-operative suspicious lymph node... | Numeric |
| Presence of pre-operative palpable axillary lymph node | Y / N |
| Diagnostic imaging: US | Y / N |
| Histological subtype: IDC | Y / N |
| Histological subtype: DCIS | Y / N |
| Lymphovascular invasion | Y / N |
| Presence of carcinoma on axillary lymph node biopsy | Y / N / NA |
| HER status | positive / negative |
| ER status | positive / negative |
| Pre-operative biopsy method | core needle biopsy / surgical biopsy / fine needle aspirate |

Missing values may be left blank. The app will impute them using training-set means.

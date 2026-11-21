# Vercel Deployment Package: US Accident Severity Model

This package contains the pre-trained machine learning model and FastAPI web service configured for seamless deployment to **Vercel Serverless Functions**.

---

## 📁 Package Contents

| File / Folder | Purpose |
| :--- | :--- |
| `accident_severity_model.joblib` | Production model bundle (compressed, 2.9 MB) |
| `accident_severity_model.pkl` | Alternative pickle model artifact (7.3 MB) |
| `model_details.json` | Model schema metadata, raw features, and classes |
| `validation_metrics.csv` | Cross-validation and test benchmark metrics |
| `reporting_shift_metrics.csv` | Reporting shift robustness metrics |
| `model_inference.py` | Feature engineering pipeline & model wrapper |
| `app.py` | FastAPI application, validation, & web UI |
| `requirements.txt` | Python runtime dependencies for Vercel |
| `vercel.json` | Vercel routing and serverless function rewrite rules |
| `.vercelignore` | Excludes unnecessary files from upload bundle |
| `api/index.py` | Vercel serverless entrypoint |

---

## 🚀 How to Host on Vercel

### Option A: Deploy via Vercel CLI

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm i -g vercel
   ```

2. **Navigate to the `hosting` directory**:
   ```bash
   cd hosting
   ```

3. **Deploy**:
   ```bash
   vercel
   ```
   Follow the prompts to log in and select/create a project.

4. **Deploy to Production**:
   ```bash
   vercel --prod
   ```

---

### Option B: Deploy via GitHub / GitLab / Bitbucket

1. Push your repository to GitHub.
2. In the [Vercel Dashboard](https://vercel.com/dashboard), click **Add New Project**.
3. Import your Git repository.
4. Set the **Root Directory** setting to `hosting` in the project configuration.
5. Click **Deploy**.

---

## ⚙️ Environment Variables (Optional)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_FILE` | `accident_severity_model.joblib` | Set to `accident_severity_model.pkl` if you want to switch to the `.pkl` artifact. |

---

## 🌐 Endpoints

- `GET /`: Interactive web dashboard to test model predictions from any browser.
- `GET /docs`: Interactive Swagger UI with API schema and tester.
- `GET /health`: Healthcheck & model readiness verification.
- `GET /model/info`: Metadata, features, class weights, and validation metrics.
- `POST /predict`: Predict severity for a single accident record (JSON).
- `POST /predict/batch`: Predict severity for multiple accident records (up to 1,000).

---

## 💻 Local Testing Before Deploying

To test the hosting bundle locally:

```bash
cd hosting
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000` in your web browser.

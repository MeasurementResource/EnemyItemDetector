# Item Relationship Detector

A Streamlit web app for detecting clone, variant, and enemy item pairs in competency-based assessment item banks using the Claude API.

**Shared-key mode** — users just open the URL and go. No Anthropic account needed.

## Methodology

Based on:
- **Micir et al. (2022)** — NLP + ML pipeline for enemy item detection (NBME "Smokey" system)
- **Stevenor et al. (2025)** — Semantic similarity (stem + answer key) outperforms lexical approaches

---

## Deploying to Streamlit Community Cloud (recommended)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/enemy-item-detector.git
git push -u origin main
```

> ⚠️ `.streamlit/secrets.toml` is already in `.gitignore`. Never commit your API key.

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Select your repo, branch (`main`), and set main file to `app.py`
4. Click **Advanced settings → Secrets** and paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

5. Click **Deploy** — share the resulting URL with your team.

---

## Running locally

```bash
pip install -r requirements.txt
# Add your key to .streamlit/secrets.toml
streamlit run app.py
```

## Item bank format

Minimum: a **stem** column. Optional but recommended: Item ID, Correct answer key, Option A/B/C/D, Content area.

## Relationship types

| Type | Definition |
|------|-----------|
| **Clone** | Virtually identical — same assessment point, trivially reworded |
| **Variant** | Same learning objective, different wording or distractors |
| **Enemy** | One item reveals or clues the answer to the other |

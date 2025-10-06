# AI Blasting Suite — Streamlit (with Login)

## Local run
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
python tools/create_hashed_password.py   # paste the hash into secrets
streamlit run streamlit_app.py

## Deploy (Streamlit Community Cloud)
- Push this folder to GitHub.
- New app → entry file: `streamlit_app.py`.
- In App → Settings → Secrets, paste `.streamlit/secrets.toml.example` and set the hashed password.

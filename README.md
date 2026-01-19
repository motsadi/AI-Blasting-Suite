## AI Blasting Suite — Web Deployment (React + FastAPI)

This repo contains the original Streamlit bundle in `AI_Blasting_Suite_Full_Streamlit_Bundle/` plus a new web architecture:

- `frontend/`: React (Vercel)
- `backend/`: FastAPI (Google Cloud Run)

### Important constraint

The Python “core” modules inside `AI_Blasting_Suite_Full_Streamlit_Bundle/` are treated as **read-only**. The backend wraps them via HTTP without modifying their functions.

### High-level flow

- **Auth/DB**: InstantDB (email verification-code login)
- **Frontend**: calls backend API with `Authorization: Bearer <instantdb_token>`
- **Backend**: verifies the token (implementation pending; depends on InstantDB verification mechanism), performs model inference / calculations, reads & writes artifacts via GCS.

### References

- Repo: `https://github.com/motsadi/AI-Blasting-Suite.git`


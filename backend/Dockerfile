FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Build context should be the repo root (`AI-Blasting-Suite/`), not `backend/`.
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app
COPY AI_Blasting_Suite_Full_Streamlit_Bundle /app/core_bundle

ENV CORE_BUNDLE_PATH=/app/core_bundle

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]


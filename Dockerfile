# Serves the fraud model as a REST API. The model is trained during the build,
# so the image is self-contained and reproducible.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
RUN python src/fraud/train.py

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "fraud.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

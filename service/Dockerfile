FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy the whole repo so we can install the library code
COPY . /srv

# Install service deps + install registry library (editable)
RUN pip install --no-cache-dir -r service/requirements.txt && \
    pip install --no-cache-dir -e .

EXPOSE 8095
CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8095"]

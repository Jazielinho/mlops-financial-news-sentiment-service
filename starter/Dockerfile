FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

COPY requirements-prod.txt requirements-prod.lock ./

RUN uv pip install \
    --system \
    --no-cache \
    --torch-backend cpu \
    -r requirements-prod.lock

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

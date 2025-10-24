FROM python:3.11 AS builder

WORKDIR /app

COPY requirements.txt .

RUN python -m venv /opt/venv

RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY frontend/ ./frontend
COPY backend/ ./backend
COPY main.py ./main.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

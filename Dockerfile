FROM python:3.11-slim

# HF Spaces runs as a non-root user; create one that matches
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Install dependencies first (layer cache)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user . .

EXPOSE 7860

# Single worker — environment state is in-memory per session; multiple workers
# would cause reset() and step() to hit different processes and lose state.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

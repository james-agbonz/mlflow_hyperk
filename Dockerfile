FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Create directory for code
RUN mkdir -p /app/code

# Copy only requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create empty .env file if none exists (prevents errors)
RUN touch /app/.env
RUN pip install captum tensorflow


# Default command
CMD ["python", "--version"]

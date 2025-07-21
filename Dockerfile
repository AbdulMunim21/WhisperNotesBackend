# Use official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with prebuilt binaries
RUN pip install --upgrade pip && pip install --prefer-binary -r requirements.txt

# Copy the entire app
COPY . .

# Expose port Gunicorn will run on
EXPOSE 8080

# Start with Gunicorn (entry point: main.app)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]

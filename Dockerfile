FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with prebuilt binaries
RUN pip install --upgrade pip && pip install --prefer-binary -r requirements.txt

# Download model from GitHub release and extract it
RUN wget https://github.com/AbdulMunim21/WhisperNotesBackend/releases/download/v1.1/model.zip && \
    unzip model.zip -d temp_model && \
    # Find the actual model folder inside temp_model and move contents to ./model
    mkdir -p model && \
    find temp_model -mindepth 1 -maxdepth 1 -type d -exec cp -r {}/. model/ \; && \
    rm -rf temp_model model.zip && \
    echo "Model contents:" && ls -la model

# Remove .venv if it exists (in case it was added to context)
RUN rm -rf .venv

# Copy the rest of your application
COPY . .

# Expose port for Gunicorn
EXPOSE 8080

# Run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
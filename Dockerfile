FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# # Create NLTK data directory
# RUN mkdir -p /usr/share/nltk_data

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with prebuilt binaries
RUN pip install --upgrade pip && pip install --prefer-binary -r requirements.txt

# # Download NLTK data
# RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data'); nltk.download('punkt_tab', download_dir='/usr/share/nltk_data')"

# Remove .venv if it exists (in case it was added to context)
RUN rm -rf .venv

# Copy the rest of your application
COPY . .

# # Set environment variable for NLTK data
# ENV NLTK_DATA=/usr/share/nltk_data

# Expose port (Railway will set PORT environment variable)
EXPOSE 8080

# Run the Flask app with Gunicorn
# Railway automatically sets the PORT environment variable
CMD gunicorn --bind 0.0.0.0:$PORT main:app
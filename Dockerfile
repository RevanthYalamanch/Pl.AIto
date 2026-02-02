# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8080

# Run Streamlit (configured for Cloud Run's port requirement)
CMD ["streamlit", "run", "app_5.py", "--server.port=8080", "--server.address=0.0.0.0"]
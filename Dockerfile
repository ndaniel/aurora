# Use official Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list first (for build caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app when container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


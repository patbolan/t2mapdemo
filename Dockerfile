# Use an official Python runtime as the base
FROM python:3.13-slim

# Update pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Gunicorn will listen on
EXPOSE 5000

# Run Gunicorn (adjust module:callable and workers as needed)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]

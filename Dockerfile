# Set the base image
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app module to the container
COPY app.py .

# Expose port 5000 for Flask
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

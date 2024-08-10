# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app/AutoTestify/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    git \
    gcc \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/AutoTestify/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/AutoTestify/

# Make port 8000 available to the world outside this container
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Define the command to start the Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8888"]

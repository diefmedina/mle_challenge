# Use the official Python image from Docker Hub
FROM python:3.10.13

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=./challenge
ENV FLIGHT_DELAY_DATA=FLIGHT_DELAY_DATA=data/data.csv
ENV FLIGHT_DELAY_MODEL=flight_delay_model.joblib


# Copy your project files into the container
COPY . .

# Expose the port your FastAPI app is running on (8000 by default)
EXPOSE 8000

# Command to run your FastAPI application with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Use the specific Python version
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Define the command to run your application
# ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "main:app", "--worker-class", "uvicorn.workers.UvicornWorker"]
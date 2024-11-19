# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the needed application file and directories
COPY models/ /app/models/
COPY data/ /app/data/
COPY notebooks/ /app/notebooks/
COPY README.md /app/README.md

# Specify the command to run the application
ENTRYPOINT ["python", "notebooks/main.py"]
CMD ["--mode", "inference", "--json_file", "sample.json"]

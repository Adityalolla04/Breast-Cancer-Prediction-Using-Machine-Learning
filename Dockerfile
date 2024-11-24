# Use the official Python image.
FROM python:3.11

# Set the working directory in the container.
WORKDIR /app

# Copy the current directory contents into the container at /app.
COPY . /app

# Install dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port the app runs on.
EXPOSE 8501

# Define environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501

# Run Streamlit
CMD ["streamlit", "run", "app.py"]

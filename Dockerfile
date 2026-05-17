# 1. Start from a lightweight official Python image
FROM python:3.13-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code and the fine-tuned model into the container
COPY . .

# 6. Run the python file main.py
RUN python main.py

# 7. Run the uvicorn command, after the creation of image
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

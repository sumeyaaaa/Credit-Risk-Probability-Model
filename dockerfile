FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install deps first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code and mlruns folder
COPY . /app

# You can also explicitly copy mlruns if it's outside project root
# COPY mlruns /app/mlruns

# Expose port 8000 (optional)
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

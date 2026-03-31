FROM python:3.10-slim

WORKDIR /app

# Install dependencies first
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy entire project
COPY . /app

# Install your src package
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
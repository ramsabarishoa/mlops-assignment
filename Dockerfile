FROM python:3.13.2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model during the image build
RUN python model.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
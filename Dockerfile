# Base image: Python versi ringan
FROM python:3.11-slim

# Folder kerja di dalam container
WORKDIR /app

# Copy daftar library dulu, install duluan
# (biar kalau kode berubah tapi requirements.txt tidak, Docker tidak install ulang dari nol)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode & model yang sudah dilatih ke dalam container
COPY preprocessing.py .
COPY main.py .
COPY models/ ./models/

# Kasih tau Docker: container ini akan "mendengarkan" di port 8000
EXPOSE 8000

# Perintah yang dijalankan saat container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
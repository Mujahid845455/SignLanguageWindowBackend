# 1️⃣ Python version LOCK (MOST IMPORTANT)
FROM python:3.10.13-slim

# 2️⃣ System dependencies (MediaPipe & OpenCV ke liye)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ Working directory
WORKDIR /app

# 4️⃣ Requirements copy & install
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5️⃣ App code copy
COPY sign.py .

# 6️⃣ Render PORT expose
EXPOSE 10000

# 7️⃣ Start command
CMD ["python", "sign.py"]

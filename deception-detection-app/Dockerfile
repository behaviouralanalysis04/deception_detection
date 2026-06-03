FROM python:3.9-slim

# Install all system dependencies required for dlib, av, and OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    cmake \
    build-essential \
    g++ \
    ffmpeg \
    libblas-dev \
    liblapack-dev \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Download dlib's shape predictor
RUN python -c "import urllib.request, bz2, shutil, os; url='http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'; urllib.request.urlretrieve(url, 'tmp.bz2'); with bz2.open('tmp.bz2','rb') as src, open('shape_predictor_68_face_landmarks.dat','wb') as dst: shutil.copyfileobj(src, dst); os.remove('tmp.bz2')"

# Create directories
RUN mkdir -p model data

# Copy application files
COPY app.py .
COPY video_processor.py .
COPY audio_processor.py .
COPY utils.py .
COPY model/ ./model/
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
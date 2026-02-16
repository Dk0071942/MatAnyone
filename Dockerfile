FROM python:3.10-slim

ARG FFMPEG_VERSION=8.0.1
ARG TORCH_WHEEL_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FFMPEG_EXE=/usr/local/bin/ffmpeg \
    IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libx264-dev \
    nasm \
    pkg-config \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers; \
    make -C /tmp/nv-codec-headers install PREFIX=/usr/local; \
    rm -rf /tmp/nv-codec-headers

RUN set -eux; \
    curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o /tmp/ffmpeg.tar.xz; \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp; \
    cd "/tmp/ffmpeg-${FFMPEG_VERSION}"; \
    ./configure \
      --prefix=/usr/local \
      --disable-debug \
      --disable-doc \
      --disable-ffplay \
      --enable-gpl \
      --enable-libx264 \
      --enable-nvenc \
      --enable-shared \
      --disable-static; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    ffmpeg -version | head -n 1 | grep -q "ffmpeg version ${FFMPEG_VERSION}"; \
    ffmpeg -hide_banner -encoders | grep -q "h264_nvenc"; \
    rm -rf /tmp/ffmpeg.tar.xz "/tmp/ffmpeg-${FFMPEG_VERSION}"

COPY . /app

RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url ${TORCH_WHEEL_INDEX_URL} && \
    pip install -e . && \
    pip install -r hugging_face/requirements.txt

RUN groupadd --gid ${APP_GID} app && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/bash app && \
    mkdir -p /app/pretrained_models /app/hugging_face/results && \
    chown -R app:app /app

WORKDIR /app/hugging_face

EXPOSE 7860

USER app

CMD ["python", "app_auto_keying.py"]

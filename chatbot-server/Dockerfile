FROM python:3.11-slim

WORKDIR /chatbot-server

RUN pip install --upgrade pip
ENV PIP_DEFAULT_TIMEOUT=1200

# Install runtime dependencies, CA certificates, and Rust toolchain for builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl openssl build-essential git \
    libglib2.0-0 libgl1 libjpeg-dev zlib1g-dev libopenjp2-7-dev ffmpeg && \
    update-ca-certificates && \
    sed -i 's/@SECLEVEL=2/@SECLEVEL=1/g' /etc/ssl/openssl.cnf || true && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    /bin/bash -c "source $HOME/.cargo/env" && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt /chatbot-server
# Install CPU-only PyTorch wheels from the official PyTorch index first to avoid
# large downloads failing during the combined requirements install. This uses
# the CPU wheel which is appropriate for CI/build without GPU.
RUN /bin/bash -c "source $HOME/.cargo/env && \
    pip --no-cache-dir install --prefer-binary --timeout 1200 \
    torch==2.4.1 torchvision==0.18.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html || true"

RUN /bin/bash -c "source $HOME/.cargo/env && pip install --no-cache-dir --timeout=1200 -r requirements.txt --index-url https://pypi.org/simple"

# Copy the entire project (including model.yaml)
COPY . /chatbot-server

# Set the working directory to the project root so src is on the module path
WORKDIR /chatbot-server

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
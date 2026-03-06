FROM nvcr.io/nvidia/pytorch:25.02-py3

RUN pip install -U pip setuptools

# 先装 aceso 依赖（确保这里的 requirements.txt 不再 pin numpy==1.20.0）
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace
RUN pip install -r requirements.txt

# 编译 apex（sm_120）
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDA_HOME=/usr/local/cuda


# COPY external/apex /workspace/apex
# WORKDIR /workspace/apex

# # apex requirements 这句可留可不留（留着也安全，因为都是 >=）
# RUN pip install -r requirements.txt

# ENV APEX_CPP_EXT=1
# ENV APEX_CUDA_EXT=1
# RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./

# 工具（如果 aceso 需要）
RUN apt-get update && apt-get -y install pssh coinor-cbc && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
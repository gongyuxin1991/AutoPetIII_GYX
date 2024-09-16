# 使用 Python 3.10 镜像作为基础镜像
FROM python:3.10-slim

# 安装必要的系统包和 PyTorch
RUN apt-get update && apt-get install -y nano \
    && pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 安装 Python 依赖
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN pip install --upgrade pip && pip install -r /opt/algorithm/requirements.txt

# 创建用户和设置工作目录
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm \
    && mkdir -p /opt/algorithm /input /output \
    && chown -R algorithm:algorithm /opt/algorithm /input /output

# 切换到非 root 用户
USER algorithm
WORKDIR /opt/algorithm

# 复制 NIfTI 文件到容器中的 /input 目录
COPY input/images/ct /input/images/ct/
COPY input/images/pet /input/images/pet/

# 复制数据和脚本
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm nnUNet_results /nnUNet_results
COPY --chown=algorithm:algorithm nnUnet_raw /opt/algorithm/nnUNet_raw/
COPY --chown=algorithm:algorithm nnUNet_results /opt/algorithm//nnUNet_results/
COPY --chown=algorithm:algorithm Dataset131_AutopetIII_result /Dataset131_AutopetIII_result/

# 设置环境变量
ENV PATH="/home/algorithm/.local/bin:${PATH}"
ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

# 入口点
ENTRYPOINT ["python", "process.py"]

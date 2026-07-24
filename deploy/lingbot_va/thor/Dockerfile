FROM nvcr.io/nvidia/pytorch:25.08-py3

COPY wheels/lerobot-0.6.0-py3-none-any.whl /tmp/lerobot-0.6.0-py3-none-any.whl

# Keep the CUDA-enabled PyTorch/TorchVision supplied by NVIDIA's Thor image.
RUN python3 -m pip install --no-cache-dir --no-deps /tmp/lerobot-0.6.0-py3-none-any.whl && \
    python3 -m pip install --no-cache-dir \
      'numpy==1.26.4' \
      'opencv-python-headless==4.11.0.86' \
      'Pillow>=10,<13' \
      'einops>=0.8,<0.9' \
      'draccus==0.10.0' \
      'huggingface-hub>=1.0,<2' \
      'requests>=2.32,<3' \
      'gymnasium>=1.1.1,<2' \
      'safetensors>=0.4.3,<1' \
      'packaging>=24.2,<26' \
      'termcolor>=2.4,<4' \
      'tqdm>=4.66,<5' \
      'transformers>=5.4,<5.6' \
      'diffusers>=0.27.2,<0.36' \
      'accelerate>=1.14,<2' \
      'peft>=0.18,<1'

COPY Autolife_VLA_Tools/deploy /opt/lingbot/deploy
ENV PYTHONPATH=/opt/lingbot/deploy
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
WORKDIR /opt/lingbot/deploy

ENTRYPOINT ["python3", "/opt/lingbot/deploy/lingbot_va/serve_lingbot_va.py"]

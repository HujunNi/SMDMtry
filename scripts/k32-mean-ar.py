import os
import subprocess

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["SWANLAB_API_KEY"] = "2Ugz1FxnbIK5eOwarbYAN"

conda_activate = "source /yinyongjing/anaconda3/etc/profile.d/conda.sh && conda activate smdm"

cmd = """
lightning run model \
    --node-rank=0 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    pretrain/train_mdm.py \
    --model 170 \
    --flops 10.0 \
    --micro_batch_size 32 \
    --use_latent_token \
    --model_m_ckpt ar-170M-100e18.pth \
    --zloss_K 32 \
    --extract_method meanK \
    --zloss_type cosine \
    --zloss_weight 0.2 \
    --experiment_name "k32-mean-ar" 
"""

full_command = f'{conda_activate} && cd /yinyongjing/SMDM && {cmd}'
subprocess.run(full_command, shell=True, executable="/bin/bash")

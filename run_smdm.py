import os
import subprocess

#os.environ["WANDB_API_KEY"] = "92cb3d228da9a4adfdee4ccff637982d50556cfd"
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["SWANLAB_API_KEY"] = "2Ugz1FxnbIK5eOwarbYAN"
conda_activate = "source /yinyongjing/anaconda3/etc/profile.d/conda.sh && conda activate smdm"

cmd = """
lightning run model \
    --node-rank=0 \
    --accelerator=cuda \
    --devices=2 \
    --num-nodes=1 \
    pretrain/train_mdm.py \
    --model 170 \
    --flops 10.0
"""

full_command = f'{conda_activate} && cd /yinyongjing/SMDM && {cmd}'

subprocess.run(full_command, shell=True, executable="/bin/bash")

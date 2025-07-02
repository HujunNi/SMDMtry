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
    --flops 10 \
    --micro_batch_size 32 \
    --arm_ckpt_dir workdir/scaling_debug/arm-170M-10.0 \
    --experiment_name "mdm_from_ar" 
"""

full_command = f'{conda_activate} && cd /yinyongjing/SMDM && {cmd}'

subprocess.run(full_command, shell=True, executable="/bin/bash")

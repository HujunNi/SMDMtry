import torch._dynamo
torch._dynamo.config.suppress_errors = True
import os  #added
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" #added
import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import re  #added
import sys 
from pathlib import Path
import lightning as L
import torch
import torch.nn as nn  #added
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.diffmodel import TransEncoder, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from lit_gpt.config import Config #added
from pytorch_lightning.loggers import WandbLogger
import swanlab
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import random
import argparse
import torch.nn.functional as F  #added
from pretrain.semantic_loss import SemanticLossWrapper #added
from transformers import AutoModelForMaskedLM, AutoTokenizer,PreTrainedModel #added
from huggingface_hub import hf_hub_download, snapshot_download #added

# ================ add latent_token ==================
def add_latent_tokens(tokenizer, latent_size: int = 64):
    new_tokens = [f"<LATENT_{i}>" for i in range(latent_size)] # define new tokens
    #print("Before adding latent tokens:", len(tokenizer)) 
    tokenizer.add_tokens(new_tokens) #add to tokenizer's voabulary
    return tokenizer.convert_tokens_to_ids(new_tokens)
    #print("After adding latent tokens:", len(tokenizer))
#================= end =============================

#================== added : load pretrained model ========================
def load_pretrained_m_model(
    model_m_name: str = "ModernBERT",
    mdm_ckpt: str = "mdm-113M-10e18.pth",
    ar_ckpt: str = "ar-113M-10e18.pth",
)-> nn.Module:
    """
    Load a pretrained ModernBERT encoder from Hugging Face.
    Returns a model that outputs last_hidden_state of shape [B, L, D].
    """
    if model_m_name == "ModernBERT":
        model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base",torch_dtype=torch.float16,device_map=None,attn_implementation="sdpa")
        tokenizer_m = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base",
            use_fast=True,
            local_files_only=False,
        )
        model.config.mask_token_id = tokenizer_m.mask_token_id

    elif model_m_name in ("mdm", "ar"):
        #repo_dir = snapshot_download(repo_id="nieshen/SMDM")
        subdir = "mdm" if model_m_name == "mdm" else "ar"
        filename = mdm_ckpt if model_m_name == "mdm" else ar_ckpt
        #ckpt_path = os.path.join(repo_dir, subdir, filename)
        ckpt_path = hf_hub_download(
            repo_id="nieshen/SMDM",
            subfolder=subdir,
            filename=filename,
            repo_type="model"       
        )
        data = torch.load(ckpt_path, map_location="cpu")
        if isinstance(data, dict):
            if "model" in data:
                sd = data["model"]
            elif "state_dict" in data:
                sd = data["state_dict"]
            else:
                sd = data  
        else:
            sd = data

        ckpt_rows, emb_dim = sd["transformer.wte.weight"].shape    
        m = re.search(r"-(\d+)M", filename)
        size = m.group(1)
        
        sd = { k: v for k, v in sd.items() if not k.startswith("lm_head") }
        config = Config.from_name(f"Diff_LLaMA_{size}M")
        config.padded_vocab_size = ckpt_rows - 1
        config.vocab_size = ckpt_rows - 1

        model = TransEncoder(config)
        model.load_state_dict(sd,strict=False)
        del data
    
    else:
        raise ValueError(f"Unknown model_m_name {model_m_name!r}")
    
    #model = model.to("cuda")
    model.eval()
    return model
#========================== end =======================================

#========================== added：extract Y ==========================
@torch.no_grad()
def extract_semantic_Y(
    input_ids:torch.Tensor,
    model_m: nn.Module,
    method:str = "meanK",
    K=64
) -> torch.Tensor:
    """
    input_ids: [B, L]
    method: "meanK" or "maskK"
    Returns:
        [B, K, D]
    """
    if isinstance(model_m, TransEncoder):
        _, hidden = model_m(input_ids, return_hidden=True)
    elif isinstance(model_m, PreTrainedModel):
        out = model_m(
            input_ids=input_ids,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
    else:
        raise ValueError(f"doesnt support model_m: {type(model_m)}")

    B, L, D = hidden.shape

    if method == "meanK":
        chunk = L // K
        Y = hidden.view(B, K, chunk, D).mean(dim=2)  # [B, K, D]

    elif method == "maskK":
        step = L // K
        positions = (torch.arange(K, device=hidden.device) * step).clamp(max=L - 1)
        masked = input_ids.clone()
       
        mask_tok = getattr(model_m.config, "mask_token_id", None)
        if mask_tok is None:
            raise ValueError("model doesn’t support maskK: no mask_token_id set on model_m.config")
        masked[:, positions] = mask_tok

        if isinstance(model_m, TransEncoder):
            _, hidden2 = model_m(masked, return_hidden=True)
        else:  # PreTrainedModel
            out2 = model_m(
                input_ids=masked,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden2 = out2.hidden_states[-1]

        Y = hidden2[:, positions, :]  # [B, K, D]

    else:
        raise ValueError(f"Unknown extract_method: {method}")

    return Y
#========================== end ==========================

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, help='model parameters')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    parse.add_argument('--flops', type=float, help='FLOPs, *e18')
    parse.add_argument('--batch_size', type=int, default=256, help='global_batch_size')
    #========================== added: attention_types==========================
    parse.add_argument('--use_latent_token', action='store_true', help= 'whether to use latent tokens')
    #========================== added: Z-Loss options==========================
    parse.add_argument('--zloss_type', type=str, default="cosine", choices=["cosine", "diffusion"],help="Semantic latent prediction loss type: cosine or diffusion")
    parse.add_argument('--zloss_dim', type=int, default=768, help="semantic embedding dim")
    parse.add_argument("--zloss_K", type=int, default=64, help="Number of semantic chunks per block.")
    parse.add_argument("--zloss_weight", type=float, default=0.0, help="Weight of z-loss.")
    parse.add_argument("--zloss_diffusion_steps", type=str, default="100", help="steps for diffusion sampling")
    parse.add_argument('--grad_checkpointing',action='store_true',help='Enable gradient checkpointing in the diffusion loss network')
    #========================== added: extracting Y ways ==========================
    parse.add_argument("--extract_method", type=str, choices=["meanK", "maskK"], help="How to extract K semantic vectors from each block (meanK or maskK)")
    #========================== added: choose which frozen model ==========================
    parse.add_argument("--model_m_name", type=str, choices=["ModernBERT", "mdm", "ar"], help="Which pretrained model M to use for semantic extraction")
    parse.add_argument("--mdm_ckpt", type=str, default="mdm-113M-10e18.pth",help="Filename under mdm/ in nieshen/SMDM repo")
    parse.add_argument("--ar_ckpt", type=str, default="ar-2121M-100e18.pth",help="Filename under ar/ in nieshen/SMDM repo")
    #========================== added: diffusion loss modification ==========================
    parse.add_argument('--include_unmasked_loss', action='store_true', help='Whether to also compute CE loss on unmasked positions')
    parse.add_argument('--unmask_weight', type = float, default = 0.0,  help='weight to unmasked token loss, from 0.0 ~ 1.0')
    # ========================== end =========================g
    args = parse.parse_args()
    return args

args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'  # config
out_dir = Path('workdir')

model_para_config = {
    '6': 6.294784, '19': 18.880896, '34': 33.563136, '48': 47.786688, '66': 65.54944,
    '85': 85.21408, '75': 75.38752, '113': 113.265408, '142': 141.581568, '170': 169.897728,
    '180': 179.856768, '206': 205.550464, '231': 231.24416, '268': 268.469248, '302': 302.027776,
    '336': 335.586304, '472': 471.90656, '551': 550.55744, '571': 571.001728, '629': 629.20832,
    '666': 666.168448, '717': 717.285888, '761': 761.335168, '831': 830.541312, '944': 943.796736,
    '1028': 1027.677952, '1233': 1233.213184, '1476': 1476.487168, '1678': 1677.826048, '2121': 2121.39328
}

# Hyperparameters
num_of_devices = 2
global_batch_size = int(args.batch_size / args.nodes_num)
learning_rate = 2e-4
if args.model <= 20:
    micro_batch_size = 32
elif args.model <= 50:
    micro_batch_size = 16
elif args.model <= 1000:
    micro_batch_size = 8
else:
    micro_batch_size = 4
max_step = int(args.flops * 1e12 / (6 * model_para_config[f'{args.model}'] * global_batch_size * 2048) / args.nodes_num)
warmup_steps = int(max_step / 100) if int(max_step / 100) > 100 else 100
log_step_interval = 10
eval_iters = int(100 * 1024 / global_batch_size)
save_step_interval = 5000
eval_step_interval = 999999999999 #inf


weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 2e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps




max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


# Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
train_data_config = [
    ("train_slim", 1.0)
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


def setup(
    devices: int = 2,
    train_data_dir: Path = Path("/ssdwork/yinyongjing/slimpajama62b_prepared/train"),
    val_data_dir: Path = Path("/ssdwork/yinyongjing/slimpajama62b_prepared/validation"),
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'mdm-{args.model}M-{args.flops}'
    out_dir = Path('workdir/scaling_debug') / hp_name
    
    #wandb_logger = WandbLogger(name=f'{hp_name}-mc', save_dir=out_dir, project='scaling')
    swanlab.init(project='scaling')
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    #fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger])
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision)
    fabric.print(hparams)
    #fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)
    
    config = Config.from_name(model_name)

    #==========================added: initialize tokenizer with latent tokens#==========================
    if hasattr(config, "use_latent_token") and config.use_latent_token:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        latent_token_ids = add_latent_tokens(tokenizer, config.latent_size)
        config.vocab_size = len(tokenizer)
        config.padded_vocab_size = len(tokenizer)
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        model.resize_token_embeddings(len(tokenizer))
        #print("mask_token_id:", tokenizer.mask_token_id)
    else:
        tokenizer = None
        latent_token_ids = None
   # #========================== end #==========================

    # ========================== initialze model m ==========================
    if args.zloss_weight > 0 and args.zloss_type != "none":
        model_m = load_pretrained_m_model(
            model_m_name=args.model_m_name,
            mdm_ckpt=args.mdm_ckpt,
            ar_ckpt=args.ar_ckpt,
        )
        model_m = model_m.to(fabric.device)
        model_m.config.mask_token_id = tokenizer.mask_token_id

        for p in model_m.parameters():
            p.requires_grad_(False)
        model_m.eval()
    else:
        model_m = None
    # ========================== end ==========================
    # ==========================initialize loss function ==========================
    if args.zloss_weight > 0 and args.zloss_type != "none":
        zloss_fn = SemanticLossWrapper(
        loss_type=args.zloss_type,
        target_channels=args.zloss_dim,
        z_channels=args.zloss_dim,
        depth=2,
        width=args.zloss_dim,
        num_sampling_steps=args.zloss_diffusion_steps,
        include_unmasked_loss=args.include_unmasked_loss, 
        grad_checkpointing=args.grad_checkpointing,
        )
        zloss_fn = zloss_fn.to(fabric.device)
    else:
        zloss_fn = None
    #========================== end ==========================

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
        tokenizer=tokenizer,  #added attributes
        latent_size=config.latent_size, #added attributes
        use_latent_tokens=args.use_latent_token, #added attributes
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        import re
        def extract_number(filename):
            match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
            return int(match.group(1)) if match else 0
        try:
            resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
        except:
            resume = False
    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume,model_m, zloss_fn, latent_token_ids) #added  attribute 
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume, model_m, zloss_fn, latent_token_ids):
    model = state["model"]
    optimizer = state["optimizer"]
    # added
    if latent_token_ids is not None:
        latent_size = len(latent_token_ids)  #added
    else:
        latent_size = 0
    K = args.zloss_K #added

    # if val_dataloader is not None:
    #     validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = TransEncoder(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    
    initial_iter = state["iter_num"]
    curr_iter = 0
            
    loss_func = CrossEntropyLoss(reduction='none')
    for  train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break
        
        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()
        # >>>> MODIFIED
        if args.use_latent_token:
            real_block = train_data[:, latent_size : latent_size + model.config.block_size].contiguous()  #tokens only 
            full_input = train_data[:, : latent_size + model.config.block_size].contiguous()  #tokens + latent
            noisy_input, mask_indices, p_mask = forward_process(full_input)
        # >>>> MODIFIED END
        else:
            input_ids = train_data[:, : model.config.block_size].contiguous()
            noisy_input, mask_indices, p_mask = forward_process(input_ids)

        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            # >>>> MODIFIED
            if args.use_latent_token:
                logits, hiddens = model(noisy_input, return_hidden=True)
                logits_real = logits[:, latent_size:, :]
                mask_real = mask_indices[:, latent_size:]
                p_mask_real = p_mask[:, latent_size:]
                ce_values = loss_func(logits_real[mask_real],real_block[mask_real]) / p_mask_real[mask_real] 
                masked_loss = ce_values.sum() / (train_data.size(0) * model.config.block_size)

                if args.include_unmasked_loss:
                    inv_real = ~mask_real
                    ce_inv = loss_func(logits_real[inv_real],real_block[inv_real]) / p_mask_real[inv_real]
                    unmasked_loss = ce_inv.sum() / (train_data.size(0) * model.config.block_size)
                    w = args.unmask_weight
                    ce_loss = (1 - w) * masked_loss + w * unmasked_loss
                else:
                    ce_loss = masked_loss

                if args.zloss_weight > 0 and args.zloss_type != "none" and model_m is not None:
                    Y = extract_semantic_Y(real_block,model_m,method=args.extract_method,K=K)
                    z_pred = hiddens[:, :K, :]              
                    z_loss = zloss_fn(Y, z_pred)
                    loss = ce_loss + args.zloss_weight * z_loss
                else:
                    loss = ce_loss
            # >>>> MODIFIED end

            else: #original branch
                logits = model(noisy_input)
                loss = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
                loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
                # >>>> MODIFIED
                if args.include_unmasked_loss:
                    inv = ~mask_indices
                    loss_unmasked = loss_func(logits[inv],input_ids[inv]) 
                    loss_unmasked = loss_unmasked.sum() / inv.sum()
                    w = args.unmask_weight
                    loss = (1 - w) * loss + w * loss_unmasked
                    # >>>> MODIFIED
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        # input_id: B L 
        if args.use_latent_token:
            total_lengths += real_block.size(1)
        else:
            total_lengths += model.config.block_size 
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )

            
            
            
        if val_dataloader is not None and not is_accumulating and (state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_step):
            
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, model_m, zloss_fn, latent_token_ids) #added attributes
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.barrier()
        if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step):
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        mc_loss = torch.zeros(128, device=fabric.device)  # mc_num=128
        for i in range(128):
            input_ids = val_data[:, 0 : model.config.block_size].contiguous()
            noisy_input, mask_indices, p_mask = forward_process(input_ids)
            logits = model(noisy_input)
            loss = torch.nn.functional.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            mc_loss[i] = loss

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = mc_loss.mean().item()

    losses = fabric.all_reduce(losses, reduce_op="mean")
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int, orig_block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train",tokenizer = None, latent_size: int = 64, use_latent_tokens: bool = False 
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8 if split == "train" else 1,
            orig_block_size=orig_block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
            tokenizer=tokenizer,  #added
            latent_size=latent_size, #added
            use_latent_tokens=use_latent_tokens
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
    tokenizer=None,  #added
    latent_size: int = 64,#added
    use_latent_tokens: bool = False 
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    orig_bs = block_size + 1 
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        orig_block_size=orig_bs,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train",
        tokenizer=tokenizer,  #added
        latent_size=latent_size,  #added
        use_latent_tokens=use_latent_tokens
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            orig_block_size=orig_bs,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation",
            tokenizer=tokenizer,  #added
            latent_size=latent_size,  #added
            use_latent_tokens=use_latent_tokens #added
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup()

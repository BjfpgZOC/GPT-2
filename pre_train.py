import os
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import wandb

from gpt2 import GPT, GPTConfig
from data.hellaswag import iterate_examples, render_example

import tiktoken

# ------------------------------------ DATALOADER ------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        data_root = "data/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
        
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos : self.current_pos + (B * T) + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_pos += B * T * self.num_processes

        if self.current_pos + (B * T * self.num_processes) + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = self.B * self.T * self.process_rank
        
        return x, y

def load_tokens(filename):
    npt  = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype = torch.long)
    return ptt

# ------------------------------------ LEARNING RATE SCHEDULER ------------------------------------

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ------------------------------------ DISTRIBUTED TRAINING ------------------------------------

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA is necessary!"
    init_process_group(backend = "nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

torch.manual_seed(99)
if torch.cuda.is_available():
    torch.cuda.manual_seed(99)

# ------------------------------------ TRAINING PARAMETERS ------------------------------------

device_type = "cuda" if device.startswith("cuda") else "cpu"

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "Ensure total_batch_size is divisble by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(F"Calcaulted gradient accumulation steps: {grad_accum_steps}")

print("DDP Rank:", ddp_rank)

# ------------------------------------ INITIALIZE TRAINER ------------------------------------

train_loader = DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_processes = ddp_world_size, split = "train")
val_loader = DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_processes = ddp_world_size, split = "val")

torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig(vocab_size = 50304))
model.to(device)

use_compile = False
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])

raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device_type = device_type)

log_dir = "log"
os.makedirs(log_dir, exist_ok = True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass

# ------------------------------------ HELLASWAG EVALUATION ------------------------------------

def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# ------------------------------------ WANDB LOGGING ------------------------------------

use_wandb = True
if master_process and use_wandb:
    wandb_run = wandb.init(
        entity = "seyal99",
        project = "GPT-2",
        name = "gpt2-pretrain",
        config = {
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "total_batch_size": total_batch_size,
            "micro_batch_size": B,
            "seq_len": T,
            "grad_accum_steps": grad_accum_steps,
            "vocab_size": 50304,
            "weight_decay": 0.1,
        }
    )

    wandb_run.define_metric("trainer_step")
    wandb_run.define_metric("train_loss", step_metric = "trainer_step")
    wandb_run.define_metric("val_loss", step_metric = "trainer_step")
    wandb_run.define_metric("hellaSwag_acc", step_metric = "trainer_step")
    wandb_run.define_metric("lr", step_metric = "trainer_step")
    wandb_run.define_metric("grad_norm", step_metric = "trainer_step")
    wandb_run.define_metric("tokens_per_second", step_metric = "trainer_step")

# ------------------------------------ TRAINING LOOP ------------------------------------

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x = x.to(device)
                y = y.to(device)

                with torch.autocast(device_type = device_type, dtype = torch.bfloat16):
                    logits, loss = model(x, y)
                
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation Loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val loss: {val_loss_accum.item():.4f}\n")

            if use_wandb:
                wandb_run.log({
                    "trainer_step": step,
                    "val_loss": val_loss_accum.item()
                })

            if (step > 0 and step % 4000 == 0) or last_step:
                ckpt_path = os.path.join(log_dir, f"gpt2_{step:05d}.pt")
                ckpt = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,       
                }
                torch.save(ckpt, ckpt_path)

    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        
        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                with torch.autocast(device_type = device_type, dtype = torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        if ddp:
            num_total = torch.tensor(num_total, dtype = torch.long, device = device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype = torch.long, device = device)
            dist.all_reduce(num_total, op = dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op = dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag Accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella acc: {acc_norm:.4f}\n")

            if use_wandb:
                wandb_run.log({
                    "trainer_step": step,
                    "hellaSwag_acc": acc_norm
                })


    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        if ddp: 
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type = device_type, dtype = torch.bfloat16):
            logits, loss = model(x, y)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)    

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_second = tokens_processed / dt
    if master_process:
        print(f"Step {step:4d}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f} ms, tok/sec: {tokens_per_second:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train loss: {loss_accum.item():.4f}\n")

        if use_wandb:
            wandb_run.log({
                "trainer_step": step,
                "train_loss": loss_accum.item()
            })

            wandb_run.log({
                "trainer_step": step,
                "lr": lr,
                "grad_norm": norm,
                "tokens_per_second": tokens_per_second,
            })
          
if master_process and use_wandb:
    wandb_run.finish()

if ddp:
    destroy_process_group()

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import time
from model import SimpleTransformer
from dataset import get_dataloader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def setup_ddp():
    """Initialize DDP"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        rank = 0
        world_size = 1
        local_rank = 0

    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def train_step(model, batch, optimizer, device):
    """Single training step"""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    
    # Extract logits from the output (HuggingFace models return a tuple or CausalLMOutput)
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs[0]  # First element is usually logits
    
    # Calculate loss (shift logits and labels for language modeling)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--memory_profile", action="store_true", help="Enable memory profiling")
    parser.add_argument("--memory_profile_steps", type=int, default=3, help="Number of steps to profile (for memory profiling)")
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Memory profiler
    if args.memory_profile and rank == 0:
        torch.cuda.empty_cache()
        torch.cuda.memory._record_memory_history(
            enabled='all',
            context='all',
            stacks='all',
        )
        print("Memory profiling enabled with full context and stack recording")
    
    if rank == 0:
        print(f"Training on {world_size} GPUs")
        print(f"Device: {device}")
    
    # Profiler setup
    if args.profile and rank == 0:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    
    # Model setup
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_config(config).to(device)
    # model = SimpleTransformer(
    #     vocab_size=100256,
    #     d_model=512,
    #     nhead=8,
    #     num_layers=6,
    #     max_len=args.max_length
    # ).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    # model = FSDP(model)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # DataLoader
    dataloader = get_dataloader(
        batch_size=args.batch_size,
        max_length=args.max_length,
        tokenizer_name="Qwen/Qwen3-0.6B"
    )
    
    # Training loop
    model.train()
    total_steps = 0
    
    for epoch in range(args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(dataloader):
            with record_function("train_step"):
                loss = train_step(model, batch, optimizer, device)
            
            total_steps += 1
            
            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
            
            # Profiler step
            if args.profile and rank == 0:
                profiler.step()
            
            # Memory snapshot
            if args.memory_profile and rank == 0 and step == args.memory_profile_steps:
                try:
                    torch.cuda.memory._dump_snapshot(f"memory_snapshot_step_{step}.pickle")
                except Exception as e:
                    print(f"Failed to capture memory snapshot {e}")
                torch.cuda.memory._record_memory_history(enabled=False)
            
            # Limit steps for demo
            if step >= 100:
                break
    
    # Cleanup profiler
    if args.profile and rank == 0:
        profiler.stop()
        print("Profiler stopped. Check ./profiler_logs for results")
    
    if args.memory_profile and rank == 0:
        torch.cuda.memory._record_memory_history(enabled=False)
        print("Memory profiling stopped. Check memory_snapshot_*.pickle files")
    
    # Save model
    if rank == 0:
        torch.save(model.module.state_dict(), "model_checkpoint.pt")
        print("Model saved to model_checkpoint.pt")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()

import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import time
import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
# from torch.nn.attention.flex_attention import BlockMask, flex_attention
# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min


from model import GPT, Muon, next_multiple_of_n


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


class DistributedDataLoader:
    def __init__(self, filename_pattern: str, batch_size: int, rank: int, world_size: int):
        self.files = sorted(Path.cwd().glob(filename_pattern))
        assert batch_size % world_size == 0, "Batch size must be divisible by world size"

        self.batch_size = batch_size
        self.local_batch_size = batch_size // world_size
        self.rank = rank
        self.world_size = world_size

        self.file_iter = iter(self.files)  # Replace with itertools.cycle(self.files) for multi-epoch training
        self.tokens, self.pos = self._load_data_shard(next(self.file_iter)), 0

    @staticmethod
    def _load_data_shard(file: Path):
        header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)  # header is 256 int32
        assert header[0] == 20240520, "Magic number mismatch in the data .bin file"
        assert header[1] == 1, "Unsupported version"

        num_tokens = int(header[2])  # Number of tokens (claimed)
        with file.open("rb", buffering=0) as f:
            tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)  # Avoid pin_memory copy
            f.seek(256 * 4)
            nbytes = f.readinto(tokens.numpy())  # Avoid bytes->array copy
            assert nbytes == 2 * num_tokens, "Number of tokens read does not match header"
        return tokens

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos + self.batch_size + 1 >= len(self.tokens):
            self.tokens, self.pos = self._load_data_shard(next(self.file_iter)), 0

        buf = self.tokens[self.pos + self.rank * self.local_batch_size:][:self.local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)

        self.pos += self.batch_size
        return inputs, targets


# -----------------------------------------------------------------------------
# int main


# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 8  # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0)  # this process will do logging, checkpointing etc.


@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    val_tokens = 10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # optimization
    train_num_iterations = 1770  # number of iterations to run for the first training
    cooldown_frac = 0.4  # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    n_layers: int = 12
    dim: int = 768
    # evaluation and logging
    val_loss_every = 125  # every how many steps to evaluate val loss? 0 for only at the end
    val_batch_size: int = world_size * 4 * 64 * 1024
    train_batch_size: int = world_size * 48 * 1024
    # implementation
    train_seq_len = 48 * 1024  # FlexAttention sequence length
    val_seq_len = 4 * 64 * 1024  # FlexAttention sequence length for validation
    save_checkpoint = False
    # dist
    world_size: int = world_size
    rank: int = rank


args = Hyperparameters()

# begin logging
logfile = None
run_id = uuid.uuid4()
if master_process:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)


def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)


# begin by printing this file (the Python code)
print0(code)
print0("=" * 100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")


def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


print0(nvidia_smi())
print0("=" * 100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len), args=args).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
adam_params = [
    {"params": head_params, "lr": 0.22 / 768 ** 0.5},
    {"params": embed_params, "lr": 0.6},
    {"params": scalar_params, "lr": 0.04},
]

# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
# optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
# optimizers = [optimizer1, optimizer2]


# Layer-wise learning rates
hidden_matrix_param_groups = []
num_blocks = len(model.blocks)
base_lr = 0.05  # same as before

for i, block in enumerate(model.blocks):
    block_params = [p for n, p in block.named_parameters() if p.ndim >= 2 and "embed" not in n]
    scale = 1.2 ** (num_blocks - i - 1)  # deeper layers get smaller LR
    lr_i = base_lr * scale
    hidden_matrix_param_groups.append({"params": block_params, "lr": lr_i})


optimizer2 = Muon(hidden_matrix_param_groups, momentum=0.95, rank=rank, world_size=world_size)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]


# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.train_num_iterations  # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1


# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)


def get_window_size_blocks(step: int):
    x = step / args.train_num_iterations  # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)


model: nn.Module = torch.compile(model, dynamic=False)

num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print0(f"Model num trainable parameters: {num_trainable_parameters:.1f}M", console=True)

# for idx, block in enumerate(model.blocks):
#     num_trainable_parameters = sum(p.numel() for p in block.parameters() if p.requires_grad) / 1e6
#     print0(f"Block: {idx}, Num trainable parameters: {num_trainable_parameters:.1f}M", console=True)

########################################
#            Warmup kernels            #
########################################
print0("Warmup\n", console=True)

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers])  # save the initial state
for _ in tqdm(range(warmup_steps)):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    loss = model(inputs.to(torch.int32), targets, get_window_size_blocks(0))
    loss.backward()

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state


########################################
#    Stage 1 Training and validation   #
########################################

def val_loop(model, args, rank, world_size, step):
    """
    Runs a validation pass and returns the average val_loss across all processes.
    """
    model.eval()
    assert args.val_tokens % args.val_batch_size == 0
    val_steps = args.val_tokens // args.val_batch_size

    val_loader = DistributedDataLoader(
        filename_pattern=args.val_files,
        batch_size=args.val_batch_size,
        rank=rank,
        world_size=world_size
    )

    total_val_loss = 0.0
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets = next(val_loader)
            loss = model(inputs, targets, get_window_size_blocks(step))
            total_val_loss += loss

    avg_val_loss = total_val_loss / val_steps
    dist.all_reduce(avg_val_loss, op=dist.ReduceOp.AVG)

    return avg_val_loss


def train_loop(model, train_loader, optimizers, optimizer2, args,
               rank, world_size, run_id, master_process, train_steps):
    # Timing
    training_time_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_loss = 0.0

    model.train()

    for step in range(train_steps + 1):
        last_step = (step == train_steps)

        # --------------- VALIDATION SECTION -----------------
        do_validation = (
                last_step or
                (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        )
        if do_validation:
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)

            val_loss = val_loop(model, args, rank, world_size, step)
            print0(
                f"step:{step}/{train_steps}, train loss:{train_loss:.4f}, val_loss:{val_loss:.4f} "
                f"train_time:{training_time_ms:.0f}ms "
                f"step_avg:{training_time_ms / max(step, 1):.2f}ms",
                console=True
            )

            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            # Save a checkpoint if needed
            if master_process and args.save_checkpoint:
                log = dict(
                    step=step,
                    model=model.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers]
                )
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)

        train_loss = model(inputs, targets, get_window_size_blocks(step))
        train_loss.backward()

        # Average gradients across distributed processes
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        # LR scheduling
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)

        # Momentum warmup scheduling for `optimizer2`
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1.0)
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        # Optimizer step
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        # Logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(
            f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms "
            f"step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
            console=True
        )


# Prepare training data loader
train_loader = DistributedDataLoader(
    filename_pattern=args.train_files,
    batch_size=args.train_batch_size,
    rank=rank,
    world_size=world_size
)

# Train model
train_loop(
    model=model,
    train_loader=train_loader,
    optimizers=optimizers,
    optimizer2=optimizer2,
    args=args,
    rank=rank,
    world_size=world_size,
    run_id=run_id,
    master_process=master_process,
    train_steps=args.train_num_iterations
)

print0(
    f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB \n"
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB\n"
    , console=True,
)

dist.destroy_process_group()

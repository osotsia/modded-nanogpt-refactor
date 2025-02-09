import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention


# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)


@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)


@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[
    Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)


@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)


def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None


def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)


mm_op.register_autograd(backward, setup_context=setup_context)


# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz

    Muon is an optimization algorithm that extends standard SGD with momentum by applying an
    orthogonalization post-processing step, where each 2D parameter's update is replaced with
    the nearest orthogonal matrix. The orthogonalization is performed using Newton-Schulz iterations,
    which can be stably run in `bfloat16` on the GPU.

    Reference:
        https://kellerjordan.github.io/posts/muon/

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():  # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5)

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    if g is None:
                        # continue
                        g = torch.zeros_like(p)  # Force a zero grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()  # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i: base_i + self.world_size]
            update_prev()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    """
    A modified linear layer that optionally performs computation in FP8 precision.

    This class extends `torch.nn.Linear` with an optional FP8 computation mode, controlled by `use_fp8`.
    When FP8 is enabled, the forward pass utilizes a custom matrix multiplication operation (`nanogpt::mm`)
    that scales inputs, weights, and gradients before performing the computation in FP8 precision.

    Attributes:
        use_fp8 (bool): If True, enables FP8 computation during training.
        x_s (float): Scaling factor for input tensor when using FP8.
        w_s (float): Scaling factor for weights when using FP8.
        grad_s (float): Scaling factor for gradients when using FP8.

    Note:
        - The FP8 computation is only used during training.
        - The custom operation `nanogpt::mm` is used for FP8 matrix multiplication, which handles input scaling
          and precision conversion to maintain numerical stability.
    """
    def __init__(self, in_features: int, out_features: int, use_fp8: bool = False, x_s: float = 1.0, w_s: float = 1.0,
                 grad_s: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5)  # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std  # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads,
                                                                             self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)  # @KoszarskyB & @Grad62304977
        else:  # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask,
                           scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, multiplier: float | int = 4):
        super().__init__()
        hdim = int(multiplier * dim)
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        # self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(
            x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class ProductKeyRouter(nn.Module):
    """
    Routes each input token to a small subset of experts using product-key retrieval.
    Optionally applies dropout or batchnorm to the projected query.
    """
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.model_dim = args.dim
        self.query_dim = args.peer_query_dim
        self.n_experts = args.peer_n_experts
        self.k = args.peer_topk
        self.n_heads = args.peer_n_heads

        # Each side = sqrt(n_experts)
        self.side = int(args.peer_n_experts**0.5)
        assert self.side * self.side == args.peer_n_expert, "n_experts must be a perfect square."
        assert args.peer_query_dim % 2 == 0, "query_dim must be even for product-key routing."
        self.sub_dim = args.peer_query_dim // 2

        # Router config
        self.query_batchnorm = args.peer_query_batchnorm
        self.input_dropout = args.peer_input_dropout
        self.query_dropout = args.peer_query_dropout

        # Sub-keys: [n_heads, 2, side, sub_dim]
        self.keys = nn.Parameter(
            torch.randn(args.peer_n_heads, 2, self.side, self.sub_dim) * (1.0 / (self.sub_dim**0.5))
        )

        # Query projection (model_dim -> n_heads * query_dim)
        layers = [nn.Linear(args.dim, args.peer_n_heads * args.peer_query_dim)]
        if args.peer_query_batchnorm:
            # BN is tricky if you have variable padding in the same batch.
            layers.append(nn.BatchNorm1d(args.peer_n_heads * args.peer_query_dim))
        self.query_proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len, model_dim]
        Returns:
          indices: [batch_size, seq_len, n_heads, k]  (top expert IDs)
          scores:  [batch_size, seq_len, n_heads, k]  (unnormalized gating scores)
        """
        bsz, seq_len, _ = x.size()

        # Optional dropout on input
        if self.input_dropout > 0.0:
            x = F.dropout(x, p=self.input_dropout, training=self.training)

        # Query projection
        # queries: [bsz, seq_len, n_heads * query_dim]
        queries = self.query_proj(x)
        if self.query_batchnorm:
            # If using BatchNorm1d, it expects shape [N, C], so we flatten temporal dims.
            # The forward method in BN needs [batch, features].
            # We'll do an extra reshape: [bsz*seq_len, n_heads*query_dim], then reshape back.
            pass  # The above sequential should handle it, but be mindful of shape if you separate the BN layer.

        # Reshape to [bsz, seq_len, n_heads, query_dim]
        queries = queries.view(bsz, seq_len, self.n_heads, self.query_dim)

        # Optional dropout on queries
        if self.query_dropout > 0.0:
            queries = F.dropout(queries, p=self.query_dropout, training=self.training)

        # Split each query into two sub-queries
        q1, q2 = queries.split(self.sub_dim, dim=-1)  # each: [bsz, seq_len, n_heads, sub_dim]

        # Flatten for matmul => shape [bsz*seq_len*n_heads, sub_dim]
        q1_flat = q1.view(-1, self.sub_dim)
        q2_flat = q2.view(-1, self.sub_dim)

        # keys: [n_heads, 2, side, sub_dim]
        # We'll do a separate top-k for each head, so we chunk the keys along dim=0 and loop or do a batched approach:
        # For simpler code: gather them all at once, then index by head.
        # shape after gather: [head, subkeys, side, sub_dim]

        # We'll define a small helper to do the top-k for one head's subkeys:
        def product_key_topk(q1_, q2_, head_idx):
            # subkeys shape: [2, side, sub_dim]
            subkeys = self.keys[head_idx]
            # Scores for subkey1: (q1_ -> [batch', sub_dim]) x [side, sub_dim]^T -> [batch', side]
            scores1 = torch.matmul(q1_, subkeys[0].transpose(0, 1))  # [batch', side]
            scores2 = torch.matmul(q2_, subkeys[1].transpose(0, 1))  # [batch', side]

            # top-k from each side => [batch', k], [batch', k]
            topk1, idx1 = torch.topk(scores1, self.k, dim=-1)
            topk2, idx2 = torch.topk(scores2, self.k, dim=-1)

            # Combine => k^2 candidates
            # topk1_expanded: [batch', k, 1], topk2_expanded: [batch', 1, k]
            topk1_expanded = topk1.unsqueeze(2)
            topk2_expanded = topk2.unsqueeze(1)
            combined = topk1_expanded + topk2_expanded  # [batch', k, k]
            combined_flat = combined.view(-1, self.k * self.k)

            # final topk => [batch', k]
            final_vals, final_idx = torch.topk(combined_flat, self.k, dim=-1)

            # get actual side indices
            subidx1 = final_idx // self.k
            subidx2 = final_idx % self.k
            actual_idx1 = idx1.gather(1, subidx1)
            actual_idx2 = idx2.gather(1, subidx2)

            # unify into single expert ID => id = idx1 * side + idx2
            expert_id = actual_idx1 * self.side + actual_idx2  # [batch', k]

            return final_vals, expert_id

        # We'll loop over heads. Another approach is to chunk q1_flat, q2_flat by head range.
        # shape of q1_flat: [bsz*seq_len*n_heads, sub_dim]
        # total 'batch'' = bsz*seq_len per head, so we just reshape.
        chunk_size = bsz * seq_len
        all_scores = []
        all_indices = []
        for h in range(self.n_heads):
            start = h * chunk_size
            end = (h + 1) * chunk_size
            vals_h, idx_h = product_key_topk(q1_flat[start:end], q2_flat[start:end], h)
            all_scores.append(vals_h)
            all_indices.append(idx_h)

        # Merge results => shape [bsz*seq_len, n_heads, k]
        scores_merged = torch.stack(all_scores, dim=1)
        idx_merged = torch.stack(all_indices, dim=1)

        # Reshape to final: [bsz, seq_len, n_heads, k]
        scores_merged = scores_merged.view(bsz, seq_len, self.n_heads, self.k)
        idx_merged = idx_merged.view(bsz, seq_len, self.n_heads, self.k)

        return idx_merged, scores_merged


class PEERLayer(nn.Module):
    """
    A PEER layer storing a massive set of single-neuron experts.
    Each expert: e_i(x) = activation(u_i^T x) * v_i.
    """
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.model_dim = args.dim
        self.n_experts = args.peer_n_experts
        self.k = args.peer_topk
        self.n_heads = args.peer_n_heads
        self.activation = args.peer_activation
        self.value_dropout = args.peer_value_dropout

        # Router
        self.router = ProductKeyRouter(args)

        # Each expert is (u_i, v_i). We'll store them in embeddings [n_experts, model_dim].
        self.u_emb = nn.Embedding(args.peer_n_experts, args.dim)
        self.v_emb = nn.Embedding(args.peer_n_experts, args.dim)

        # Optional final linear
        self.output_proj = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, model_dim]
        Returns:
          [batch_size, seq_len, model_dim]
        """
        bsz, seq_len, _ = x.size()

        # 1) Retrieve top-k experts via product-key router
        indices, scores = self.router(x)  # [bsz, seq_len, n_heads, k], same shape for scores

        # 2) Convert scores to gating probabilities. If you prefer a different gating (e.g. sigmoid),
        #    you can adapt here. Softmax along the k dimension:
        gate = F.softmax(scores, dim=-1)

        # 3) Gather (u_i, v_i) for each selected expert.
        # shapes => [bsz, seq_len, n_heads, k, model_dim]
        u = self.u_emb(indices)
        v = self.v_emb(indices)

        # 4) Expert forward => dot = u_i^T x
        # We broadcast x to [bsz, seq_len, n_heads, 1, model_dim], then do an elementwise multiply + sum over model_dim.
        x_expanded = x.unsqueeze(2).unsqueeze(3)  # shape [bsz, seq_len, n_heads, 1, model_dim]
        dot = (x_expanded * u).sum(dim=-1)        # shape [bsz, seq_len, n_heads, k]
        dot_activated = self.activation(dot)

        # 5) Multiply by v_i => shape [bsz, seq_len, n_heads, k, model_dim]
        expert_output = dot_activated.unsqueeze(-1) * v

        # Optional dropout on the “value” side, similar to PKM's value_dropout
        if self.value_dropout > 0.0:
            expert_output = F.dropout(expert_output, p=self.value_dropout, training=self.training)

        # 6) Weighted sum => sum_{k} gate_{k} * expert_output_{k}
        combined = (expert_output * gate.unsqueeze(-1)).sum(dim=3)  # [bsz, seq_len, n_heads, model_dim]

        # 7) Sum over heads => [bsz, seq_len, model_dim]
        combined = combined.sum(dim=2)

        # 8) Optional final projection
        out = self.output_proj(combined)
        return out


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, args):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim) if layer_idx != 3 else PEERLayer(args)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x


# -----------------------------------------------------------------------------
# The main model

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, args):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i, args) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128), use_fp8=True, x_s=0.5,
                                    w_s=2 ** -9, grad_s=2 ** -19)
        self.lm_head.weight.detach().zero_()  # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers // 2))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor) -> Tensor:
        assert input_seq.ndim == 1, "input_seq must be 1D"

        value_embeddings = [embed(input_seq) for embed in self.value_embeds]
        # Pattern: 0,1,2 + None blocks + 0,1,2. Credit @YouJiacheng, improved on @leloykun's U-net structure
        value_embeddings = (
                [value_embeddings[0], value_embeddings[1], value_embeddings[2]]
                + [None] * (len(self.blocks) - 6)
                + [value_embeddings[0], value_embeddings[1], value_embeddings[2]]
        )
        assert len(value_embeddings) == len(self.blocks)

        # Create block masks
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [
            long_bm, short_bm, short_bm, short_bm, long_bm, short_bm,
            short_bm, long_bm, short_bm, short_bm, short_bm, long_bm
        ]
        assert len(block_masks) == len(self.blocks)

        # Initial embedding + normalization
        x0 = norm(self.embed(input_seq)[None])  # use of norm here by @Grad62304977
        x = x0

        # U-Net style skip connections: down -> up path
        skip_connections = []
        num_skip = len(self.skip_weights)

        # "Down" pass: gather skip connections
        for i in range(num_skip):
            x = self.blocks[i](x, value_embeddings[i], x0, block_masks[i])
            skip_connections.append(x)

        # "Up" pass: retrieve and apply skip connections
        for i in range(num_skip, len(self.blocks)):
            x = x + self.skip_weights[i - num_skip] * skip_connections.pop()
            x = self.blocks[i](x, value_embeddings[i], x0, block_masks[i])

        x = norm(x)
        logits = self.lm_head(x)

        # @Grad62304977 added tanh softcapping following Gemma 2 paper,
        # @KoszarskyB reduced it from 30 to 15,
        # @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)

        # Cross-entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        return loss

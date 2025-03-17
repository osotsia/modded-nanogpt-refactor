import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention


class TestModel(nn.Module):
    def __init__(self, num_heads, head_dim, dim, use_dconv=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_dconv = use_dconv
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))

        if use_dconv:
            self.dconv_q = nn.Conv1d(head_dim, head_dim, kernel_size=3, padding=1, groups=head_dim, bias=False).to(torch.bfloat16)
            self.dconv_k = nn.Conv1d(head_dim, head_dim, kernel_size=3, padding=1, groups=head_dim, bias=False).to(torch.bfloat16)
            self.dconv_v = nn.Conv1d(head_dim, head_dim, kernel_size=3, padding=1, groups=head_dim, bias=False).to(torch.bfloat16)

        self.attn_scale = head_dim ** -0.5

    def _apply_dconv(self, tensor: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        B, T, nH, dH = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1).reshape(B * nH, dH, T)
        tensor = conv(tensor)
        tensor = tensor.reshape(B, nH, dH, T).permute(0, 3, 1, 2)
        return tensor

    def forward(self, x):
        B, T, D = x.shape
        assert B == 1, "Must use batch size = 1 for FlexAttention"

        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)) \
            .view(B, T, 3 * self.num_heads, self.head_dim) \
            .chunk(3, dim=-2)

        if self.use_dconv:
            q = self._apply_dconv(q, self.dconv_q)
            k = self._apply_dconv(k, self.dconv_k)
            v = self._apply_dconv(v, self.dconv_v)

        out = flex_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            scale=self.attn_scale
        ).transpose(1, 2)

        return out

def test_forward(use_dconv):
    B, T, D = 1, 10, 32  # Batch size must be 1 per assertion
    num_heads = 4
    head_dim = D // num_heads
    dim = D

    model = TestModel(num_heads, head_dim, dim, use_dconv)
    x = torch.randn(B, T, D, dtype=torch.bfloat16)
    out = model(x)

    assert out.shape == (B, T, num_heads, head_dim), f"Unexpected output shape: {out.shape}"
    print(f"Test passed with dconv={use_dconv}: Output shape is {out.shape}")

# Run tests in both configurations
test_forward(use_dconv=False)
test_forward(use_dconv=True)

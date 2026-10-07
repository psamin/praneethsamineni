"""Policy families, scaled down for a browser.

  mlp        6 -> h -> h -> 2. One action per decision (the baseline).
  diffusion  Diffusion Policy (Chi et al., RSS 2023), state-based: an MLP
             denoiser predicts the noise on an H-step action chunk conditioned
             on the observation and the diffusion step. DDIM sampling with
             S steps at inference; the first E actions of each chunk are
             executed (receding horizon).
  act        ACT (Zhao et al., RSS 2023), scaled down: a small pre-LN
             transformer over [z, obs, H action queries] predicts an H-step
             chunk; a CVAE encoder provides z during training only (z = 0 at
             test, as in ACT); temporal ensembling at execution.

Every inference path here is mirrored by src/robot/tinyPolicy.ts. The random
numbers used by diffusion sampling come from Rng (mulberry32 + Box-Muller),
which is implemented identically in TypeScript, so the two agree exactly.
"""

import math

import numpy as np
import torch
from torch import nn

H = 8            # action chunk length
E = 4            # actions executed per diffusion chunk (receding horizon)
K = 50           # diffusion training steps
S = 10           # DDIM sampling steps
TEMB = 8         # diffusion-step embedding size
ACT_Z = 8        # ACT latent size
ACT_ENSEMBLE_M = 0.1  # ACT temporal-ensembling weight decay (w_i = exp(-m * i), i = 0 oldest)


# --------------------------------------------------------------------- RNG
M32 = 0xFFFFFFFF


def _imul(a, b):
    return (a * b) & M32


class Rng:
    """mulberry32 + Box-Muller. Bit-identical to the TypeScript version."""

    def __init__(self, seed):
        self.a = seed & M32

    def uniform(self):
        self.a = (self.a + 0x6D2B79F5) & M32
        a = self.a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296

    def normal(self):
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# --------------------------------------------------------------- diffusion
def cosine_alpha_bar(k_steps=K, s=0.008):
    """Improved-DDPM cosine schedule, betas clipped at 0.999."""
    f = lambda t: math.cos((t / k_steps + s) / (1 + s) * math.pi / 2) ** 2
    ab, prod = [], 1.0
    for k in range(k_steps):
        beta = min(1 - f(k + 1) / f(k), 0.999)
        prod *= 1 - beta
        ab.append(prod)
    return ab


ALPHA_BAR = cosine_alpha_bar()
DDIM_STEPS = [K - 1 - i * (K // S) for i in range(S)]  # 49, 44, ..., 4


def step_embedding(k):
    freqs = [math.exp(-math.log(1000.0) * i / (TEMB // 2)) for i in range(TEMB // 2)]
    return [math.sin(k * f) for f in freqs] + [math.cos(k * f) for f in freqs]


# ------------------------------------------------------------ torch models
def make_mlp(n_in, h, n_out=2):
    return nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_out))


class DiffusionDenoiser(nn.Module):
    def __init__(self, n_obs, h):
        super().__init__()
        self.net = make_mlp(2 * H + n_obs + TEMB, h, 2 * H)
        self.register_buffer("temb", torch.tensor([step_embedding(k) for k in range(K)], dtype=torch.float32))
        self.register_buffer("ab", torch.tensor(ALPHA_BAR, dtype=torch.float32))

    def forward(self, x_noisy, obs, k):
        return self.net(torch.cat([x_noisy, obs, self.temb[k]], dim=-1))

    def loss(self, obs, chunk):
        x0 = chunk.reshape(len(chunk), -1)
        k = torch.randint(0, K, (len(x0),), device=x0.device)
        eps = torch.randn_like(x0)
        ab = self.ab[k].unsqueeze(-1)
        xk = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return nn.functional.mse_loss(self(xk, obs, k), eps)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)
        self.heads = heads

    def forward(self, x):
        B, T, D = x.shape
        hs = D // self.heads
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.heads, hs).transpose(1, 2) for t in (q, k, v))
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(hs), dim=-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.ff2(torch.relu(self.ff1(self.ln2(x))))


class TinyACT(nn.Module):
    def __init__(self, n_obs, d, heads=2, layers=2):
        super().__init__()
        self.obs_proj = nn.Linear(n_obs, d)
        self.z_proj = nn.Linear(ACT_Z, d)
        self.pos = nn.Parameter(torch.randn(2 + H, d) * 0.02)  # [z, obs, q_0..q_H-1]
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        # CVAE encoder: training only
        self.enc = make_mlp(n_obs + 2 * H, 64, 2 * ACT_Z)

    def decode(self, obs, z):
        B = len(obs)
        tok = torch.cat([self.z_proj(z)[:, None], self.obs_proj(obs)[:, None],
                         torch.zeros(B, H, self.pos.shape[1], device=obs.device)], dim=1) + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(self.ln_f(tok[:, 2:]))  # (B, H, 2)

    def loss(self, obs, chunk, beta=10.0):
        mu, logvar = self.enc(torch.cat([obs, chunk.reshape(len(chunk), -1)], -1)).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = (self.decode(obs, z) - chunk).abs().mean()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return rec + beta * kl / (2 * H)


def build(kind, n_obs, size):
    if kind == "mlp":
        return make_mlp(n_obs, size)
    if kind == "diffusion":
        return DiffusionDenoiser(n_obs, size)
    if kind == "act":
        return TinyACT(n_obs, size)
    raise ValueError(kind)


def deploy_params(kind, model):
    """Parameters that ship to the browser (the ACT CVAE encoder does not)."""
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith("enc."))


# ---------------------------------------------------------- numpy inference
def _f64(t):
    return t.detach().cpu().numpy().astype(np.float32).astype(np.float64)


def numpy_weights(kind, model):
    """Float32-rounded weights as float64 arrays: exactly what the browser uses."""
    sd = {k: _f64(v) for k, v in model.state_dict().items()}
    if kind == "mlp":
        return [(sd[f"{i}.weight"], sd[f"{i}.bias"]) for i in (0, 2, 4)]
    if kind == "diffusion":
        return [(sd[f"net.{i}.weight"], sd[f"net.{i}.bias"]) for i in (0, 2, 4)]
    return {k: v for k, v in sd.items() if not k.startswith("enc.")}


def mlp_forward(layers, x):
    h = np.asarray(x, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        h = W @ h + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def diffusion_sample(layers, obs, rng):
    """DDIM (eta = 0) from Gaussian noise drawn with `rng`. Returns (H, 2)."""
    x = np.array([rng.normal() for _ in range(2 * H)])
    obs = np.asarray(obs, dtype=np.float64)
    for i, k in enumerate(DDIM_STEPS):
        ab = ALPHA_BAR[k]
        ab_prev = ALPHA_BAR[DDIM_STEPS[i + 1]] if i + 1 < S else 1.0
        eps = mlp_forward(layers, np.concatenate([x, obs, step_embedding(k)]))
        x0 = np.clip((x - math.sqrt(1 - ab) * eps) / math.sqrt(ab), -1.0, 1.0)
        eps = (x - math.sqrt(ab) * x0) / math.sqrt(1 - ab)
        x = math.sqrt(ab_prev) * x0 + math.sqrt(1 - ab_prev) * eps
    return x.reshape(H, 2)


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b


def act_forward(w, obs, layers=2, heads=2):
    """Decoder with z = 0. Returns (H, 2)."""
    obs = np.asarray(obs, dtype=np.float64)
    d = w["obs_proj.weight"].shape[0]
    tok = np.zeros((2 + H, d))
    tok[0] = w["z_proj.bias"]                     # z = 0
    tok[1] = w["obs_proj.weight"] @ obs + w["obs_proj.bias"]
    tok += w["pos"]
    hs = d // heads
    for li in range(layers):
        p = f"blocks.{li}."
        qkv = _ln(tok, w[p + "ln1.weight"], w[p + "ln1.bias"]) @ w[p + "qkv.weight"].T + w[p + "qkv.bias"]
        out = np.zeros_like(tok)
        for hd in range(heads):
            q = qkv[:, hd * hs:(hd + 1) * hs]
            k = qkv[:, d + hd * hs:d + (hd + 1) * hs]
            v = qkv[:, 2 * d + hd * hs:2 * d + (hd + 1) * hs]
            s = q @ k.T / math.sqrt(hs)
            s = np.exp(s - s.max(-1, keepdims=True))
            out[:, hd * hs:(hd + 1) * hs] = (s / s.sum(-1, keepdims=True)) @ v
        tok = tok + out @ w[p + "proj.weight"].T + w[p + "proj.bias"]
        f = np.maximum(_ln(tok, w[p + "ln2.weight"], w[p + "ln2.bias"]) @ w[p + "ff1.weight"].T + w[p + "ff1.bias"], 0)
        tok = tok + f @ w[p + "ff2.weight"].T + w[p + "ff2.bias"]
    return _ln(tok[2:], w["ln_f.weight"], w["ln_f.bias"]) @ w["head.weight"].T + w["head.bias"]


class Runner:
    """Stateful closed-loop controller around any policy. Mirrored in TS."""

    def __init__(self, kind, weights, feature_kind, seed=0):
        from environment import features
        self.kind, self.w, self.fk = kind, weights, feature_kind
        self.features = features
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = Rng(self.seed)
        self.queue = []       # diffusion: actions left in the current chunk
        self.history = []     # act: (chunk, step it was predicted at)
        self.t = 0

    def act(self, state):
        obs = self.features(state, self.fk)
        if self.kind == "mlp":
            a = mlp_forward(self.w, obs)
            return float(a[0]), float(a[1])
        if self.kind == "diffusion":
            if not self.queue:
                self.queue = [tuple(r) for r in diffusion_sample(self.w, obs, self.rng)[:E]]
            ax, ay = self.queue.pop(0)
            return float(ax), float(ay)
        # act: temporal ensembling over every chunk that covers this step
        self.history.append((act_forward(self.w, obs), self.t))
        self.history = [(c, t0) for c, t0 in self.history if self.t - t0 < H]
        num, den = np.zeros(2), 0.0
        for i, (c, t0) in enumerate(self.history):  # i = 0 is the oldest
            wgt = math.exp(-ACT_ENSEMBLE_M * i)
            num += wgt * c[self.t - t0]
            den += wgt
        self.t += 1
        return float(num[0] / den), float(num[1] / den)

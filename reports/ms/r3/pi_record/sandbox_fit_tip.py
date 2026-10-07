"""PI-side sandbox (not repository evidence): can the actor class resolve the stage-2 cusp at
RL-matched optimiser budgets?  Supervised MSE fit of the Beta-mean head to the closed-form tent
e*(d) = e0 * max(0, 1 - |d| / (2q)), with the actor's architecture (2 -> 64 -> 64, orthogonal init
gain sqrt(2), zero output layer, mu = sigmoid(z0), effort = 100 mu, input (1, d / B), B = 100 + 2q),
Adam (lr 3e-4, betas 0.9/0.999, eps 1e-8), global grad-norm clip 0.5, minibatch 256.
Schedule mirrors 20 minibatch steps per PPO update: `const` steps at 3e-4 then `decay` steps linear to 3e-5.
"""
import sys, json, math
import numpy as np


def orth(rng, n_out, n_in, gain):
    a = rng.standard_normal((max(n_out, n_in), min(n_out, n_in)))
    qm, r = np.linalg.qr(a)
    qm = qm * np.sign(np.diag(r))
    w = qm if n_out >= n_in else qm.T
    return gain * w[:n_out, :n_in]


def run(act, in_scale, sampler, q, seed, const_steps, decay_steps, checkpoints):
    rng = np.random.default_rng(seed)
    H = 64
    B = 100.0 + 2 * q
    e0 = 4.0 / (4 * (1 / 3500.0) * q)          # DW / (4 k q)
    W1 = orth(rng, H, 2, math.sqrt(2)); b1 = np.zeros(H)
    W2 = orth(rng, H, H, math.sqrt(2)); b2 = np.zeros(H)
    W3 = np.zeros((1, H)); b3 = np.zeros(1)
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    f = (np.tanh, lambda x: 1 - np.tanh(x) ** 2) if act == 'tanh' else (lambda x: np.maximum(x, 0), lambda x: (x > 0).astype(float))

    def target(d):
        return e0 * np.maximum(0.0, 1.0 - np.abs(d) / (2 * q))

    def draw(n):
        if sampler == 'bb':
            return rng.uniform(-B, B, n)
        # stratified: tail |d| >= 2q share 0.5 (q50) / 20/44 (q60), near |d| < 20 share 0.35, middle the rest
        lam_T = (B - 2 * q) / B; lam_P = 0.35; lam_M = 1 - lam_T - lam_P
        u = rng.random(n); mag = np.empty(n)
        k = u < lam_P; mag[k] = rng.uniform(0, 20, k.sum())
        k2 = (u >= lam_P) & (u < lam_P + lam_M); mag[k2] = rng.uniform(20, 2 * q, k2.sum())
        k3 = u >= lam_P + lam_M; mag[k3] = rng.uniform(2 * q, B, k3.sum())
        return mag * np.where(rng.random(n) < 0.5, -1.0, 1.0)

    def forward(d):
        x = np.stack([np.ones_like(d), d / B * in_scale], 1)
        a1 = x @ W1.T + b1; h1 = f[0](a1)
        a2 = h1 @ W2.T + b2; h2 = f[0](a2)
        z = (h2 @ W3.T + b3)[:, 0]
        mu = 1 / (1 + np.exp(-z))
        return x, a1, h1, a2, h2, z, mu

    out = {}
    total = const_steps + decay_steps
    for step in range(1, total + 1):
        lr = 3e-4 if step <= const_steps else 3e-4 + (3e-5 - 3e-4) * (step - const_steps) / decay_steps
        d = draw(256)
        x, a1, h1, a2, h2, z, mu = forward(d)
        e = 100 * mu
        g_e = 2 * (e - target(d)) / d.size          # dL/de, L = mean (e - e*)^2
        g_z = g_e * 100 * mu * (1 - mu)
        gW3 = g_z[None, :] @ h2; gb3 = np.array([g_z.sum()])
        g_h2 = g_z[:, None] * W3
        g_a2 = g_h2 * f[1](a2)
        gW2 = g_a2.T @ h1; gb2 = g_a2.sum(0)
        g_h1 = g_a2 @ W2
        g_a1 = g_h1 * f[1](a1)
        gW1 = g_a1.T @ x; gb1 = g_a1.sum(0)
        grads = [gW1, gb1, gW2, gb2, gW3, gb3]
        gn = math.sqrt(sum(float((g * g).sum()) for g in grads))
        if gn > 0.5:
            grads = [g * (0.5 / gn) for g in grads]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
            mh = m[i] / (1 - 0.9 ** step); vh = v[i] / (1 - 0.999 ** step)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        if step in checkpoints or step == total:
            grid = np.arange(-2 * q, 2 * q + 0.25, 0.5)
            eh = 100 * forward(grid)[6]
            et = target(grid)
            tip = float(100 * forward(np.zeros(1))[6][0])
            rmse = float(np.sqrt(np.mean((eh - et) ** 2)))
            out[step] = {"e_hat_0": tip, "deficit": e0 - tip, "rel": (tip - e0) / e0,
                         "rmse_pos": rmse, "max_abs_W1_d": float(np.abs(W1[:, 1]).max() * in_scale)}
    return e0, out


if __name__ == '__main__':
    act, in_scale, sampler, q, seed = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    const, decay = int(sys.argv[6]), int(sys.argv[7])
    cps = {16000, 32000, 48000}
    e0, res = run(act, in_scale, sampler, q, seed, const, decay, cps)
    print(json.dumps({"act": act, "in_scale": in_scale, "sampler": sampler, "q": q, "seed": seed, "e0": e0,
                      "res": {str(k): v for k, v in sorted(res.items())}}))

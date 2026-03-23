# Phase 01: Global Configuration & Styling

## Objective

Update shared config and plotting defaults that all subsequent figure phases depend on.

## Steps

1. Confirm `Q_VALUES = [35.0, 40.0, 55.0]` is already set in `config.py` (done in prior commit)
2. Add a global shading alpha constant (e.g., `SHADE_ALPHA = 0.15`) to `config.py` — lighter than current
3. Add a helper to format q values without `.0` suffix (e.g., `format_q(35.0)` → `"q = 35"`)
4. Update `WEIGHT_VARIANT_LABELS` to include `k` value: e.g., `$w_H=6.5,\; w_L=3.0,\; k=0.0004$`
5. Update `ABLATION_COLORS` keys and any label mappings to use new terminology:
   - `baseline` → `TEL-PPO`
   - `no_cheap_gate` → `No stability screening`
   - `no_exploitability` → `No exploitability verification`
6. Add convergence vertical line style constants (color, linestyle, linewidth)
7. Run `python -m paper.generator --dry-run` to verify no import errors

## Files to modify

- `paper/generator/config.py`

## Verification

- `python -m paper.generator --dry-run` exits cleanly
- New constants are importable from `config`

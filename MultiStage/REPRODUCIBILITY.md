# Reproduction environment

Run commands from the repository root. The numerical experiments were executed on Linux, CPU, with one computational thread per worker.

The recorded environment is Python 3.12.3, NumPy 2.5.0 and PyTorch 2.5.1+cu121. The CUDA-tagged PyTorch build was used for CPU calculations. Matplotlib 3.11.0 generates figures; pandas 3.0.3 is present for tabular analysis. Existing T2 version checks are retained. A different numerical stack may run the model but is not an exact reproduction of these trajectories.

The root requirements files describe the broader repository. For these experiments, use the versions above and the study-specific instructions rather than silently upgrading the numerical libraries. No GPU is required.

Set these before launching Python:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
```

Follow [T2 instructions](two_stage/README.md) and [T3 instructions](three_stage/README.md) for the actual manifests, tests, smoke runs, verifier replay and full runs. Smoke outputs demonstrate code paths, not scientific success rates.

Use a fresh directory beneath `scratch/` for every reproduction. Original compact results in dated experiment directories are read-only evidence. Keep failed and incomplete attempts visible rather than replacing them with a successful rerun.

The figures/tables bundled here can be read without running training. Where dense raw tables were omitted, their availability and regeneration requirements are documented in the study README. Do not assume that a compact archive supports every legacy report-generation command.

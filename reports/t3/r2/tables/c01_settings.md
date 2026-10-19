| item | value |
|---|---|
| HEAD at the launch (the commit that holds the LOCK record) | `2b63ffe469fd75457938c715c194e944cb735587` |
| lock commit (in the LOCK record) | `72bd773f784f40bcd190113acbeb2dd8450d5b9a` |
| lock JSON SHA-256 (LOCK record) | `4f02c7ae89f2578a7510e1f3934caeb4fc6085e851d8898c26fff6fe1487a741` |
| lock MD SHA-256 (LOCK record) | `14f8e2e7e14e4398bf8ca7a46e7a762d7b25581977944580e23e975d2e08a063` |
| code commit recorded by the launcher | `be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7` |
| `git diff --stat be2b7e9f HEAD -- agents envs utils run tools/ms tools/v2` at the launch | empty |
| `git status --porcelain -- . ':(exclude)results'` at the launch (the clean-tree definition) | empty |
| `git status --porcelain` at the launch (recorded by the launcher in `launch_*.json`) | 34 entries, all of them untracked files under `results/` (the audit's re-run weights and arrays, `a7_emat.npz`, and `results/t3_r2/confirmation/` itself) |
| workers (single-threaded) | 40 |
| `nproc` | 64 |
| load average at the start | 14.20, 13.88, 14.97 |
| free disk at the start | 879 GB of 5952 GB |
| parameters file SHA-256 | `db5156962a2d7a3eca0b7d0440d995d84dc57bb88c7066e38d7d61e40148e976` |
| planned / finished | 40 / 40 |
| final state of the launch | done |

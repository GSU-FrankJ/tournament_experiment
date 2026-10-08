# report_scripts

Scripts of the T=2 status pack. They read tracked records only; nothing is trained, evaluated on weights, launched or re-analysed. Run them from anywhere inside the repository with `/home/fjiang4/tournament_experiment/.venv/bin/python`.

| script | what it does |
|---|---|
| `build_pack.py` | Builds `evidence/` (copies of the cited tracked files, each read with `git show <last commit that touched it, at be4fd202>:<path>`), `figures/` (figure copies plus the two generated figures) and the manifest and `SHA256SUMS` files. Items of the 100526 pack are cited in place and verified against that pack's manifest. `--check` rebuilds into a temporary directory and compares every file byte for byte with the folder. The `used_in` column of the manifest is read from the citations in `report.md`, so rebuild after editing the report. |
| `tables.py` | Produces every table of `report.md` from the evidence copies (this pack and the 100526 pack). `--list` names the blocks and ids; `--block NAME` prints one; `--inject FILE` rewrites the `<!-- TBL:NAME -->` blocks of the report; `--verify FILE` exits 1 if a block differs from the script output. The few text tables (rounds, gates of the PI, goals, issues, candidate measures) are static text in the script with their citations. |
| `figures.py` | Draws F1 (`FIG-12`, |peak error| per run for every arm and the v2.0 confirmation) and F2 (`FIG-13`, the gap split into smoothing part and remainder) only from evidence copies. |
| `check_numbers.py` | Checks that the tables in `report.md` equal the script output, that no table is typed by hand outside a block, that every cited id exists in the manifest and that every numeric sentence outside the tables cites an evidence item (numbers that are not data, such as section numbers, seed names, dates and q values, are exempt by pattern). |
| `check_links.py` | Checks that every relative link and image of `report.md`, `README.md` and `handoff.md` resolves, in the working tree or, with `--ref`, in `git ls-tree -r <ref>` (then it also checks that every file of the folder is in that tree). |

Order for a rebuild: `build_pack.py` (evidence, figures, manifest) then `tables.py --inject ../report.md` then `build_pack.py` again if the report's citations changed, then the checks. The figures are drawn with matplotlib without a timestamp in the PNG metadata, so the rebuild is byte-identical.

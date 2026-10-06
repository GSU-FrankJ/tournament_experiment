# report_scripts: the scripts the section writers used while writing `100526report.md`

These are the generator and check scripts that the drafting agents ran against the evidence pack to build the tables of each section (`W-A` section 3, `W-B` sections 4.1-4.4, `W-C` sections 4.5-4.6, `W-D` section 5, `W-E` section 6, `W-F` sections 2 and 7) and to verify the PI-side readings (`pi_readings`). They are kept as a record of how the tables were computed; they are **not** tools of the repository:

- They are copied as they were run. They read the pack copies under `reports/t2_refine_100526/evidence/` (absolute worktree paths) and write to a scratch directory of the drafting session, so they do not run unchanged elsewhere.
- The prose of each section was edited after generation (patches, the assembler's replacements of 'needs item' markers by pack IDs, voice), so re-running a script does not reproduce a section file byte for byte; the tables are what the scripts produce.
- Exploratory and patch scripts of the drafting session are not included.
- `check_report_numbers.py` is the number-presence backstop given to the reviewers as a hint; `final_checks.py` is the editor's last mechanical check of the folder (links, item tags, table shapes, checksum files, forbidden files and sizes; the worktree path is hard-coded).
- Their correctness is not assumed: the numbers of the report were checked independently (`../pi_record/01_factcheck.md`).

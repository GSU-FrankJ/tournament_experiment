# T=2 status pack: record the PI's reply (docs only, no experiments)

Context. A previous session built the T=2 status pack for a coworker's decision, following the prompt saved at reports/t2_status_100826/pi_record/21_t2_status_pack_prompt.md. It pushed branch t2-status-pack (head e4f93dc8d5a6f0a848cb1c3bbbd17a115fd7d47e; the report permalink is at bbbbf7fe) and stopped at its step P4, waiting for the PI's answers to its section D7. This message is that answer. The only task of this session is to record it.

Repository /home/fjiang4/tournament_experiment. Read docs/STATE.md first, as usual.

## The PI's reply

1. The pack at e4f93dc8 is accepted as delivered, together with the differences from its prompt that the previous session reported:
   - the sign statement corrected to the record (eight runs with a non-negative signed peak error; report sections 7.3 and 11);
   - the trajectory drift as recorded, +1.83, not "about 1.7" (section 7.5);
   - the smoothing-floor range at s = 1 of 2.2-2.7 % once the t10 arms are included (section 10, reference points);
   - the fix of the empty table blocks in report.md (ledger rows S2-1 and S3-53);
   - the extra manifest prefixes TBL-, BG- and T2R:100526report, and the project instruction file stored as evidence/dot-claude/CLAUDE.md.txt;
   - the condensed fact-check ledger (non-OK rows listed with dispositions, OK rows counted per slice);
   - the English handoff text of 209 words.
2. Handoff (the answers to D7). The PI sends the text of handoff.md personally, outside the repository, to the coworker the PI has chosen. This session sends nothing to anyone: no email, message, issue, pull request, mention or comment. No name or contact detail of the recipient goes into the repository.
3. Nothing else in the pack changes. report.md, handoff.md, README.md, evidence/, figures/, report_scripts/ and pi_record/01_factcheck.md stay as they are.

## Task

1. Verify the starting point. git ls-remote origin t2-status-pack must equal e4f93dc8d5a6f0a848cb1c3bbbd17a115fd7d47e, and reports/t2_status_100826/ must exist at that commit. If either fails, stop and report.
2. Run git fetch origin t2-status-pack. Then create your own worktree from origin/t2-status-pack on a new local branch, e.g. t2-status-pack-reply. Do not use or modify the previous session's worktree (.claude/worktrees/t2-status-pack-0c452c).
3. Save this message verbatim as reports/t2_status_100826/pi_record/22_pi_reply.md.
4. Append a dated section "P5: the PI's reply (2026-10-08)" to reports/t2_status_100826/pi_record/00_build_log.md. It records:
   - the pack accepted at e4f93dc8;
   - the accepted differences, as listed above;
   - the handoff sent by the PI personally, outside the repository, and nothing sent by this session;
   - the next step: the coworker's decision.
5. In the T=2 status-pack section at the top of docs/STATE.md, add one line: the PI's reply is recorded in pi_record/22_pi_reply.md, the handoff is sent by the PI, and the work is waiting for the coworker's decision.
6. Make one commit: "docs: record the PI reply to the T=2 status pack". Check that git diff --stat e4f93dc8 HEAD lists exactly these three files.
7. Push as a fast-forward: git push origin HEAD:t2-status-pack, never with force. If it is not a fast-forward, stop and report. Then check that git ls-remote origin t2-status-pack equals the new local HEAD.
8. Report the new head commit, the diff stat, and that nothing was run or sent. Then stop.

Out of scope: any experiment, training run, analysis or rebuild of the pack; any change to main, the round branches or the tags; any pull request.

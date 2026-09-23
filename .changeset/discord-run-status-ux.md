---
"@herdctl/discord": patch
---

Run status UX. The progress run card is now a live status only: its header shows the elapsed time and the last five tools, the trace lists tool names without arguments or output previews (these can carry paths, commands or secrets), and the card is deleted when the run ends instead of being finalized as "Task complete" above the answer. When `acknowledge_emoji` is set, the acknowledgement reaction is swapped for ✅ or ❌ at the end of the run, on the triggering message and on every message injected into that run.

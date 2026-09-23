---
"@herdctl/discord": patch
---

One live session per Discord channel (vulpes-pack, 2026-09-23 Swift double-job). Discord jobs now run session-backed (`interactive: true`). A message arriving while a job runs for the same channel (or thread) is pushed into that job via `FleetManager.sendToJob` as a new user turn instead of starting a second, parallel job in a different session; its answer is relayed per turn by the running job. If injection is refused (job not session-backed yet, winding down, or a voice/attachment message), the message waits for the running job to finish and then resumes the channel session as before. Different channels still run in parallel.

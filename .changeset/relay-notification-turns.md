---
"@herdctl/discord": patch
---

Relay assistant turns produced after background-task notifications (vulpes-pack#649). A turn triggered by a background subagent's completion (not a new Discord message) reused the previous turn's delta-streamed message handle instead of creating its own, so its content got appended onto and then silently overwrote the prior turn's already-delivered message rather than being relayed as its own reply. The live-streamed-message state is now reset at each turn boundary (the `result` message), so every finalized turn — regardless of what triggered it — gets its own Discord message.

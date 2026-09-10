---
"@herdctl/discord": patch
---

Include the Discord message author (display name + user id) in the prompt sent to the agent (vulpes-pack#354).

`handleMessage()` built `Current user message: ${prompt}` without saying who sent it. The `[Name at <ts>]:` context lines are only prepended when there is no existing session, so a resumed channel session lost author attribution entirely — the agent could not tell who was speaking, even though `event.metadata.userId`/`username` were already available.

The prompt now always opens with `Current user message from ${username} (<@${userId}>): ...`, for both fresh and resumed sessions.

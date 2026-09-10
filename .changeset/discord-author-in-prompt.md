---
"@herdctl/discord": patch
---

Include the Discord message author (username and user id) in the prompt sent to the agent. Previously this was dropped for resumed sessions and voice messages, so the agent could not always tell who was speaking.

---
"@herdctl/discord": patch
---

Include the current message's author in the assembled prompt when prior channel context is prepended.

When a fresh session (no resumed session id) has prior channel messages, the
manager prepends that history to the prompt. Each history line carries its
author (`[<authorName> at <ts>]: ...`), but the current message was appended as
a bare `Current user message: <prompt>` with no author at all. The agent then
had to guess who was currently speaking from the older lines alone, and could
attribute the message to whoever spoke last in the history instead of the
actual sender.

The current-message line now reads `Current user message (from
<username>): <prompt>`, using `event.metadata.username` — already available on
every message event. The context lines use `displayName ?? username`; wiring
`displayName` through the connector's metadata as well would have required a
new required field on `DiscordConnectorEventMap["message"]["metadata"]`,
touching dozens of existing test literals across `manager.test.ts` for a
one-line prompt fix, so this patch uses `username` for the current-message
label instead. `username` and `displayName` are usually the same value for
accounts without a set global display name.

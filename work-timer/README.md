# Work Timer

A single-file focus timer: 40 minutes of work, 10 minutes of break, repeating.
Open `index.html` in a browser. Nothing to install, everything is stored in the
browser's localStorage.

The side panel holds a task list, today's focused minutes and completed cycles,
and a session log. Click a task to make it the active one; minutes from each
work block are credited to whichever task was active, and switching tasks
mid-block splits the block between them. A sound and a desktop notification
play at each phase change.

## Working with Claude

Out: "Copy log for Claude" puts a plain summary plus the raw JSON on the
clipboard, ready to paste into a chat. "Download JSON" saves the same data as a
file Claude can read.

In: Claude can hand tasks back in any of three ways.

1. Paste JSON into the Claude box and press Import (a ```json fenced block is fine).
2. Open a link of the form `index.html#import=<url-encoded JSON>`.
3. Save the JSON as `tasks.json` next to `index.html`, serve the folder
   (`python3 -m http.server`), and open `index.html?load=tasks.json`.
   This replaces the current task list.

Accepted format (see `tasks.example.json`):

```json
{
  "tasks": [
    { "name": "Read chapter 3", "estimate": 40, "notes": "optional" },
    "A bare string also works"
  ],
  "activeTask": "Read chapter 3"
}
```

`estimate` is in minutes. A file previously exported by the timer can be
imported too; its sessions are merged into the log without duplicates.

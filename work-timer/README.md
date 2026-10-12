# Work Timer

A single-file focus timer that runs 40 minutes of work and 10 minutes of break
on repeat, with a task side panel, a session log, stats, and a two-way handoff
with Claude. Open `index.html` in a browser. Nothing to install. Data stays in
the browser's localStorage.

Ideas borrowed from other open-source timers (Pomofocus-style task estimates,
Super Productivity's history, terminal timers' keyboard controls) and kept to
one file.

## Using it

- Space starts and pauses, S skips to the next phase, R restarts the phase.
  Press ? for the full list.
- Add tasks in the side panel. The `1×` stepper sets an estimate in work blocks.
  Click a task to focus it, double-click to rename, drag to reorder. Minutes from
  each block are credited to the focused task, and switching mid-block splits
  the block between tasks.
- After each work block a short prompt asks what you got done. The note lands in
  the log and in what Claude sees.
- The Stats tab shows today, the last 7 days, your streak, and today's time per task.
- Gear icon: durations (default 40/10), auto-start, sound, notifications, and the
  focus/blocking hook below. F hides the side panel. The moon icon cycles the theme.

## Working with Claude

Claude tab:

- "Copy log for Claude" copies a readable summary of today, the last 7 days or
  everything, plus the raw JSON.
- "Copy a planning prompt" copies a prompt that asks Claude to break a goal into
  40-minute tasks and reply with importable JSON.
- Paste Claude's reply into the import box. Surrounding text is fine; the last
  ```json block is used.

Other ways in: drop a `.json` file onto the page, open
`index.html#import=<url-encoded JSON>`, or serve the folder
(`python3 -m http.server`) and open `index.html?load=tasks.json`, which replaces
the task list.

Format (see `tasks.example.json`). `estimate` is in minutes:

```json
{
  "tasks": [
    { "name": "Read chapter 3", "estimate": 40, "notes": "optional" },
    "A bare string also works"
  ],
  "activeTask": "Read chapter 3"
}
```

## Focus mode and site blocking

A web page can't control Apple Screen Time or Windows Family Safety directly;
those only accept native apps. Instead, the timer can trigger something on your
computer when a work block starts and when a break starts. Set it in Settings →
Focus & site blocking, using the Mac / iPhone or Windows button to fill in the
defaults. The browser will ask the first time whether to allow opening Shortcuts
or `worktimer`. Tick "always allow" so it can fire on its own when the timer
rolls over.

The block list for both platforms is `blocking/blocklist.txt`. Blocking works
through the hosts file, so already-open tabs may keep working until reloaded,
and a browser using its own secure DNS may need a minute to notice.

### Mac (and iPhone / iPad for Focus)

1. In the Shortcuts app, create a shortcut named `Work Timer Focus` with the
   action "Set Focus" → Work (or Do Not Disturb) → On.
2. Create `Work Timer Break` with "Set Focus" → Off.
3. Click "Mac / iPhone" in the timer's settings, then "Test work start".

On iPhone/iPad that's as far as it goes: Focus can silence apps and hide home
screen pages, but Shortcuts can't toggle Screen Time website limits.

To also block sites on a Mac, install the script as root-owned (so passwordless
sudo can't be abused by editing it), then allow it without a password:

```sh
sudo install -o root -g wheel -m 755 blocking/mac/block-sites.sh /usr/local/bin/work-timer-block
sudo install -d /usr/local/etc
sudo install -o root -g wheel -m 644 blocking/blocklist.txt /usr/local/etc/work-timer-blocklist.txt
echo "$(whoami) ALL=(root) NOPASSWD: /usr/local/bin/work-timer-block" | sudo tee /etc/sudoers.d/work-timer
sudo chmod 440 /etc/sudoers.d/work-timer
```

Then add a "Run Shell Script" action to each shortcut:
`sudo /usr/local/bin/work-timer-block on` in `Work Timer Focus` and
`sudo /usr/local/bin/work-timer-block off` in `Work Timer Break`. After editing
the block list, re-run the second `install` line.

### Windows

1. Open PowerShell as administrator in `blocking\windows` and run
   `powershell -ExecutionPolicy Bypass -File install.ps1`.
   This creates two scheduled tasks that run `block-sites.ps1` with admin rights
   and registers `worktimer://block` and `worktimer://unblock` links that trigger
   them. No UAC prompt each time. Your Windows account needs to be an administrator.
2. Click "Windows" in the timer's settings, then "Test work start", and check
   that a blocked site no longer loads.
3. Edit `blocking\blocklist.txt` any time; the next block picks it up. Keep the
   folder where it was when you installed.

To remove it: `powershell -ExecutionPolicy Bypass -File uninstall.ps1` (as admin).
To unblock by hand: run `block-sites.ps1 off` from an admin PowerShell.

Windows Focus Assist / Do Not Disturb has no supported command line switch, so
the Windows hook only does site blocking.

#!/bin/sh
# Block or unblock distracting sites on macOS by editing /etc/hosts.
# Usage: sudo block-sites.sh on|off|status
# Sites come from $WORK_TIMER_BLOCKLIST, else /usr/local/etc/work-timer-blocklist.txt
# (where the README's install step puts it), else ../blocklist.txt.
set -eu

HOSTS=/etc/hosts
BEGIN="# >>> work-timer block >>>"
END="# <<< work-timer block <<<"
LIST=${WORK_TIMER_BLOCKLIST:-/usr/local/etc/work-timer-blocklist.txt}
[ -f "$LIST" ] || LIST="$(cd "$(dirname "$0")/.." && pwd)/blocklist.txt"

strip() {
  tmp=$(mktemp)
  sed "/^$BEGIN\$/,/^$END\$/d" "$HOSTS" > "$tmp"
  cat "$tmp" > "$HOSTS"
  rm -f "$tmp"
}

flush() {
  dscacheutil -flushcache 2>/dev/null || true
  killall -HUP mDNSResponder 2>/dev/null || true
}

case "${1:-}" in
  on)
    [ -f "$LIST" ] || { echo "No blocklist at $LIST" >&2; exit 1; }
    strip
    {
      echo "$BEGIN"
      grep -v '^[[:space:]]*\(#\|$\)' "$LIST" | while read -r d; do
        d=${d#www.}
        printf '0.0.0.0 %s\n0.0.0.0 www.%s\n::1 %s\n::1 www.%s\n' "$d" "$d" "$d" "$d"
      done
      echo "$END"
    } >> "$HOSTS"
    flush
    echo "Blocked sites from $LIST"
    ;;
  off)
    strip
    flush
    echo "Unblocked"
    ;;
  status)
    if grep -q "^$BEGIN\$" "$HOSTS"; then echo "on"; else echo "off"; fi
    ;;
  *)
    echo "Usage: sudo $0 on|off|status" >&2
    exit 2
    ;;
esac

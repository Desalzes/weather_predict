#!/usr/bin/env bash
# PreToolUse hook. Blocks Edit/Write/MultiEdit/NotebookEdit/Bash
# until .claude/markers/oriented-<session_id> exists.
#
# Protocol: exit 0 = allow, exit 2 = block and surface stderr to
# Claude. Input is JSON on stdin with at least .tool_name and
# .session_id.
set -euo pipefail

INPUT="$(cat)"

# jq is assumed present. If it isn't, fail open with a warning —
# a broken hook bricking every session is worse than skipping the
# check. Log the warning so the user notices.
if ! command -v jq >/dev/null 2>&1; then
  echo "[pre-tool-use] WARNING: jq not installed; orientation check skipped" >&2
  exit 0
fi

TOOL_NAME="$(echo "$INPUT" | jq -r '.tool_name // empty')"
SESSION_ID="$(echo "$INPUT" | jq -r '.session_id // "default"')"

# Sanitize session id for filesystem
SESSION_ID="${SESSION_ID//[^A-Za-z0-9_-]/_}"

# Subagent escape hatch. Subagent tool calls are exempt — the
# parent session has already oriented, and forcing subagents to
# re-orient defeats the point. Claude Code surfaces a
# .tool_input.subagent_type or similar flag in some versions; we
# look for any "is_subagent" / "subagent" signal and bail if set.
SUBAGENT="$(echo "$INPUT" | jq -r '.tool_input.subagent_type // .is_subagent // empty')"
if [[ -n "$SUBAGENT" ]]; then
  exit 0
fi

# Only gate mutation tools. Reads (Read, Glob, Grep) and MCP
# tools pass through so `orient` itself can run.
case "$TOOL_NAME" in
  Edit|Write|MultiEdit|NotebookEdit|Bash) : ;;
  *) exit 0 ;;
esac

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$PWD}"
MARKER_DIR="$PROJECT_DIR/.claude/markers"
MARKER="$MARKER_DIR/oriented-$SESSION_ID"

if [[ -f "$MARKER" ]]; then
  exit 0
fi

# Block. stderr is surfaced to Claude as the block reason.
cat <<EOF >&2
BLOCKED: session not oriented.

Tool: $TOOL_NAME
Session: $SESSION_ID
Expected marker: $MARKER

Run the \`orient\` skill. It loads the four tiers
(ADRs, session hub, in-flight state, git) and writes the
marker that unblocks this tool.

If this is trivial Q&A that truly will not touch code, the
user can grant a skip:
  mkdir -p "$MARKER_DIR"
  echo "skipped: <reason>" > "$MARKER"

Do not work around this hook. It is the whole point of the
project setup.
EOF

exit 2

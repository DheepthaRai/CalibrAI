#!/usr/bin/env bash
# ─── Dino Multi-Instance Launcher ────────────────────────────────────────────
# Starts five Dino bot instances in parallel, one per safety level (L1–L5).
# Each instance loads its own .env.L* file.
# Logs are written to dino-L{1..5}.log in this directory.
#
# Usage:
#   chmod +x start_all.sh
#   ./start_all.sh
#
# To stop all instances:
#   pkill -f "bun run bot.js"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

start_instance() {
  local level="$1"
  local env_file="${SCRIPT_DIR}/.env.L${level}"
  local log_file="${SCRIPT_DIR}/dino-L${level}.log"

  if [[ ! -f "$env_file" ]]; then
    echo "❌ Missing env file: $env_file" >&2
    return 1
  fi

  echo "🦕 Starting Dino L${level} (env: .env.L${level}, log: dino-L${level}.log)..."

  # Load the env file and launch bun in the background
  (
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
    cd "$SCRIPT_DIR"
    bun run bot.js >> "$log_file" 2>&1
  ) &

  echo "   PID $! → L${level}"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Dino Multi-Instance Launcher"
echo " Safety levels: L1 (Very Permissive) → L5 (Very Strict)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for level in 1 2 3 4 5; do
  start_instance "$level"
done

echo ""
echo "✅ All five instances launched. To monitor:"
echo "   tail -f dino-L3.log          # follow a single instance"
echo "   tail -f dino-L{1..5}.log     # follow all (bash only)"
echo ""
echo "To stop all instances:"
echo "   pkill -f 'bun run bot.js'"

# Keep the script alive so the shell session holds all child processes
wait

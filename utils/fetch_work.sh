#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG (edit if needed) ---
JUMP="dkotowsk@sshjump.uni-muenster.de"
DEST="dkotowsk@myri"
# -------------------------------

LOCAL_WORK="${WORK:-$HOME/palma_work}"
mkdir -p "$LOCAL_WORK"

echo "==> Fetching remote ~/files  -->  local $LOCAL_WORK"
echo "    via jump host: $JUMP"

# add -n to DRY RUN: ./fetch_remote_files.sh -n
EXTRA_FLAGS="${1:-}"
rsync -avh $EXTRA_FLAGS \
  -e "ssh -J ${JUMP}" \
  "${DEST}:~/files/" "$LOCAL_WORK/"

echo "==> Done. Files are in: $LOCAL_WORK"

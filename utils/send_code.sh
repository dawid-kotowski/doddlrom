set -euo pipefail

# Remote definitions
JUMP="dkotowsk@sshjump.uni-muenster.de"
DEST="dkotowsk@myri"

# Local project dir = current dir where the script is located
LOCAL_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

echo "==> Sending repo: $LOCAL_ROOT  -->  ${DEST}:~/master-project-1"
echo "    via jump host: $JUMP"

EXTRA_FLAGS="${1:-}"
rsync -avh --delete $EXTRA_FLAGS \
  --exclude=".git" \
  --exclude="__pycache__" --exclude=".mypy_cache" --exclude=".pytest_cache" \
  --exclude=".DS_Store" \
  -e "ssh -J ${JUMP}" \
  "${LOCAL_ROOT}/" "${DEST}:~/master-project-1/"

echo "==> Done."

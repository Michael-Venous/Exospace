#!/usr/bin/env bash
# Run the optimized experiment from this source tree, including through a symlink.
set -euo pipefail
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
export EXO_OPTIMIZED=1
exec "$(dirname -- "$script_path")/talk_candidate.sh" "$@"

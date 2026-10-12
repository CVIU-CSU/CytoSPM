#!/usr/bin/env bash
# Download checkpoints from Hugging Face into checkpoints/ and verify SHA256.
#
#   bash tools/fetch_checkpoints.sh                         # WeDetect-Tiny init (training)
#   CKPT_FILES=cytospm_convnext_t.pth bash tools/fetch_checkpoints.sh   # CytoSPM weight
#
# Set HF_ENDPOINT=https://hf-mirror.com to download through the mirror.
set -euo pipefail
cd "$(dirname "$0")/.."

PYBIN="${PYBIN:-python}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
CKPT_FILES="${CKPT_FILES:-wedetect_tiny.pth}"

declare -A REPO=(
  [wedetect_tiny.pth]="fushh7/WeDetect"
  [cytospm_convnext_t.pth]="Doilion111/CytoSPM"
)
declare -A REVISION=(
  [wedetect_tiny.pth]="3b9d0670cbf300d0b48f900ae87a6c4785d88939"
  [cytospm_convnext_t.pth]="5533e885759371f62838577998aea8b68ae882ef"
)
declare -A SHA256=(
  [wedetect_tiny.pth]="e1c881120c3ea51b09242f16a7da66f62423f47adabe7ffd2814ad002d2ac62c"
  [cytospm_convnext_t.pth]="bc94215314319f712c731e8f5f24388df29d254cc2029695d32e815efb3aac4d"
)

log() { printf '[fetch] %s\n' "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

verify() {  # $1=path $2=filename
  local want="${SHA256[$2]}" got
  [ -s "$1" ] || die "missing or empty: $1"
  got="$(sha256sum "$1" | cut -d' ' -f1)"
  [ "$got" = "$want" ] || die "SHA256 mismatch for $2 (expected $want, got $got)"
  log "SHA256 OK: $2"
}

read -r -a files <<< "$CKPT_FILES"
[ "${#files[@]}" -gt 0 ] || die "CKPT_FILES is empty"
for f in "${files[@]}"; do
  [ -n "${SHA256[$f]:-}" ] || die "no SHA256 registered for '$f'"
done

mkdir -p "$CKPT_DIR"
for f in "${files[@]}"; do
  if [ -e "$CKPT_DIR/$f" ]; then
    verify "$CKPT_DIR/$f" "$f"
    continue
  fi
  tmp_dir="$(mktemp -d "$CKPT_DIR/.fetch.XXXXXX")"
  log "downloading ${REPO[$f]}@${REVISION[$f]} :: $f"
  "$PYBIN" - "${REPO[$f]}" "$f" "${REVISION[$f]}" "$tmp_dir" <<'PY'
import shutil
import sys

from huggingface_hub import hf_hub_download

repo, filename, revision, destination = sys.argv[1:]
src = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
shutil.copyfile(src, f'{destination}/{filename}')
PY
  verify "$tmp_dir/$f" "$f"
  mv -f "$tmp_dir/$f" "$CKPT_DIR/$f"
  rm -rf "$tmp_dir"
done
log "done -> $CKPT_DIR (${files[*]})"

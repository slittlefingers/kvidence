#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$script_dir/data/overnight_20261025/shard_0001/prefix_vectors"
remote='ubuntu@YOUR_CLOUD_HOST'
destination='/home/ubuntu/training/data/hiertext_remaining_8639/shard_0001/prefix_vectors/'
[[ -f "$source_dir/prefix_embeddings.npz" && -f "$source_dir/metadata.json" ]] || {
  echo 'Prefix export is not complete.' >&2
  exit 1
}
ssh "$remote" "mkdir -p '$destination'"
rsync -avhP --exclude='device/' "$source_dir/" "$remote:$destination"

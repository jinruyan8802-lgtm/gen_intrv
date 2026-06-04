#!/bin/bash
set -euo pipefail

# =============================================================================
# Audio Pipeline — 生成 / 收集 / 合并 MP3
# =============================================================================
# Usage:
#   ./pipeline.sh generate              对所有 topic 运行 main.py --audio-only
#   ./pipeline.sh copy                  将 output/ 下 mp3 复制到 all_mp3s/
#   ./pipeline.sh merge [output.mp3]    合并 all_mp3s/ 下 mp3 为单个文件
#   ./pipeline.sh all                   一键执行 generate → copy → merge
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------------------------------------------------------------------------
# Subcommands
# -------------------------------------------------------------------------

cmd_generate() {
    local topics=("云计算" "技术中台" "高可用" "高并发系统" "技术中台")

    for topic in "${topics[@]}"; do
        echo "[generate] Running main.py for topic: $topic"
        python main.py "$topic" --audio-only
    done
}

cmd_copy() {
    mkdir -p all_mp3s

    local count
    count=$(find output -name '*.mp3' ! -name 'q0*' | wc -l)

    find output -name "*.mp3" ! -name "q0*" -exec cp {} all_mp3s/ \;
    echo "[copy] Copied $count mp3 files to all_mp3s/"
}

cmd_merge() {
    local input_dir="${1:-all_mp3s}"
    local output="${2:-merged.mp3}"

    if [ ! -d "$input_dir" ]; then
        echo "[merge] Error: directory '$input_dir' not found" >&2
        exit 1
    fi

    local count
    count=$(find "$input_dir" -name '*.mp3' | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "[merge] Error: no mp3 files found in '$input_dir'" >&2
        exit 1
    fi

    local filelist
    filelist=$(mktemp)
    find "$(realpath "$input_dir")" -name "*.mp3" | sort | \
        while read -r f; do echo "file '$f'"; done > "$filelist"

    echo "[merge] Merging $count mp3 files into $output ..."
    ffmpeg -f concat -safe 0 -i "$filelist" -c copy "$output"
    rm "$filelist"

    echo "[merge] Done: $output"
}

cmd_all() {
    cmd_generate
    cmd_copy
    cmd_merge "all_mp3s" "merged.mp3"
}

# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------

cmd="${1:-}"
shift || true

case "$cmd" in
    generate|gen)
        cmd_generate "$@"
        ;;
    copy|cp)
        cmd_copy "$@"
        ;;
    merge|m)
        cmd_merge "$@"
        ;;
    all|a)
        cmd_all "$@"
        ;;
    help|--help|-h|"")
        cat <<'EOF'
Usage: ./pipeline.sh <command> [args]

Commands:
  generate, gen    对所有 topic 运行 main.py --audio-only
  copy, cp         将 output/ 下的 mp3 复制到 all_mp3s/（排除 q0*）
  merge, m [out]   合并 mp3 目录为一个文件（默认输入 all_mp3s/，输出 merged.mp3）
  all, a           一键执行 generate → copy → merge
  help             显示本帮助

Examples:
  ./pipeline.sh generate
  ./pipeline.sh copy
  ./pipeline.sh merge                  # 合并 all_mp3s/ → merged.mp3
  ./pipeline.sh merge my_dir out.mp3   # 合并 my_dir/ → out.mp3
  ./pipeline.sh all
EOF
        ;;
    *)
        echo "Unknown command: $cmd" >&2
        echo "Run './pipeline.sh help' for usage." >&2
        exit 1
        ;;
esac

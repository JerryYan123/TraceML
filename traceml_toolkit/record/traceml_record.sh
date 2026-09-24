#!/usr/bin/env bash
# traceml record: harness-agnostic run recorder + agent respawn loop.
#
# Records any CLI agent's run in the shape `traceml analyze` consumes:
#   - a git sidecar in the workspace: one commit per poll tick that changed files
#   - tags v<N>-true-<score> / v<N>-na whenever submission.csv changes (graded
#     with `mlebench grade-sample`)
#   - native_session.log with respawn markers, used for session-health checks
#   - run_meta.json (harness, competition) if the workspace has none
#
# usage: traceml_record.sh <run_dir> <slug> <minutes> <harness> -- <agent-cmd ...>
#   The agent command is re-run until time is up. It must contain the literal
#   token {PROMPT}: session 1 receives the text of task.md, later sessions the
#   text of cont.md. Each session is capped at 50 minutes (exit code 124).
#
# environment:
#   TRACEML_MLEBENCH        mlebench executable            (default: mlebench)
#   TRACEML_MLEBENCH_CACHE  prepared MLE-bench data dir    (required)
#   TRACEML_POLL            seconds between polls          (default: 60)
#   TRACEML_SESSION_CAP     max seconds per agent session  (default: 3000)
#   TRACEML_BUDGET_SECONDS  overrides <minutes> (used by tests)
set -u

if [ $# -lt 6 ] || [ "$5" != "--" ]; then
  echo "usage: $0 <run_dir> <slug> <minutes> <harness> -- <agent-cmd containing {PROMPT}>" >&2
  exit 2
fi
RUN_DIR=$1; SLUG=$2; MINUTES=$3; HARNESS=$4; shift 5
AGENT_CMD=("$@")

MLEBENCH=${TRACEML_MLEBENCH:-mlebench}
CACHE=${TRACEML_MLEBENCH_CACHE:-}
POLL=${TRACEML_POLL:-60}
SESSION_CAP=${TRACEML_SESSION_CAP:-3000}

die() { echo "traceml record: $*" >&2; exit 2; }
[ -n "$CACHE" ] || die "set TRACEML_MLEBENCH_CACHE to a prepared MLE-bench data directory"
command -v "$MLEBENCH" >/dev/null 2>&1 || die "mlebench not found ($MLEBENCH); set TRACEML_MLEBENCH"
command -v git >/dev/null 2>&1 || die "git not found"
case " ${AGENT_CMD[*]} " in *"{PROMPT}"*) ;; *) die "the agent command must contain the token {PROMPT}";; esac
if command -v md5sum >/dev/null 2>&1; then MD5="md5sum"; else MD5="md5 -r"; fi
TIMEOUT=$(command -v timeout || command -v gtimeout || true)
[ -n "$TIMEOUT" ] || die "GNU timeout not found (macOS: brew install coreutils)"

cd "$RUN_DIR" || die "cannot cd into $RUN_DIR"
[ -f task.md ] || die "task.md (the task prompt) is missing in $RUN_DIR"
[ -f cont.md ] || die "cont.md (the continuation prompt) is missing in $RUN_DIR"
[ -f run_meta.json ] || printf '{"harness": "%s", "competition": "%s", "minutes": %s, "started_unix": %s, "recorder": "traceml_record.sh"}\n' \
  "$HARNESS" "$SLUG" "$MINUTES" "$(date +%s)" > run_meta.json
END=$(( $(date +%s) + ${TRACEML_BUDGET_SECONDS:-$((MINUTES*60))} ))

[ -d .git ] || git init -q
git config user.email traceml@local >/dev/null; git config user.name traceml >/dev/null
for pat in cache/ __pycache__/ '*.pt' '*.pth' '*.bin' '*.ckpt' '*.safetensors' '*.npy' '*.npz' \
           '*.h5' '*.pkl' '*.joblib' '*.parquet' '*.feather' native_session.log tracker.log; do
  grep -qxF "$pat" .gitignore 2>/dev/null || echo "$pat" >> .gitignore
done

# ---- tracker: poll, grade on submission change, commit + tag --------------
tracker() {
  local n prev_md5=""
  n=$(git tag -l 'v*' | sed -E 's/^v([0-9]+)-.*/\1/' | sort -n | tail -1)
  n=${n:-0}
  while [ "$(date +%s)" -lt "$END" ] && [ ! -f _STOP ]; do
    if [ -f submission.csv ]; then
      local md5; md5=$($MD5 submission.csv | cut -d' ' -f1)
      if [ "$md5" != "$prev_md5" ]; then
        prev_md5=$md5; n=$((n+1))
        git add -A >/dev/null 2>&1
        git commit -qm "pre-grade $(date +%H:%M:%S)" --allow-empty >/dev/null 2>&1
        local out score
        out=$("$MLEBENCH" grade-sample submission.csv "$SLUG" --data-dir "$CACHE" 2>&1)
        score=$(echo "$out" | grep -o '"score": *[0-9.eE+-]*' | head -1 | grep -o '[0-9.eE+-]*$')
        if [ -n "$score" ]; then git tag "v${n}-true-${score}" >/dev/null 2>&1
        else git tag "v${n}-na" >/dev/null 2>&1; fi
        echo "[tracker] v${n} score=${score:-na} $(date +%H:%M:%S)" >> tracker.log
      fi
    fi
    # checkpoint commit for code changes between submissions
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
      git add -A >/dev/null 2>&1
      git commit -qm "checkpoint $(date +%H:%M:%S)" >/dev/null 2>&1
    fi
    sleep "$POLL"
  done
}
tracker & TRACKER_PID=$!
trap 'touch _STOP; kill $TRACKER_PID 2>/dev/null' INT TERM

# ---- agent respawn loop -----------------------------------------------------
echo "[loop start pid=$$ end=$END harness=$HARNESS]" >> native_session.log
ITER=0
while [ "$(date +%s)" -lt "$END" ] && [ ! -f _STOP ]; do
  ITER=$((ITER+1))
  REMAIN=$(( END - $(date +%s) ))
  if [ $ITER -eq 1 ]; then PROMPT_FILE=task.md; TAG=task; else PROMPT_FILE=cont.md; TAG=cont; fi
  echo "[respawn iter=$ITER remain=${REMAIN}s prompt=$TAG]" >> native_session.log
  PROMPT_TEXT=$(cat "$PROMPT_FILE")
  CMD=()
  for tok in "${AGENT_CMD[@]}"; do
    if [ "$tok" = "{PROMPT}" ]; then CMD+=("$PROMPT_TEXT"); else CMD+=("$tok"); fi
  done
  CAP=$(( REMAIN < SESSION_CAP ? REMAIN : SESSION_CAP ))
  "$TIMEOUT" "$CAP" "${CMD[@]}" >> native_session.log 2>&1
  RC=$?
  echo "[respawn iter=$ITER exited rc=$RC tag=$TAG]" >> native_session.log
  sleep 5
done
touch _STOP
wait $TRACKER_PID 2>/dev/null
# final checkpoint so the last code state is committed
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  git add -A >/dev/null 2>&1; git commit -qm "final $(date +%H:%M:%S)" >/dev/null 2>&1
fi
rm -f _STOP
echo "[loop end $(date +%H:%M:%S)]" >> native_session.log

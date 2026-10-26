#!/usr/bin/env bash
# Wrap every new levels/level4/models/az_res4f_v3*.onnx with eg_wrap.py once it is stable. Log: data/eg_watch.log
cd /c/Users/melis/Downloads/cloudflight-coding-contest
LOG=levels/level4/agents/hand/data/eg_watch.log
declare -A done
end=$(( $(date +%s) + 3600 ))
while [ $(date +%s) -lt $end ]; do
  for f in levels/level4/models/az_res4f_v3*.onnx; do
    [ -f "$f" ] || continue
    case "$f" in *_eg.onnx|*.eg_part.onnx) continue;; esac
    [ -n "${done[$f]}" ] && continue
    dst0="${f%.onnx}_eg.onnx"; [ -f "$dst0" ] && [ "$dst0" -nt "$f" ] && continue
    s1=$(stat -c %s "$f"); sleep 8; s2=$(stat -c %s "$f")
    [ "$s1" != "$s2" ] && continue
    dst="${f%.onnx}_eg.onnx"
    echo "$(date -u +%H:%M:%S) wrapping $f -> $dst (src md5 $(md5sum "$f" | cut -c1-32))" >> $LOG
    uv run python levels/level4/agents/hand/eg_wrap.py "$f" "$dst" 400 2>&1 | grep -E "^\[|PASS|FAIL|nodes|validator|Error|Traceback" >> $LOG
    done[$f]=1
  done
  sleep 5
done

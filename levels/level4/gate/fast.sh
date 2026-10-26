#!/bin/sh
# Fast protocol for one candidate vs the live seat. usage: sh levels/level4/gate/fast.sh name.onnx [seat.onnx]
# step 1: md5 + caps + validator; step 2: 60-game screen (8 workers). Extension is run separately.
cd /c/Users/melis/Downloads/cloudflight-coding-contest || exit 1
N=$1; SEAT=${2:-az_res4f_v2.onnx}; G=levels/level4/gate; M=levels/level4/models
TAG=$(echo "$N" | sed 's/.onnx//')_$(md5sum $M/$N | cut -c1-8)
date -u +%H:%M:%S
md5sum $M/$N | tee -a $G/tested_md5.txt
uv run python -c "
import onnx; m=onnx.load('$M/$N'); print('caps: nodes',len(m.graph.node),'inits',len(m.graph.initializer))"
for i in 1 2 3; do
  out=$(uv run python levels/level4/c1.py validate $M/$N 2>&1)
  echo "validator try $i: $(echo "$out" | grep -iE "dry run|too slow|ACCEPTED|REJECT" | head -3 | tr '\n' ' ')"
  echo "$out" | grep -q "ACCEPTED --" && break
done
uv run python $G/garena.py $G/fast_$TAG.jsonl 8 $N:$SEAT:60 > $G/fast_$TAG.log 2>&1
uv run python $G/summarize.py $G/fast_$TAG.jsonl
date -u +%H:%M:%S

#!/bin/sh
# Poll models/ for new candidate files until END (UTC HHMM); fast-gate each vs az_res4f_v2. Log: gate/poll.log
cd /c/Users/melis/Downloads/cloudflight-coding-contest || exit 1
G=levels/level4/gate; M=levels/level4/models; END=${1:-1612}; SEAT=az_res4f_v2.onnx
while [ "$(date -u +%H%M)" -lt "$END" ]; do
  for f in $M/*.onnx; do
    n=$(basename "$f")
    case "$n" in champion*|rollback*|random*|seat_*) continue;; esac
    h=$(md5sum "$f" | cut -d' ' -f1)
    grep -q "$h" $G/tested_md5.txt $G/seen_md5.txt && continue
    s1=$(stat -c %Y "$f"); sleep 5; s2=$(stat -c %Y "$f"); [ "$s1" != "$s2" ] && continue
    echo "=== NEW $n $h $(date -u +%H:%M:%S)"
    sh $G/fast.sh "$n" $SEAT
    TAG=$(echo "$n" | sed 's/.onnx//')_$(echo $h | cut -c1-8)
    r=$(uv run python -c "
import json; rows=[json.loads(l) for l in open('$G/fast_$TAG.jsonl') if json.loads(l)['b']=='$SEAT']
print(sum(r['a_won'] for r in rows)/max(1,len(rows)))")
    echo "screen rate $r"
    if uv run python -c "import sys; sys.exit(0 if $r >= 0.58 else 1)"; then
      echo "promising -> extend 160 + greedy 20 + stopgap 30"
      uv run python $G/garena.py $G/fast_$TAG.jsonl 8 $n:$SEAT:160 $n:greedy:20 $n:stopgap_v1.onnx:30 > $G/fast_${TAG}_ext.log 2>&1
      uv run python $G/summarize.py $G/fast_$TAG.jsonl
      uv run python -c "
import json,sys; sys.path.insert(0,'$G'); from summarize import summarize
rows=[json.loads(l) for l in open('$G/fast_$TAG.jsonl')]
v=summarize([r for r in rows if r['b']=='$SEAT']); gr=summarize([r for r in rows if r['b']=='greedy'])
ok = v['n']>=200 and v['lo']>0.60 and min(v['cyan'],v['mag'])>=0.5 and gr['p']>=0.95
print('GATE', 'PASS' if ok else 'FAIL', round(v['lo'],3))
if ok: open('$G/PASS_$TAG','w').write('pass')"
    fi
  done
  sleep 25
done
echo "poll end $(date -u +%H:%M:%S)"

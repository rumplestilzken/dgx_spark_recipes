#!/usr/bin/env bash
set -euo pipefail
KEY=~/.config/NVIDIA/Sync/config/nvsync.key

echo "=== container status ==="
for ip in 133 134 135 136; do
  C=$(ssh -i "$KEY" nick@10.0.0."$ip" 'docker ps --filter name=sparkrun_ --format "{{.Names}}" | grep "node_'${ip/13?/$(( (ip == 133 ) ? 0 : ip - 133 ))))"' | head -1' 2>/dev/null || echo "")
  if [ -n "$C" ]; then
    ST=$(ssh -i "$KEY" nick@10.0.0."$ip" "docker inspect $C --format '{{.State.Health.Status}}' 2>/dev/null || echo unknown")
    echo "node $ip: $C → $ST"
  else
    echo "node $ip: no container"
  fi
done

echo ""
echo "=== RDMA ready? ==="
ssh -i "$KEY" nick@10.0.0.133 'docker ps --filter name=sparkrun_ --format "{{.Names}}" | grep node_0 | head -1' 2>/dev/null | xargs -I{} ssh -i "$KEY" nick@10.0.0.133 "docker exec {} cat /tmp/sparkrun_roce_ready 2>/dev/null || echo no-ready-file"

echo ""
echo "=== port 8000 listening? ==="
for ip in 133 134 135 136; do
  LISTEN=$(ssh -i "$KEY" nick@10.0.0."$ip" 'ss -tlnp | grep ":8000" | head -c 20' 2>/dev/null || echo "no")
  echo "node $ip: port 8000 → $LISTEN"
done

RD=~/Sync/llm_wiki/scratch/dsv41-10m/server_node0_boot.log
if [ -f "$RD" ]; then
  lines=$(wc -l < "$RD")
  echo ""
  echo "=== boot log: ${lines} lines ==="
  grep -nE "(Ready|ROCE_TP|max_total_num_tokens|pool_size|error|Error)" "$RD" 2>/dev/null | tail -15 || echo "(no matches)"
fi

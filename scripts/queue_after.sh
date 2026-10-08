#!/bin/zsh
cd /Users/rishabh/startups/slm
source .venv/bin/activate
export PYTHONPATH=src
while pgrep -f "queue_bakeoff.sh" > /dev/null; do sleep 120; done
mkdir -p results/mqar
python scripts/run_mqar.py > results/mqar/run.log 2>&1
echo "MQAR DONE $(date)" >> results/queue.log

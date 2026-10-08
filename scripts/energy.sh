#!/bin/zsh
# Measure energy per token with powermetrics. Run:  sudo scripts/energy.sh
# (sudo is only needed for powermetrics; the Python workload runs as your user.)
cd "$(dirname "$0")/.."
USER_NAME=${SUDO_USER:-$USER}
mkdir -p results/energy && chown "$USER_NAME" results/energy
powermetrics --samplers cpu_power,gpu_power -i 250 > results/energy/powermetrics.txt 2>&1 &
PM=$!
sudo -u "$USER_NAME" zsh -c "source .venv/bin/activate && PYTHONPATH=src python scripts/energy_bench.py $*"
kill $PM
chown "$USER_NAME" results/energy/powermetrics.txt
sudo -u "$USER_NAME" zsh -c "source .venv/bin/activate && PYTHONPATH=src python scripts/energy_bench.py --parse"

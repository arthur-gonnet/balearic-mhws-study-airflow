#!/bin/bash
# Generates an SSH keypair used only for local Slurm-test-cluster testing (docker compose
# --profile slurm-test), so the Airflow worker container can SSH into the test cluster without a
# password prompt. Not for production use - see hpc/test-cluster/README.md.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p ssh

if [ ! -f ssh/id_ed25519 ]; then
    ssh-keygen -t ed25519 -N "" -f ssh/id_ed25519 -C "slurm-test-cluster"
fi

cp ssh/id_ed25519.pub ssh/authorized_keys
chmod 600 ssh/id_ed25519
chmod 644 ssh/id_ed25519.pub ssh/authorized_keys

cat > ssh/config <<EOF
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    IdentityFile /home/airflow/.ssh/id_ed25519
EOF
chmod 600 ssh/config

echo "SSH keys ready in $(pwd)/ssh"

# id_ed25519 is mode 600 (owner-only) and bind-mounted into the airflow-worker container, which
# runs as AIRFLOW_UID from .env - the container can only read the key if that UID matches the
# host user who generated it (you, right now). Warn loudly if they don't match instead of letting
# it fail later as an unexplained SSH "Permission denied".
env_uid="$(grep -E '^AIRFLOW_UID=' ../../.env 2>/dev/null | cut -d= -f2)"
if [ -n "$env_uid" ] && [ "$env_uid" != "$(id -u)" ]; then
    echo
    echo "WARNING: .env has AIRFLOW_UID=$env_uid, but this key was generated as UID $(id -u)."
    echo "The airflow-worker container won't be able to read it as-is - set AIRFLOW_UID=$(id -u)"
    echo "in .env (or re-run this script as UID $env_uid) before using the Slurm test cluster."
fi

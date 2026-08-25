#!/bin/bash
set -euo pipefail

HOSTNAME="$(hostname)"
sed -i "s/NODE_HOSTNAME/${HOSTNAME}/g" /etc/slurm/slurm.conf

# munge
mkdir -p /etc/munge /run/munge
chown munge:munge /run/munge
if [ ! -f /etc/munge/munge.key ]; then
    /usr/sbin/mungekey -c -f
fi
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key
runuser -u munge -- /usr/sbin/munged

# Slurm spool/log dirs (in case of a fresh volume mount)
mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
chown slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

/usr/sbin/slurmctld
/usr/sbin/slurmd

# SSH
ssh-keygen -A
mkdir -p /run/sshd

echo "Waiting for slurmd to register with slurmctld..."
for _ in $(seq 1 30); do
    if sinfo -h -o "%T" | grep -qE "idle|alloc|mixed"; then
        echo "Node is up."
        break
    fi
    sleep 1
done
sinfo

exec /usr/sbin/sshd -D -e

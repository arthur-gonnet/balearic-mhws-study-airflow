FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# No geo/plotting system libs (eccodes, proj, geos, ...) here unlike the root Dockerfile - this
# node only ever runs `pipelines.mhws_pipeline compute`/`download`, never the plotting code that
# needs them. gcc/g++ stay, for any dependency without a prebuilt wheel on this platform.
RUN apt-get update \
    && apt-get install -y \
        munge \
        slurmd \
        slurmctld \
        slurm-client \
        openssh-server \
        python3 \
        python3-pip \
        gcc \
        g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Single-node "cluster": same container plays controller (slurmctld) and compute node (slurmd).
# No cgroup.conf - ProctrackType=proctrack/linuxproc + TaskPlugin=task/none in slurm.conf avoid
# needing cgroups at all, which containers don't reliably have delegated to them.
COPY hpc/test-cluster/slurm.conf /etc/slurm/slurm.conf

RUN mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm \
    && chown slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

RUN useradd -m -s /bin/bash slurmuser \
    && mkdir -p /home/slurmuser/.ssh \
    && chmod 700 /home/slurmuser/.ssh \
    && chown -R slurmuser:slurmuser /home/slurmuser/.ssh

RUN mkdir -p /var/run/sshd

COPY hpc/test-cluster/requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY hpc/test-cluster/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 22

ENTRYPOINT ["/entrypoint.sh"]

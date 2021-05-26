#!/usr/bin/env bash

set -x

REPO=$1
PARTITION=$2
JOB_NAME=$3
CONFIG=$4
WORK_DIR=$5
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-"-x SH-IDC1-10-198-4-[92,94]"}
PY_ARGS=${@:6}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    mim run ${REPO} train ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}

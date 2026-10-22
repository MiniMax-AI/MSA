"""A launch whose peer rank never arrives must fault at ``wait_timeout_s``, not hang.

Every rank first runs a few launches with the deadline set, which must complete
(normal calls must complete before the deadline). Then ranks 0-2 launch while
rank 3 does not: their TP4 waits can never be satisfied. Each launching rank must
fail with a CUDA launch failure (the kernel prints ``M3_DECODE_WAIT_DEADLINE``)
close to the deadline. A trapped kernel leaves its process's CUDA context
unusable, so each rank reports over the gloo group and exits.

    torchrun --standalone --nproc-per-node=4 \\
        -m tests.inference.high_tps.msa_v1.mega.test_wait_deadline
"""

import argparse
import functools
import json
import os
import time

import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.common.communication import create_tp_allreduce_workspace
from inference.high_tps.msa_v1.common.jit import (
    get_m3_decode_mega_atten_variant,
)
from tests.inference.high_tps.msa_v1.common.m3_decode_fixture import (
    configure_router,
    launch_mega_atten,
    make_model_weights,
    make_slot,
    synchronize_checked,
)

ABSENT_RANK = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("attention",), default="attention")
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--enable-placement", action="store_true")
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    if dist.get_world_size() != 4:
        raise RuntimeError("wait deadline test requires single-node TP4")

    get_m3_decode_mega_atten_variant(wait_deadline=True, enable_placement=args.enable_placement)
    run = functools.partial(launch_mega_atten, enable_placement=args.enable_placement)
    dist.barrier()
    workspace = create_tp_allreduce_workspace(
        rank, 4, f"wait-deadline-{os.environ['MASTER_PORT']}", 0, device.index
    )
    weights = make_model_weights(device, args.seed + rank * 100003, zero_math=False)
    slot = make_slot(device, rank, args.kv_len, args.seed, weights)
    slot.post_attention_rms_weight.fill_(0.125)
    configure_router(slot, slot.layer_input, 20)

    for _ in range(3):
        dist.barrier()
        run(slot, workspace, materialize_topk=True, wait_timeout_s=args.timeout)
        synchronize_checked(device, "wait-deadline")
    dist.barrier()

    if rank == ABSENT_RANK:
        result = {"rank": rank, "case": "absent peer", "launched": False}
        ok = True
    else:
        start = time.monotonic()
        run(slot, workspace, materialize_topk=True, wait_timeout_s=args.timeout)
        try:
            synchronize_checked(device, "wait-deadline")
            outcome = "completed"
        except RuntimeError as error:  # torch.AcceleratorError derives from RuntimeError
            outcome = "launch failure" if "launch failure" in str(error) else str(error)
        elapsed = time.monotonic() - start
        ok = outcome == "launch failure" and elapsed < args.timeout + 5.0
        result = {"rank": rank, "case": "waits on absent peer", "outcome": outcome,
                  "elapsed_s": round(elapsed, 2), "timeout_s": args.timeout}
    print(json.dumps(result), flush=True)
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, ok)
    if rank == 0:
        print("PASS wait deadline" if all(results) else f"FAIL wait deadline {results}",
              flush=True)
    dist.destroy_process_group()
    # The trapped context cannot release the NVLS workspace; exit without teardown.
    os._exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

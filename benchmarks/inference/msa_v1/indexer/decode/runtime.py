"""Resource checks shared by formal decode-indexer benchmarks."""

import socket
import subprocess

import torch


def check_benchmark_device():
    hostname = socket.gethostname().split(".")[0]
    if "galaxy" in hostname.lower() or hostname in ("2u2g-gen-0176", "2u2g-gen-0300"):
        raise RuntimeError("benchmark excludes B300A and galaxy nodes")
    properties = torch.cuda.get_device_properties("cuda")
    if "GB300" not in properties.name or "B300A" in properties.name:
        raise RuntimeError("benchmark requires GB300 NVL, excluding B300A")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("benchmark requires SM103")
    # Establish our context before counting processes; container PIDs differ from host PIDs.
    context_probe = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    gpu_uuid = str(properties.uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    gpu_index = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + gpu_uuid,
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    rows = (line.split() for line in topology.splitlines())
    links = next(
        (row[1:] for row in rows if row and row[0] == "GPU" + gpu_index and "X" in row),
        [],
    )
    if not any(link.startswith("NV") and link[2:].isdigit() for link in links):
        raise RuntimeError("benchmark requires a verified GB300 NVL topology")
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + gpu_uuid,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    if len(processes.split()) != 1:
        raise RuntimeError("benchmark requires an exclusive GPU")
    del context_probe
    return hostname, properties

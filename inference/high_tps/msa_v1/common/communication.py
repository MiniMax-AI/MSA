"""TP4 workspace ownership and IPC rendezvous."""

import array
import hashlib
import os
import socket
import tempfile
import time
from contextlib import suppress

import torch

from .backend import _get_cpp


class Tp4AllReduceWorkspace:
    """Graph-stable TP4 communication workspace owned by one rank."""

    def __init__(self, impl, rank: int, group_id: str, slot_id: int):
        self._impl = impl
        self.rank = int(rank)
        self.group_id = str(group_id)
        self.slot_id = int(slot_id)
        self._next_hidden = impl.next_hidden
        self._closed = False

    @property
    def device(self) -> int:
        return int(self._impl.device)

    @property
    def world_size(self) -> int:
        return int(self._impl.world_size)

    @property
    def size(self) -> int:
        return int(self._impl.size)

    @property
    def unicast_ptr(self) -> int:
        if self._closed:
            raise RuntimeError("TP4 NVLS workspace is closed")
        return int(self._impl.unicast_ptr)

    @property
    def multicast_ptr(self) -> int:
        if self._closed:
            raise RuntimeError("TP4 NVLS workspace is closed")
        return int(self._impl.multicast_ptr)

    @property
    def ready(self) -> bool:
        return not self._closed and bool(self._impl.ready)

    @property
    def next_hidden(self) -> torch.Tensor:
        """BF16 ``[8, 6144]`` view of this rank's NVLS output payload."""

        if self._closed:
            raise RuntimeError("TP4 NVLS workspace is closed")
        return self._next_hidden

    def _close_local(self) -> None:
        if not self._closed:
            self._impl.close()
            self._closed = True

    def __del__(self):
        with suppress(Exception):
            self._close_local()


def _workspace_socket_path(group_id: str, slot_id: int, phase: str) -> str:
    key = f"{group_id}:{slot_id}:{phase}".encode()
    digest = hashlib.sha256(key).hexdigest()[:24]
    return os.path.join(tempfile.gettempdir(), f"mega_atten_nvls_{digest}.sock")


def _connect_workspace_socket(path: str, timeout_s: float) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    while True:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(path)
            client.settimeout(timeout_s)
            return client
        except (FileNotFoundError, ConnectionRefusedError):
            client.close()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out connecting to NVLS rendezvous {path}")
            time.sleep(0.01)


def _send_fd(connection: socket.socket, fd: int) -> None:
    payload = array.array("i", [fd])
    connection.sendmsg([b"N"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, payload)])


def _recv_fd(connection: socket.socket) -> int:
    item_size = array.array("i").itemsize
    message, ancillary, _flags, _address = connection.recvmsg(
        1, socket.CMSG_SPACE(item_size)
    )
    if message != b"N" or not ancillary:
        raise RuntimeError("failed to receive NVLS multicast file descriptor")
    received = array.array("i")
    received.frombytes(ancillary[0][2][:item_size])
    return int(received[0])


def create_tp_allreduce_workspace(
    rank: int,
    world_size: int,
    group_id: str,
    slot_id: int,
    device=None,
    *,
    requested_size: int = 1024 * 1024,
    timeout_s: float = 60.0,
) -> Tp4AllReduceWorkspace:
    """Collectively create one single-node TP4 multicast workspace.

    All four ranks must call this function with the same group and slot IDs.
    The returned virtual addresses stay stable until explicit destruction.
    ``requested_size`` is a lower bound; allocation also reserves the required
    communication state and rounds up to the device allocation granularity.
    """

    rank = int(rank)
    world_size = int(world_size)
    slot_id = int(slot_id)
    timeout_s = float(timeout_s)
    if world_size != 4 or rank < 0 or rank >= world_size:
        raise ValueError("TP4 workspace requires world_size=4 and rank in [0, 4)")
    if not group_id:
        raise ValueError("group_id must be a non-empty string")
    if slot_id < 0:
        raise ValueError("slot_id must be non-negative")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "workspace creation is not allowed during CUDA Graph capture"
        )

    if device is None:
        device_index = torch.cuda.current_device()
    elif isinstance(device, int):
        device_index = device
    else:
        parsed_device = torch.device(device)
        device_index = (
            torch.cuda.current_device()
            if parsed_device.index is None
            else parsed_device.index
        )
    torch.cuda.set_device(device_index)
    torch.empty(1, device=f"cuda:{device_index}")
    impl = _get_cpp().Tp4NvlsWorkspace(
        device_index, int(requested_size), world_size, rank == 0
    )
    path = _workspace_socket_path(str(group_id), slot_id, "create")

    if rank == 0:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connections = []
        exported_fd = -1
        try:
            if os.path.exists(path):
                os.unlink(path)
            server.bind(path)
            server.listen(world_size - 1)
            server.settimeout(timeout_s)
            impl.add_device()
            exported_fd = impl.export_multicast_fd()
            for _ in range(world_size - 1):
                connection, _ = server.accept()
                connection.settimeout(timeout_s)
                connections.append(connection)
                _send_fd(connection, exported_fd)
            for connection in connections:
                if connection.recv(1) != b"A":
                    raise RuntimeError("NVLS peer failed add-device rendezvous")
            impl.map_and_bind()
            impl.zero()
            torch.cuda.synchronize(device_index)
            for connection in connections:
                connection.sendall(b"M")
            for connection in connections:
                if connection.recv(1) != b"Z":
                    raise RuntimeError("NVLS peer failed zero-init rendezvous")
            for connection in connections:
                connection.sendall(b"R")
        finally:
            if exported_fd >= 0:
                os.close(exported_fd)
            for connection in connections:
                connection.close()
            server.close()
            if os.path.exists(path):
                os.unlink(path)
    else:
        with _connect_workspace_socket(path, timeout_s) as connection:
            fd = _recv_fd(connection)
            try:
                impl.import_multicast_fd(fd)
            finally:
                os.close(fd)
            impl.add_device()
            connection.sendall(b"A")
            if connection.recv(1) != b"M":
                raise RuntimeError("NVLS creator failed map rendezvous")
            impl.map_and_bind()
            impl.zero()
            torch.cuda.synchronize(device_index)
            connection.sendall(b"Z")
            if connection.recv(1) != b"R":
                raise RuntimeError("NVLS creator failed ready rendezvous")

    workspace = Tp4AllReduceWorkspace(impl, rank, str(group_id), slot_id)
    if not workspace.ready:
        raise RuntimeError("TP4 NVLS workspace initialization did not complete")
    return workspace


def destroy_tp_allreduce_workspace(
    workspace: Tp4AllReduceWorkspace, *, timeout_s: float = 60.0
) -> None:
    """Collectively synchronize and destroy a TP4 NVLS workspace."""

    if not isinstance(workspace, Tp4AllReduceWorkspace):
        raise TypeError("workspace must be a Tp4AllReduceWorkspace")
    if workspace._closed:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "workspace destruction is not allowed during CUDA Graph capture"
        )

    torch.cuda.synchronize(workspace.device)
    path = _workspace_socket_path(workspace.group_id, workspace.slot_id, "destroy")
    if workspace.rank == 0:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connections = []
        try:
            if os.path.exists(path):
                os.unlink(path)
            server.bind(path)
            server.listen(workspace.world_size - 1)
            server.settimeout(timeout_s)
            for _ in range(workspace.world_size - 1):
                connection, _ = server.accept()
                connection.settimeout(timeout_s)
                connections.append(connection)
            for connection in connections:
                if connection.recv(1) != b"D":
                    raise RuntimeError("NVLS peer failed destroy rendezvous")
            for connection in connections:
                connection.sendall(b"R")
        finally:
            for connection in connections:
                connection.close()
            server.close()
            if os.path.exists(path):
                os.unlink(path)
    else:
        with _connect_workspace_socket(path, float(timeout_s)) as connection:
            connection.sendall(b"D")
            if connection.recv(1) != b"R":
                raise RuntimeError("NVLS creator failed destroy rendezvous")
    workspace._close_local()

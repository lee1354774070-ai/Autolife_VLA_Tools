#!/usr/bin/env python3
"""Expose a TCP service through SSH session channels when forwarding is disabled.

The SSH master must already be authenticated.  Each incoming TCP connection is
bridged to a short-lived remote ``nc`` process over a multiplexed SSH session,
so no additional password or MFA prompt is needed.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import socketserver
import subprocess
import threading


COPY_SIZE = 256 * 1024


def copy_socket_to_pipe(source: socket.socket, destination) -> None:
    try:
        while data := source.recv(COPY_SIZE):
            destination.write(data)
            destination.flush()
    except (BrokenPipeError, ConnectionError, OSError):
        pass
    finally:
        try:
            destination.close()
        except OSError:
            pass


def copy_pipe_to_socket(source, destination: socket.socket) -> None:
    try:
        while data := source.read(COPY_SIZE):
            destination.sendall(data)
    except (BrokenPipeError, ConnectionError, OSError):
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class RelayHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        command = [
            server.ssh_binary,
            "-S",
            server.control_path,
            "-T",
            "-p",
            str(server.ssh_port),
            server.ssh_target,
            "nc",
            server.remote_host,
            str(server.remote_port),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        assert process.stdin is not None and process.stdout is not None
        upload = threading.Thread(
            target=copy_socket_to_pipe,
            args=(self.request, process.stdin),
            daemon=True,
        )
        upload.start()
        copy_pipe_to_socket(process.stdout, self.request)
        upload.join(timeout=1.0)
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()


class ThreadedRelay(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8765)
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=8765)
    parser.add_argument("--ssh-binary", default=shutil.which("ssh") or "ssh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with ThreadedRelay((args.listen_host, args.listen_port), RelayHandler) as server:
        server.control_path = args.control_path
        server.ssh_target = args.ssh_target
        server.ssh_port = args.ssh_port
        server.remote_host = args.remote_host
        server.remote_port = args.remote_port
        server.ssh_binary = args.ssh_binary
        print(f"Relay listening on {args.listen_host}:{args.listen_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

"""Read-only resource probe for the remote training server.

Credentials stay in the local desktop config file and are never printed.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import paramiko


CONFIG_PATH = Path(r"C:\Users\HUAWEI\Desktop\servercode.txt")


def load_connection(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([^\s:=]+)\s*[:=]?\s+(.+?)\s*$", line)
        if match:
            values[match.group(1).lower()] = match.group(2)
    host = values.get("ip") or values.get("host")
    username = values.get("用户名") or values.get("username") or values.get("user")
    password = values.get("密码") or values.get("password") or values.get("pass")
    if not all((host, username, password)):
        raise ValueError("servercode.txt is missing IP, username, or password")
    return host, username, password


def main() -> int:
    host, username, password = load_connection(CONFIG_PATH)
    client = paramiko.SSHClient()
    # Accept an unknown host for this one session but do not persist a host key.
    client.set_missing_host_key_policy(paramiko.WarningPolicy())
    try:
        client.connect(
            hostname=host,
            username=username,
            password=password,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
            look_for_keys=False,
            allow_agent=False,
        )
        transport = client.get_transport()
        key = transport.get_remote_server_key() if transport else None
        if key:
            fingerprint = hashlib.sha256(key.asbytes()).hexdigest()
            print(f"SSH host-key SHA256: {fingerprint}")
        command = r"""
echo '== host / uptime =='
hostname
uptime
printf '\n== CPU / memory ==\n'
free -h
printf '\n== disk ==\n'
df -hT -x tmpfs -x devtmpfs
printf '\n== GPU ==\n'
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits
printf '\n== GPU compute processes ==\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
"""
        _, stdout, stderr = client.exec_command(command, timeout=25)
        output = stdout.read().decode("utf-8", errors="replace").strip()
        error = stderr.read().decode("utf-8", errors="replace").strip()
        if output:
            print(output)
        if error:
            print("== remote command diagnostics ==", file=sys.stderr)
            print(error, file=sys.stderr)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

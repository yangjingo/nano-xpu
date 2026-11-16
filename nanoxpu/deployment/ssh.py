"""SSH carries a fixed helper command; configuration is sent only as JSON stdin."""
import json
from pathlib import Path
import subprocess

from nanoxpu.deployment.schema import RuntimeProfile, ipv4


class SSHExecutor:
    def __init__(self, profile: RuntimeProfile):
        self.profile = profile

    def command(self, ip):
        ip = ipv4(ip)
        ssh = self.profile.ssh
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                   "-o", "ConnectTimeout=10", "-p", str(ssh.port)]
        if ssh.identity_file:
            command += ["-i", str(Path(ssh.identity_file).expanduser())]
        if ssh.known_hosts_file:
            # OpenSSH parses -o values itself: quote paths containing spaces.
            path = Path(ssh.known_hosts_file).expanduser().as_posix().replace('\\', '\\\\').replace('"', '\\"')
            command += ["-o", f'UserKnownHostsFile="{path}"']
        # helper_python is constrained to a safe absolute path by RuntimeProfile.
        return [*command, f"{ssh.user}@{ip}", self.profile.runtime.helper_python,
                "-m", "nanoxpu.deployment.helper"]

    def request(self, ip, payload, timeout=30):
        result = subprocess.run(self.command(ip), input=json.dumps(payload), capture_output=True,
                                text=True, timeout=timeout)
        try:
            response = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            # Never echo stdin or arbitrary SSH banner/stderr into public logs.
            raise RuntimeError(f"SSH helper at {ip} returned no valid response (exit {result.returncode})") from exc
        if result.returncode or not response.get("ok"):
            raise RuntimeError(f"helper at {ip} failed: {response.get('error', 'unknown error')}")
        return response["result"]

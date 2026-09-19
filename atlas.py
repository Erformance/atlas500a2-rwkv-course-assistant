#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atlas 500 A2 远程助手（paramiko 密码登录）。

用法：
    python atlas.py run "npu-smi info"            # develop 提权到 root 后执行（默认）
    python atlas.py run "ls /" --admin            # admin 的 IES 受限 shell
    python atlas.py put local.py /home/disk/x.py  # SFTP 上传
    python atlas.py get /home/disk/x.log local.log

凭据不写在代码里。二选一：
    export ATLAS_PWD='...' ATLAS_ROOT_PWD='...'
  或者在同目录建一个 atlas.env（已加进 .gitignore，不会提交）：
    ATLAS_HOST=192.168.31.50
    ATLAS_USER=admin
    ATLAS_PWD=...
    ATLAS_ROOT_PWD=...
优先级：环境变量 > atlas.env > 默认值。
"""

import os
import re
import sys
import time

import paramiko

try:                                    # Windows 控制台默认 GBK，会写崩模型输出的特殊字符
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def _load_env_file(path):
    """读 atlas.env 里的 KEY=VALUE，已存在的环境变量优先。"""
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip().strip("'\"")
    except OSError:
        pass
    return values


_ENV_FILE = _load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "atlas.env"))


def _conf(name, default=""):
    return os.environ.get(name) or _ENV_FILE.get(name) or default


HOST = _conf("ATLAS_HOST", "192.168.31.50")
USER = _conf("ATLAS_USER", "admin")
PWD = _conf("ATLAS_PWD")
ROOT_PWD = _conf("ATLAS_ROOT_PWD") or PWD


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, 22, USER, PWD, look_for_keys=False, allow_agent=False,
                timeout=30, banner_timeout=180, auth_timeout=180)
    return cli


class Shell:
    """带 PTY 的持久 shell。root=True 时先走 develop 提权。"""

    def __init__(self, cli, root=True):
        self.chan = cli.get_transport().open_session()
        self.chan.get_pty(term="vt100", width=220, height=60)
        self.chan.settimeout(5)
        self.buf = ""
        self.chan.invoke_shell()
        if root:
            self._wait_for(r"IES:|/->", 60)
            self.buf = ""
            self.chan.sendall("develop\n")
            self._wait_for(r"Password", 60)
            self.buf = ""
            self.chan.sendall(ROOT_PWD + "\n")
            if not self._wait_for(r"#\s*$", 180):
                raise RuntimeError("develop 提权失败，收到内容：\n" + self.buf[-2000:])
        else:
            self._wait_for(r"IES:|/->", 25)

    def _drain(self):
        try:
            data = self.chan.recv(65536)
        except Exception:
            return ""
        if not data:
            return ""
        self.buf += data.decode("utf-8", "replace")
        return data

    def _wait_for(self, pattern, timeout):
        rx = re.compile(pattern, re.M)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if rx.search(self.buf):
                return True
            if not self._drain():
                time.sleep(0.05)
        return False

    def run(self, cmd, timeout=3600):
        """执行命令，返回 (exit_code, 去掉回显的文本输出)。"""
        tag = "DONE%d" % (int(time.time() * 1000) % 100000000)
        # 清掉可能残留的多行续行状态（引号没闭合时 shell 会一直等输入）
        self.buf = ""
        self.chan.sendall("\x03")
        time.sleep(0.4)
        self.buf = ""
        self.chan.sendall(cmd + "\n")
        time.sleep(0.3)
        self.chan.sendall("echo %s:$?\n" % tag)
        deadline = time.time() + timeout
        rx = re.compile(r"%s:(\d+)" % tag)
        while time.time() < deadline:
            m = rx.search(self.buf)
            if m:
                lines = self.buf[:m.start()].split("\n")
                if lines and cmd.split("\n")[0][:40] in lines[0]:
                    lines = lines[1:]
                text = "\n".join(lines)
                text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
                text = "\n".join(l for l in text.split("\n")
                                 if tag not in l and l.strip() != cmd.strip())
                return int(m.group(1)), text
            if not self._drain():
                time.sleep(0.05)
        return 124, self.buf + "\n[atlas.py] 等待超时"

    def idle_run(self, cmd, idle=3.0, timeout=1800):
        """给不支持标记串的受限 shell 用：输出静默 idle 秒即认为结束。"""
        self.buf = ""
        self.chan.sendall(cmd + "\n")
        deadline = time.time() + timeout
        last = time.time()
        while time.time() < deadline:
            if self._drain():
                last = time.time()
            elif time.time() - last > idle:
                break
            time.sleep(0.05)
        return 0, self.buf

    def close(self):
        try:
            self.chan.close()
        except Exception:
            pass


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    args = sys.argv[2:]
    admin_mode = "--admin" in args
    args = [a for a in args if a != "--admin"]

    cli = connect()
    try:
        if mode == "run":
            sh = Shell(cli, root=not admin_mode)
            code, out = (sh.idle_run(args[0]) if admin_mode else sh.run(args[0]))
            sh.close()
            sys.stdout.write(out.replace("\r\n", "\n"))
            if not out.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.write("[exit %d]\n" % code)
            return code
        if mode == "put":
            sftp = cli.open_sftp()
            sftp.put(args[0], args[1])
            sftp.close()
            print("上传 %s -> %s" % (args[0], args[1]))
            return 0
        if mode == "get":
            sftp = cli.open_sftp()
            sftp.get(args[0], args[1])
            sftp.close()
            print("下载 %s -> %s" % (args[0], args[1]))
            return 0
        print(__doc__)
        return 2
    finally:
        cli.close()


if __name__ == "__main__":
    raise SystemExit(main())

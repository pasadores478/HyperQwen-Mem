"""Detached launcher: starts server.py as its own session (survives this shell).
Usage: python launch.py  [start|stop|status]
"""
import os, signal, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.environ.get("VENV", "/tmp/hmenv"), "bin", "python")
LOG = "/tmp/hqmem.log"
PIDF = os.path.join(BASE, "data", "server.pid")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def start():
    os.makedirs(os.path.dirname(PIDF), exist_ok=True)
    if os.path.exists(PIDF):
        old = int(open(PIDF).read().strip() or 0)
        if old and _pid_alive(old):
            os.kill(old, signal.SIGTERM)
            for _ in range(20):
                if not _pid_alive(old):
                    break
                time.sleep(0.2)
    # fresh DB unless MEM_DB env points elsewhere and file exists
    logf = open(LOG, "wb")
    p = subprocess.Popen(
        [PY, os.path.join(BASE, "server.py")],
        cwd=BASE, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,  # fully detach: own session, survives our shell
    )
    open(PIDF, "w").write(str(p.pid))
    # wait for readiness
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:8321/stats", timeout=1)
            print(f"[launch] up: pid={p.pid} log={LOG}")
            return 0
        except Exception:
            time.sleep(1)
    print(f"[launch] not ready after 60s; check {LOG}")
    return 1


def stop():
    if not os.path.exists(PIDF):
        print("no pidfile")
        return 1
    pid = int(open(PIDF).read().strip() or 0)
    if pid and _pid_alive(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.2)
        print(f"stopped {pid}")
    os.remove(PIDF)
    return 0


def status():
    import urllib.request
    if not os.path.exists(PIDF):
        print("no pidfile -> not running")
        return 1
    pid = int(open(PIDF).read().strip() or 0)
    print(f"pidfile pid={pid} alive={_pid_alive(pid)}")
    try:
        print(urllib.request.urlopen("http://127.0.0.1:8321/stats", timeout=2).read().decode())
        return 0
    except Exception as e:
        print("not responding:", e)
        return 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    sys.exit({"start": start, "stop": stop, "status": status}.get(cmd, start)())

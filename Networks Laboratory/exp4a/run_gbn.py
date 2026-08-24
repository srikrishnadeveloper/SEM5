import os
import subprocess
import sys
import time

# drives the go-back-n experiment: runs one server per scenario, saves output

exp_dir = os.path.dirname(os.path.abspath(__file__))

scenarios = [
    ("tc1", "8", "4", "-1", "-1"),   # error free
    ("tc2", "8", "4", "3", "-1"),    # single frame loss
    ("tc3", "6", "3", "-1", "2"),    # ack loss
    ("tc4", "10", "4", "2,7", "-1"), # multiple frame loss
    ("tc5", "8", "2", "5", "-1"),    # different window size
]

for tag, n, w, lost_frames, lost_ack in scenarios:
    server_out = open(os.path.join(exp_dir, f"gbn_{tag}_server.txt"), "w", encoding="utf-8", buffering=1)
    server_proc = subprocess.Popen(
        [sys.executable, "-u", "gbn_server.py", lost_frames, lost_ack],
        cwd=exp_dir,
        stdout=server_out,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    time.sleep(0.8)
    with open(os.path.join(exp_dir, f"gbn_{tag}_client.txt"), "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-u", "gbn_client.py", n, w], cwd=exp_dir, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.3)
    server_proc.terminate()
    try:
        server_proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        server_proc.kill()
    server_out.flush()
    server_out.close()

print("Experiment complete. Check gbn_tc1..tc5 server and client outputs.")
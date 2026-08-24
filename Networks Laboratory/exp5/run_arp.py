import os
import subprocess
import sys
import threading
import time

# drives the arp experiment: starts the router server with packet details,
# runs three clients (two with non-matching IPs, one matching), saves output

exp_dir = os.path.dirname(os.path.abspath(__file__))

server_out = open(os.path.join(exp_dir, "arp_server_output.txt"), "w", encoding="utf-8", buffering=1)
server_proc = subprocess.Popen(
    [sys.executable, "-u", "arp_server.py", "155.157.65.128", "123.128.34.56",
     "AF-45-E5-00-97-12", "1011110000101010", "3"],
    cwd=exp_dir,
    stdout=server_out,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)
time.sleep(0.8)

# all three clients must be connected when the broadcast happens,
# so they are started at the same time
clients = [
    ("165.43.158.158", "09-DF-90-26-6C-09", "1"),
    ("155.157.65.128", "45-DA-62-21-1A-B2", "2"),
    ("15.143.158.18", "19-0F-01-63-C7-D4", "3"),
]

def run_client(ip, mac, cid):
    with open(os.path.join(exp_dir, f"arp_client{cid}_output.txt"), "w", encoding="utf-8", buffering=1) as f:
        subprocess.run(
            [sys.executable, "-u", "arp_client.py", ip, mac, cid],
            cwd=exp_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

threads = [threading.Thread(target=run_client, args=args) for args in clients]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=15)

time.sleep(1.0)
server_proc.terminate()
try:
    server_proc.wait(timeout=3)
except subprocess.TimeoutExpired:
    server_proc.kill()
server_out.flush()
server_out.close()

# validation run: invalid IP must be rejected before any broadcast
with open(os.path.join(exp_dir, "arp_validation_output.txt"), "w", encoding="utf-8") as f:
    subprocess.run(
        [sys.executable, "-u", "arp_server.py", "999.999.999.999", "123.128.34.56",
         "AF-45-E5-00-97-12", "1011110000101010", "0"],
        cwd=exp_dir,
        stdout=f,
        stderr=subprocess.STDOUT,
    )

print("Experiment complete. Check arp_server_output.txt, arp_client1..3_output.txt, arp_validation_output.txt.")
import os
import socket
import subprocess
import sys
import threading
import time

# drives the whole experiment: starts server, runs two clients at the same time, saves output

exp_dir = os.path.dirname(os.path.abspath(__file__))

server_out = open(os.path.join(exp_dir, "server_output.txt"), "w", encoding="utf-8", buffering=1)

def wait_for_server(timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect(("127.0.0.1", 5000))
            s.close()
            return True
        except Exception:
            time.sleep(0.05)
    return False

server_proc = subprocess.Popen(
    [sys.executable, "-u", "server.py"],
    cwd=exp_dir,
    stdout=server_out,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)

if not wait_for_server():
    print("Server did not start in time")
    server_proc.terminate()
    sys.exit(1)

def run_client(cid):
    with open(os.path.join(exp_dir, cid + "_output.txt"), "w", encoding="utf-8", buffering=1) as f:
        subprocess.run(
            [sys.executable, "-u", "client_driver.py", cid, exp_dir],
            cwd=exp_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

t1 = threading.Thread(target=run_client, args=("client1",))
t2 = threading.Thread(target=run_client, args=("client2",))
t1.start()
t2.start()
t1.join(timeout=15)
t2.join(timeout=15)

# let the server finish printing, then close it cleanly
time.sleep(0.6)
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 5000))
    s.sendall(b"exit")
    s.close()
except Exception:
    pass

time.sleep(0.3)
server_proc.terminate()
try:
    server_proc.wait(timeout=3)
except subprocess.TimeoutExpired:
    server_proc.kill()

server_out.flush()
server_out.close()
print("Experiment complete. Check server_output.txt, client1_output.txt, client2_output.txt.")

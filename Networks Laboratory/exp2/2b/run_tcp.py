import os
import socket
import subprocess
import sys
import threading
import time

# drives the tcp experiment: starts server, runs two clients, saves output

exp_dir = os.path.dirname(os.path.abspath(__file__))

server_out = open(os.path.join(exp_dir, "tcp_server_output.txt"), "w", encoding="utf-8", buffering=1)

server_proc = subprocess.Popen(
    [sys.executable, "-u", "tcp_server.py"],
    cwd=exp_dir,
    stdout=server_out,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)

# just give the server a moment to bind instead of probing with a dummy connection
time.sleep(0.8)

def run_client(cid, roll):
    with open(os.path.join(exp_dir, cid + "_output.txt"), "w", encoding="utf-8", buffering=1) as f:
        subprocess.run(
            [sys.executable, "-u", "tcp_client_driver.py", cid, exp_dir, roll],
            cwd=exp_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

t1 = threading.Thread(target=run_client, args=("tcp_client1", "3122245001002"))
t2 = threading.Thread(target=run_client, args=("tcp_client2", "9999999999"))
t1.start()
t2.start()
t1.join(timeout=15)
t2.join(timeout=15)

# send an exit to close the server log cleanly
time.sleep(0.4)
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 5002))
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
print("Experiment complete. Check tcp_server_output.txt, tcp_client1_output.txt, tcp_client2_output.txt.")
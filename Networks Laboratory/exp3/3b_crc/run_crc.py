import os
import subprocess
import sys
import threading
import time

# drives the crc experiment: starts server, runs the 5 test cases, saves output

exp_dir = os.path.dirname(os.path.abspath(__file__))

server_out = open(os.path.join(exp_dir, "crc_server_output.txt"), "w", encoding="utf-8", buffering=1)

server_proc = subprocess.Popen(
    [sys.executable, "-u", "crc_server.py"],
    cwd=exp_dir,
    stdout=server_out,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)

# give the server a moment to bind
time.sleep(0.8)

def run_client(cid, frame, poly, err):
    with open(os.path.join(exp_dir, cid + "_output.txt"), "w", encoding="utf-8", buffering=1) as f:
        subprocess.run(
            [sys.executable, "-u", "crc_client_driver.py", frame, poly, err],
            cwd=exp_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

# test case 1: no error
run_client("crc_client1", "1101011011", "10011", "none")
# test case 2: single-bit error
run_client("crc_client2", "1101011011", "10011", "single")
# test case 3: multiple-bit error
run_client("crc_client3", "1011101010", "1101", "multiple")
# test cases 4 & 5: burst error + clean frame, run at the same time
t4 = threading.Thread(target=run_client, args=("crc_client4", "111001101010", "10011", "burst"))
t5 = threading.Thread(target=run_client, args=("crc_client5", "100100", "1101", "none"))
t4.start()
t5.start()
t4.join(timeout=15)
t5.join(timeout=15)

# give logs a moment to flush, then stop the server
time.sleep(0.4)
server_proc.terminate()
try:
    server_proc.wait(timeout=3)
except subprocess.TimeoutExpired:
    server_proc.kill()

server_out.flush()
server_out.close()
print("Experiment complete. Check crc_server_output.txt and crc_client1..5_output.txt.")
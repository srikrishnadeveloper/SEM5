import os
import socket
import subprocess
import sys
import threading
import time

# drives the checksum experiment: starts server, runs the 5 test cases, saves output

exp_dir = os.path.dirname(os.path.abspath(__file__))

server_out = open(os.path.join(exp_dir, "checksum_server_output.txt"), "w", encoding="utf-8", buffering=1)

server_proc = subprocess.Popen(
    [sys.executable, "-u", "checksum_server.py"],
    cwd=exp_dir,
    stdout=server_out,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)

# give the server a moment to bind
time.sleep(0.8)

def run_client(cid, data, err):
    with open(os.path.join(exp_dir, cid + "_output.txt"), "w", encoding="utf-8", buffering=1) as f:
        subprocess.run(
            [sys.executable, "-u", "checksum_client_driver.py", data, err],
            cwd=exp_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

# test case 1: no error
run_client("checksum_client1", "10101010,11001100,11110000", "none")
# test case 2: single-bit error
run_client("checksum_client2", "10101010,11001100,11110000", "single")
# test case 3: multiple-bit error
run_client("checksum_client3", "10101010,11001100,11110000", "multiple")
# test cases 4 & 5: text message + banking transaction, run at the same time
t4 = threading.Thread(target=run_client, args=("checksum_client4", "Transaction: ACC1001 Deposit Rs.5000", "none"))
t5 = threading.Thread(target=run_client, args=("checksum_client5", "ACC1001 TXN45896 Amount 2500", "none"))
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
print("Experiment complete. Check checksum_server_output.txt and checksum_client1..5_output.txt.")
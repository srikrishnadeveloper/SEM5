import os
import sys
import time

# this is a helper to feed the client sample inputs and capture its output

client_id = sys.argv[1]
exp_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
os.chdir(exp_dir)

if client_id == "client1":
    requests = [
        "Account Verification - ACC1001",
        "Transaction Confirmation - TXN45879",
        "exit",
    ]
else:
    requests = [
        "Balance Enquiry - ACC2005",
        "Mini Statement Request",
        "Fund Transfer Validation",
        "exit",
    ]

idx = 0

def input(prompt=""):
    global idx
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if idx >= len(requests):
        raise EOFError
    val = requests[idx]
    idx += 1
    sys.stdout.write(val + "\n")
    sys.stdout.flush()
    time.sleep(0.05)  # tiny pause so both clients can interleave
    return val

code_path = os.path.join(exp_dir, "client.py")
with open(code_path, encoding="utf-8") as f:
    code = f.read()

g = {"__builtins__": __builtins__, "input": input}

try:
    exec(code, g)
except (EOFError, ConnectionResetError, ConnectionAbortedError, OSError, BrokenPipeError):
    pass
finally:
    obj = g.get("client")
    if obj and hasattr(obj, "close"):
        try:
            obj.close()
        except Exception:
            pass

import os
import sys
import time

# helper to feed the tcp client one roll number and capture its output

client_id = sys.argv[1]
exp_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
roll = sys.argv[3] if len(sys.argv) > 3 else "3122245001312"
os.chdir(exp_dir)

def input(prompt=""):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    sys.stdout.write(roll + "\n")
    sys.stdout.flush()
    time.sleep(0.3)
    return roll

code_path = os.path.join(exp_dir, "tcp_client.py")
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
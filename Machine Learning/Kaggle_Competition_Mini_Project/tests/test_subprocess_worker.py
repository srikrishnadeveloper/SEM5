"""
tests/test_subprocess_worker.py
Tests the isolated worker process pattern.
"""
import subprocess
import sys
import pickle
from pathlib import Path

script_content = '''
import argparse, pickle, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--out-file", type=str, required=True)
args = parser.parse_args()

print(f"Worker running for GPU {args.gpu_id}...")
results = [{"test": f"gpu_{args.gpu_id}_success"}]
with open(args.out_file, "wb") as f:
    pickle.dump(results, f)
print(f"Worker {args.gpu_id} done.")
'''

worker_file = Path("scratch/test_worker_runner.py")
worker_file.parent.mkdir(parents=True, exist_ok=True)
worker_file.write_text(script_content)

out0 = Path("scratch/out_0.pkl")
out1 = Path("scratch/out_1.pkl")

p0 = subprocess.Popen([sys.executable, str(worker_file), "--gpu-id", "0", "--out-file", str(out0)])
p1 = subprocess.Popen([sys.executable, str(worker_file), "--gpu-id", "1", "--out-file", str(out1)])

p0.wait()
p1.wait()

assert p0.returncode == 0, f"Worker 0 failed: {p0.returncode}"
assert p1.returncode == 0, f"Worker 1 failed: {p1.returncode}"

with open(out0, "rb") as f:
    d0 = pickle.load(f)
with open(out1, "rb") as f:
    d1 = pickle.load(f)

print("Worker 0 result:", d0)
print("Worker 1 result:", d1)
print("ALL SUBPROCESS TESTS PASSED!")

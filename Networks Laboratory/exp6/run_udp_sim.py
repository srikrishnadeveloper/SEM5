import os
import random
import simpy
import sys

# runs the udp client-server simulation for the four network conditions.
# tc2 is printed with full per-packet detail, the others as summaries
# usage: run_udp_sim.py [seed]

from udp_client import UDPClient
from udp_server import UDPServer

random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 42)  # reproducible runs

exp_dir = os.path.dirname(os.path.abspath(__file__))
client_out = open(os.path.join(exp_dir, "udp_client_output.txt"), "w", encoding="utf-8")
server_out = open(os.path.join(exp_dir, "udp_server_output.txt"), "w", encoding="utf-8")

# packets, size, interval, loss probability, delay min, delay max
scenarios = [
    ("TC1", 10, 100, 0.2, 0.0, 0.0, 0.0),
    ("TC2", 20, 100, 0.2, 0.1, 0.01, 0.05),
    ("TC3", 30, 200, 0.1, 0.2, 0.02, 0.10),
    ("TC4", 50, 200, 0.067, 0.3, 0.05, 0.15),
]

for tag, packets, size, interval, loss_prob, dmin, dmax in scenarios:
    detail = tag == "TC2"
    print(f"========== {tag} ==========", file=client_out)
    print(f"========== {tag} ==========", file=server_out)
    env = simpy.Environment()
    channel = simpy.Store(env)
    client = UDPClient(env, channel, packets, size, interval, loss_prob, dmin, dmax, client_out, detail)
    server = UDPServer(env, channel, packets, server_out, detail)
    env.process(client.run())
    env.process(server.run())
    # run long enough for every packet to be generated and delivered
    # (delays push generation times later, so the bound is generous)
    env.run(until=packets * (interval + dmax) + 1.0)
    server.summary()

client_out.close()
server_out.close()
print("Experiment complete. Check udp_client_output.txt and udp_server_output.txt.")
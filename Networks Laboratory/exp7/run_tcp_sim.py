import os
import simpy

# runs the tcp flow control simulation for the five scenarios.
# tc1 is printed with full per-segment detail, the others as summaries

from tcp_client import TCPClient
from tcp_server import TCPServer

exp_dir = os.path.dirname(os.path.abspath(__file__))
client_out = open(os.path.join(exp_dir, "tcp_client_output.txt"), "w", encoding="utf-8")
server_out = open(os.path.join(exp_dir, "tcp_server_output.txt"), "w", encoding="utf-8")

DELAY = 0.05  # one-way transmission delay in seconds

# total_kb, seg_kb, init window kb, receiver buffer kb, sender rate, receiver rate
scenarios = [
    ("TC1", 10, 1, 4, 8, 2, 4),
    ("TC2", 20, 1, 8, 4, 8, 4),
    ("TC3", 20, 1, 8, 2, 8, 2),
    ("TC4", 20, 1, 8, 8, 8, 4),
    ("TC5", 20, 2, 8, 8, 8, 4),
]

for tag, total_kb, seg_kb, init_win, buffer_kb, send_rate, recv_rate in scenarios:
    detail = tag == "TC1"
    print(f"========== {tag} ==========", file=client_out)
    print(f"========== {tag} ==========", file=server_out)
    env = simpy.Environment()
    data_ch = simpy.Store(env, capacity=float("inf"))
    ack_ch = simpy.Store(env, capacity=float("inf"))
    client = TCPClient(env, data_ch, ack_ch, total_kb, seg_kb, init_win, send_rate, DELAY, buffer_kb, client_out, detail)
    server = TCPServer(env, data_ch, ack_ch, buffer_kb, recv_rate, server_out, detail)
    env.process(client.run())
    env.process(server.run_receive())
    env.process(server.run_process())
    # generous bound: generation time plus processing of every segment
    segments = total_kb // seg_kb
    env.run(until=segments * (seg_kb / send_rate) + segments * (seg_kb / recv_rate) + 5.0)
    server.summary()

client_out.close()
server_out.close()
print("Experiment complete. Check tcp_client_output.txt and tcp_server_output.txt.")
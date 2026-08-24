import socket
import sys

# go-back-n sender client. sends frames inside a sliding window and retransmits
# the whole outstanding window when a timeout happens.
# usage: gbn_client.py <number_of_frames> <window_size>

n_frames = int(sys.argv[1])
window = int(sys.argv[2])

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 5006))
client.settimeout(1.0)

base = 0
total_sent = 0
retransmissions = 0
acks = 0
buf = ""

def read_line():
    # read one newline terminated line, returns None on timeout or close
    global buf
    while "\n" not in buf:
        try:
            chunk = client.recv(1024).decode()
        except socket.timeout:
            return None
        if not chunk:
            return None
        buf += chunk
    line, buf = buf.split("\n", 1)
    return line.strip()

def send_frames(frames, announce=True):
    # send one window of frames
    global total_sent
    if announce:
        if len(frames) == 1:
            print(f"Sending Frame: {frames[0]}")
        else:
            print("Sending Frames:", " ".join(map(str, frames)))
    for frame in frames:
        client.sendall(f"FRAME {frame}\n".encode())
        total_sent += 1

def wait_for_acks(end):
    # receive acks until the whole window is acked, returns False on timeout
    global base, acks
    while base < end:
        resp = read_line()
        if resp is None:
            return False
        if resp.startswith("ACK"):
            k = int(resp.split()[1])
            print(f"ACK Received: {k}")
            acks += 1
            if k == base:
                base += 1
    return True

while base < n_frames:
    end = min(base + window, n_frames)
    send_frames(list(range(base, end)))
    if wait_for_acks(end):
        continue
    # timeout: retransmit every outstanding frame from the current base
    print(f"Timeout for Frame {base}")
    retrans = list(range(base, min(base + window, n_frames)))
    print("Retransmitting Frames:", " ".join(map(str, retrans)))
    retransmissions += len(retrans)
    send_frames(retrans, announce=False)
    # keep waiting for the retransmitted window, retrying on repeated timeouts
    while not wait_for_acks(min(base + window, n_frames)):
        retrans = list(range(base, min(base + window, n_frames)))
        print(f"Timeout for Frame {base}")
        print("Retransmitting Frames:", " ".join(map(str, retrans)))
        retransmissions += len(retrans)
        send_frames(retrans, announce=False)

client.sendall(f"DONE {n_frames}\n".encode())
print("Transmission Completed Successfully")
print()
print(f"Frames Sent             : {total_sent}")
print(f"Frames Retransmitted    : {retransmissions}")
print(f"ACKs Received           : {acks}")
print(f"Transmission Efficiency : {n_frames / total_sent * 100:.2f}%")
client.close()
import socket
import sys
import threading

# go-back-n receiver server. simulates frame loss and ack loss based on user input.
# usage: gbn_server.py <lost_frames_csv or -1> <lost_ack or -1>

lost_frames = []
lost_ack = -1
if len(sys.argv) >= 2 and sys.argv[1] != "-1":
    lost_frames = [int(x) for x in sys.argv[1].split(",")]
if len(sys.argv) >= 3 and sys.argv[2] != "-1":
    lost_ack = int(sys.argv[2])

print_lock = threading.Lock()

def log(msg):
    # lock so two threads never write into the same line
    with print_lock:
        print(msg)

def handle_client(conn, addr):
    log(f"Client Connected: {addr}")
    try:
        # line based protocol so multiple frames in one tcp chunk stay separate
        f = conn.makefile("r", encoding="utf-8")
        expected = 0          # next frame number we are waiting for
        n_frames = None
        received = set()      # frames accepted so far
        retrans_pending = -1  # frame whose retransmission we are waiting for
        while True:
            line = f.readline()
            if not line:
                break
            msg = line.strip()
            parts = msg.split()
            if parts[0] == "DONE":
                n_frames = int(parts[1])
                break
            if parts[0] == "FRAME":
                n = int(parts[1])
                if n == expected:
                    if n in lost_frames:
                        # frame never arrives, sender must retransmit
                        log(f"Frame {n} Lost")
                        lost_frames.remove(n)
                        retrans_pending = n
                    elif retrans_pending == n:
                        log(f"Retransmitted Frame {n} Received")
                        log(f"ACK {n} Sent")
                        conn.sendall(f"ACK {n}\n".encode())
                        expected += 1
                        retrans_pending = -1
                        received.add(n)
                    elif n == lost_ack:
                        # frame is received but the ack is lost on the way back
                        log(f"Frame {n} Received")
                        log(f"ACK {n} Lost")
                        expected += 1
                        received.add(n)
                    else:
                        log(f"Frame {n} Received")
                        log(f"ACK {n} Sent")
                        conn.sendall(f"ACK {n}\n".encode())
                        expected += 1
                        received.add(n)
                elif n in received:
                    # duplicate: frame was already accepted, resend cumulative ack
                    log(f"Duplicate Frame {n} Received")
                    log(f"ACK {n} Sent")
                    conn.sendall(f"ACK {n}\n".encode())
                else:
                    # out of order frame while the lost one is still missing
                    log(f"Discarding Frame {n}")
        if n_frames is not None and expected == n_frames:
            log("All Frames Received Successfully")
        else:
            log("Connection closed before all frames arrived")
    except ConnectionResetError:
        pass
    conn.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 5006))
server.listen(5)
print(f"Go-Back-N Receiver Server Started (lost frames: {lost_frames}, lost ack: {lost_ack})")
print("Server listening on 127.0.0.1:5006")

while True:
    conn, addr = server.accept()
    # one thread per client so multiple senders are served concurrently
    t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
    t.start()
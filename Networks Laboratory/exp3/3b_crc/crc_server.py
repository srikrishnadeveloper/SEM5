import socket
import threading

# traffic monitoring server using CRC verification over tcp

print_lock = threading.Lock()

def log(msg):
    # lock so two threads never write into the same line
    with print_lock:
        print(msg)

def crc_verify(frame, poly):
    # divide the received frame by the generator polynomial (xor long division)
    frame = list(frame)
    n = len(poly)
    for i in range(len(frame) - n + 1):
        if frame[i] == "1":
            for j in range(n):
                frame[i + j] = "1" if frame[i + j] != poly[j] else "0"
    # remainder is the last (n-1) bits, all zeros means the frame is clean
    return "".join(frame[-(n - 1):]).count("1") == 0

def handle_client(conn, addr):
    log(f"Client Connected: {addr}")
    try:
        msg = conn.recv(1024).decode().strip()
        frame, poly = msg.split("|")
        log(f"[Client {addr[1]}] Received frame: {frame}")
        log(f"[Client {addr[1]}] Generator polynomial: {poly}")
        if crc_verify(frame, poly):
            reply = "CRC Verification Successful\nNo Error Detected\nFrame Accepted"
            log(f"[Client {addr[1]}] CRC remainder zero, frame accepted")
        else:
            reply = "CRC Verification Failed\nError Detected\nFrame Rejected"
            log(f"[Client {addr[1]}] CRC remainder non-zero, frame rejected")
        conn.sendall(reply.encode())
    except Exception as e:
        log(f"[Client {addr[1]}] Error: {e}")
    finally:
        log(f"Client {addr} disconnected.")
        conn.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 5005))
server.listen(5)
print("Traffic Monitoring CRC Server Started...")
print("Server listening on 127.0.0.1:5005")

while True:
    conn, addr = server.accept()
    # one thread per client so all cameras are served at the same time
    t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
    t.start()
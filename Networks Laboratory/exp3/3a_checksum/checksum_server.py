import socket
import threading

# banking transaction verification server using 1's complement checksum

print_lock = threading.Lock()

def log(msg):
    # lock so two threads never write into the same line
    with print_lock:
        print(msg)

def verify(words, checksum):
    # 1's complement sum of all words + checksum should give all ones (0xFF)
    total = 0
    for w in words:
        total += int(w, 2)
    total += int(checksum, 2)
    while total > 0xFF:
        # wrap the carry around back into the sum
        total = (total & 0xFF) + (total >> 8)
    return total == 0xFF

def handle_client(conn, addr):
    log(f"Client Connected: {addr}")
    try:
        msg = conn.recv(1024).decode().strip()
        words, checksum = msg.split("|")
        words = words.split(",")
        log(f"[Client {addr[1]}] Received data words: {words}")
        log(f"[Client {addr[1]}] Received checksum: {checksum}")
        if verify(words, checksum):
            reply = "Checksum Status : Valid\nTransaction Processed Successfully"
            log(f"[Client {addr[1]}] Checksum verified, transaction accepted")
        else:
            reply = "Checksum Status : Corrupted\nTransaction Rejected"
            log(f"[Client {addr[1]}] Checksum mismatch, transaction rejected")
        conn.sendall(reply.encode())
    except Exception as e:
        log(f"[Client {addr[1]}] Error: {e}")
    finally:
        log(f"Client {addr} disconnected.")
        conn.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 5004))
server.listen(5)
print("Banking Transaction Verification Server Started...")
print("Server listening on 127.0.0.1:5004")

while True:
    conn, addr = server.accept()
    # one thread per client so multiple branches are served at the same time
    t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
    t.start()
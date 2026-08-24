import socket
import threading

# socket module for network stuff, threading to handle many clients

def handle_client(conn, addr):
    print(f"Client Connected: {addr}")
    try:
        while True:
            data = conn.recv(1024).decode().strip()
            if not data:
                break

            print(f"[Client {addr[1]}] Received: {data}")

            if data.lower() == "exit":
                # client wants to leave, so we stop this thread
                break

            conn.sendall(data.encode())
            print(f"[Client {addr[1]}] Echo Sent: {data}")
    except ConnectionResetError:
        pass
    finally:
        print(f"Client {addr} disconnected.")
        conn.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# allow fast restart of server on same port

server.bind(("127.0.0.1", 5000))
# listen only on this machine, port 5000

server.listen(5)
print("Banking Validation Server Started...")
print("Server listening on 127.0.0.1:5000")

while True:
    conn, addr = server.accept()
    # every client gets its own thread so they run at the same time
    t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
    t.start()

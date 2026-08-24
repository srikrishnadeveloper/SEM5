import socket

# UDP banking validation server, connectionless so no accept/threads needed

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(("127.0.0.1", 5001))
print("UDP Banking Validation Server Started...")
print("Server listening on 127.0.0.1:5001")

while True:
    data, addr = server.recvfrom(1024)
    msg = data.decode().strip()
    print(f"[Client {addr[1]}] Received: {msg}")
    if msg.lower() == "exit":
        # udp has no connection to close, just ignore the exit request
        continue
    server.sendto(data, addr)
    print(f"[Client {addr[1]}] Echo Sent: {msg}")
import socket

# UDP client for banking transaction validation

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("UDP Client ready (connectionless, just sending datagrams).\n")

while True:
    req = input("Enter Transaction Request: ")
    client.sendto(req.encode(), ("127.0.0.1", 5001))

    if req.strip().lower() == "exit":
        # nothing to close, udp has no connection
        print("Sent exit. No connection to terminate.")
        break

    ack, addr = client.recvfrom(1024)
    print("Server Acknowledgment:")
    print(ack.decode())

client.close()
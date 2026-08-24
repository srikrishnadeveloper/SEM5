import socket

# socket module to talk to the server

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 5000))

print("Connected to Banking Validation Server.\n")

while True:
    req = input("Enter Transaction Request: ")
    client.sendall(req.encode())

    if req.strip().lower() == "exit":
        # no reply expected, just leave
        print("Disconnected from Server.")
        break

    ack = client.recv(1024).decode()
    print("Server Acknowledgment:")
    print(ack)

client.close()

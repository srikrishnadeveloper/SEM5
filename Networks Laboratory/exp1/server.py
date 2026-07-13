import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 5000))

server.listen(5)
print("Server is waiting for connection...")

conn, addr = server.accept()
print("Connected by : ", addr)

while True:

    data = conn.recv(1024).decode()

    print("Client Says : ", data)
    if data.strip().lower() == "bye":
        print("Client ended chat")
        break

    string = input("Enter your message to client : ")

    conn.send(string.encode())


conn.close()
server.close()

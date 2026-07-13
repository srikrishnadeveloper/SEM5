import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client.connect(("127.0.0.1", 5000))

print("Connected to server")

while True:

    string = input("Enter your message to server : ")
    client.send(string.encode())

    data = client.recv(1024).decode()
    print("Server says : ", data)
    if data.strip().lower() == "bye":
        print("Client ended chat")
        break


client.close()

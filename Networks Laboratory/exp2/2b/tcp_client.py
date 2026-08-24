import socket

# student client: asks for one roll number and prints what the server sends back

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 5002))

roll = input("Enter Roll Number: ")
client.sendall(roll.encode())

reply = client.recv(1024).decode()
print("Server Response:")
print(reply)

client.close()
print("Disconnected from Server.")
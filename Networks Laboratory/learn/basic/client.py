import socket

Host = "127.0.0.1"
port = 5000

#client initlize
client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

#connect
client.connect((Host,port))

#input message from the user
message = input("enter your message")

client.send(message.encode())

data = client.recv(1024).decode()
print(data)
client.close()
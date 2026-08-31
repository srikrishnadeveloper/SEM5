import socket

Host = "127.0.0.1"
port = 5000

#initlisised the server
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

#sever binding
server.bind((Host,port))

#server listen
print("listeing for the client")
server.listen()

# accepting it
conn,addr = server.accept()

# get client message
data = conn.recv(1024).decode()

#printing the client message
print(data.encode())

#sending to the client
conn.send(data.encode())

#closing the socket

conn.close()
server.close()





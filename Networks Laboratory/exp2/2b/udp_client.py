import socket

# udp student client: sends roll number and prints the server's reply

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

roll = input("Enter Roll Number: ")
client.sendto(roll.encode(), ("127.0.0.1", 5003))

reply, addr = client.recvfrom(1024)
print("Server Response:")
print(reply.decode())

client.close()
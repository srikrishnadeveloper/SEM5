import csv
import socket

# udp student record server, checks roll number and sends back the name

def load_students():
    records = {}
    with open("students.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records[row["Roll Number"].strip()] = row
    return records

records = load_students()
print(f"Loaded {len(records)} student records from students.csv")

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(("127.0.0.1", 5003))
print("UDP Student Record Server Started...")
print("Server listening on 127.0.0.1:5003")

while True:
    data, addr = server.recvfrom(1024)
    roll = data.decode().strip()
    print(f"[Client {addr[1]}] Requested Roll Number: {roll}")
    if roll.lower() == "exit":
        # connectionless, nothing to close
        continue
    if roll in records:
        reply = f"Roll Number: {roll}\nName: {records[roll]['Name']}"
        print(f"[Client {addr[1]}] Record found, sending name")
    else:
        reply = "Student Record Not Found."
        print(f"[Client {addr[1]}] Record not found")
    server.sendto(reply.encode(), addr)
import csv
import socket
import threading

# student record server, reads students.csv and serves records over tcp

def load_students():
    records = {}
    with open("students.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records[row["Roll Number"].strip()] = row
    return records

records = load_students()
print(f"Loaded {len(records)} student records from students.csv")

print_lock = threading.Lock()

def log(msg):
    # lock so two threads never write into the same line
    with print_lock:
        print(msg)

def handle_client(conn, addr):
    log(f"Client Connected: {addr}")
    try:
        roll = conn.recv(1024).decode().strip()
        log(f"[Client {addr[1]}] Requested Roll Number: {roll}")
        if roll.lower() == "exit":
            return
        if roll in records:
            r = records[roll]
            reply = (
                f"Roll Number: {r['Roll Number']}\n"
                f"Name: {r['Name']}\n"
                f"Department: {r['Department']}\n"
                f"Semester: {r['Semester']}\n"
                f"CGPA: {r['CGPA']}"
            )
            log(f"[Client {addr[1]}] Record found, sending details")
        else:
            reply = "Student Record Not Found."
            log(f"[Client {addr[1]}] Record not found")
        conn.sendall(reply.encode())
    except ConnectionResetError:
        pass
    finally:
        log(f"Client {addr} disconnected.")
        conn.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 5002))
server.listen(5)
print("Student Record Server Started...")
print("Server listening on 127.0.0.1:5002")

while True:
    conn, addr = server.accept()
    # one thread per client so everyone is served at the same time
    t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
    t.start()
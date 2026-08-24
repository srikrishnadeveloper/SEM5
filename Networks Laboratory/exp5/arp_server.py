import re
import socket
import sys
import time

# arp server (host / router). broadcasts an ARP request to all connected clients,
# waits for the ARP reply, then sends the packet to the matching destination.
# usage: arp_server.py <dest_ip> <src_ip> <src_mac> <data_16bit> <num_clients>

def is_valid_ip(ip):
    # check if the IP is four numbers 0-255 separated by dots
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit():
            return False
        if int(p) < 0 or int(p) > 255:
            return False
    return True

def is_valid_mac(mac):
    # check MAC like XX-XX-XX-XX-XX-XX with hex digits
    pattern = r"^[0-9A-Fa-f]{2}(-[0-9A-Fa-f]{2}){5}$"
    return re.match(pattern, mac) is not None

dest_ip, src_ip, src_mac, data, num = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

print("Enter the details of packet received.")
print(f"Destination IP : {dest_ip}")
print(f"Source IP : {src_ip}")
print(f"Source MAC : {src_mac}")
print(f"16 bit data : {data}")

# check the validity of IP and MAC addresses
if not (is_valid_ip(dest_ip) and is_valid_ip(src_ip)):
    print("Invalid IP address. Packet rejected.")
    sys.exit()
if not is_valid_mac(src_mac) or len(data) != 16 or any(c not in "01" for c in data):
    print("Invalid MAC address or data. Packet rejected.")
    sys.exit()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 5007))
server.listen(5)

clients = []
print("Waiting for clients to connect...")
for _ in range(num):
    conn, addr = server.accept()
    conn.settimeout(0.5)
    clients.append(conn)
print(f"{num} clients connected.")

req = f"{src_ip} | {src_mac} | {dest_ip}"
print("Developing ARP Request packet")
print(req)
for c in clients:
    # broadcast the ARP request to every client
    c.sendall(f"ARP_REQUEST|{req}\n".encode())
print("The ARP Request packet is broadcasted.")
print("Waiting for ARP Reply...")

reply = None
deadline = time.time() + 8
while time.time() < deadline and reply is None:
    for c in clients:
        try:
            line = c.recv(1024).decode().strip()
        except socket.timeout:
            continue
        if not line:
            continue
        if line.startswith("ARP_REPLY"):
            reply = (c, line)
            break

if reply:
    conn, msg = reply
    print("ARP Reply received")
    print(msg.split("|", 1)[1].strip())
    dest_mac = msg.split("|")[4].strip()
    print(f"Sending the packet to : {dest_mac}")
    packet = f"{src_ip} | {src_mac} | {dest_ip} | {dest_mac} | {data}"
    print("Packet Sent:", packet)
    conn.sendall(f"PACKET|{packet}\n".encode())
    time.sleep(0.5)
else:
    print("No ARP Reply received within the timeout.")

for c in clients:
    c.close()
server.close()
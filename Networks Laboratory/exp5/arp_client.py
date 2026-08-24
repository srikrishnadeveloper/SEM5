import socket
import sys

# arp client (end host). checks incoming ARP requests and replies only when the
# destination IP matches its own IP, then receives the packet from the server.
# usage: arp_client.py <own_ip> <own_mac> <client_no>

own_ip, own_mac, cid = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"Enter the IP address : {own_ip}")
print(f"Enter the Mac address : {own_mac}")

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 5007))
f = client.makefile("r", encoding="utf-8")

line = f.readline().strip()
parts = line.split("|")
if parts[0] == "ARP_REQUEST":
    req = " | ".join(p.strip() for p in parts[1:4])
    print(f"ARP Request Received : {req}")
    src_ip, src_mac, dest_ip = [p.strip() for p in parts[1:4]]
    if dest_ip == own_ip:
        # our own IP is being resolved, send the ARP reply
        print("IP address matches")
        reply = f"ARP_REPLY|{src_ip} | {src_mac} | {dest_ip} | {own_mac}"
        print("ARP Reply Sent :", " | ".join(p.strip() for p in reply.split("|")[1:]))
        client.sendall((reply + "\n").encode())
        # now wait for the actual packet from the server
        line2 = f.readline().strip()
        parts2 = line2.split("|")
        if parts2[0] == "PACKET":
            print("Received Packet is :", " | ".join(p.strip() for p in parts2[1:]))
    else:
        print("IP address does not match.")

client.close()
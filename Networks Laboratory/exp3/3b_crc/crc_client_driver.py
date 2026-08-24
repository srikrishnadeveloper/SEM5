import socket
import sys

# crc client driver, scenario passed as argv instead of stdin
# usage: crc_client_driver.py <frame> <poly> <error_mode>
# error_mode: none / single / multiple / burst

def crc_remainder(data, poly):
    frame = list(data + "0" * (len(poly) - 1))
    n = len(poly)
    for i in range(len(frame) - n + 1):
        if frame[i] == "1":
            for j in range(n):
                frame[i + j] = "1" if frame[i + j] != poly[j] else "0"
    return "".join(frame[-(n - 1):])

frame, poly, err = sys.argv[1], sys.argv[2], sys.argv[3]

print("Data frame:", frame)
print("Generator polynomial:", poly)
rem = crc_remainder(frame, poly)
print("CRC remainder:", rem)
tx_frame = frame + rem
print("Transmitted frame (data + CRC):", tx_frame)

if err == "single":
    # single-bit error: flip one bit in the data part
    frame = format(int(frame, 2) ^ 0b1000, "0%db" % len(frame))
    print("Single-bit error introduced ->", frame)
elif err == "multiple":
    # multiple-bit error: flip two separated bits in the data part
    frame = format(int(frame, 2) ^ 0b10001000, "0%db" % len(frame))
    print("Multiple-bit errors introduced ->", frame)
elif err == "burst":
    # burst error: flip a run of 4 consecutive bits in the data part
    frame = format(int(frame, 2) ^ 0b1111000, "0%db" % len(frame))
    print("Burst error introduced ->", frame)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 5005))
print("Frame and polynomial sent to server...")
client.sendall((frame + rem + "|" + poly).encode())
reply = client.recv(1024).decode()
print("Server reply:")
print(reply)
client.close()
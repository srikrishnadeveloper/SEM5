import socket

# traffic camera client, computes CRC and appends it to the frame

def crc_remainder(data, poly):
    # xor long division of data (padded with zeros) by the generator polynomial
    frame = list(data + "0" * (len(poly) - 1))
    n = len(poly)
    for i in range(len(frame) - n + 1):
        if frame[i] == "1":
            for j in range(n):
                frame[i + j] = "1" if frame[i + j] != poly[j] else "0"
    return "".join(frame[-(n - 1):])

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 5005))

frame = input("Enter data frame (binary): ")
poly = input("Enter generator polynomial (binary): ")
err = input("Introduce error? (0 none, 1 single-bit, 2 multiple-bit, 3 burst): ")

print("Data frame:", frame)
print("Generator polynomial:", poly)
rem = crc_remainder(frame, poly)
print("CRC remainder:", rem)
tx_frame = frame + rem
print("Transmitted frame (data + CRC):", tx_frame)

if err == "1":
    # single-bit error: flip one bit in the data part
    frame = format(int(frame, 2) ^ 0b1000, "0%db" % len(frame))
    print("Single-bit error introduced ->", frame)
elif err == "2":
    # multiple-bit error: flip two separated bits in the data part
    frame = format(int(frame, 2) ^ 0b10001000, "0%db" % len(frame))
    print("Multiple-bit errors introduced ->", frame)
elif err == "3":
    # burst error: flip a run of 4 consecutive bits in the data part
    frame = format(int(frame, 2) ^ 0b1111000, "0%db" % len(frame))
    print("Burst error introduced ->", frame)

client.sendall((frame + rem + "|" + poly).encode())
print("Frame and polynomial sent to server...")
reply = client.recv(1024).decode()
print("Server reply:")
print(reply)
client.close()
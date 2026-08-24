import socket
import sys

# checksum client driver, scenario passed as argv instead of stdin
# usage: checksum_client_driver.py <data> <error_mode>
# error_mode: none / single / multiple

def checksum_1s(words):
    total = 0
    for w in words:
        total += int(w, 2)
    while total > 0xFF:
        total = (total & 0xFF) + (total >> 8)
    return format(total ^ 0xFF, "08b")

def to_words(text):
    return [format(b, "08b") for b in text.encode()]

data, err = sys.argv[1], sys.argv[2]

if "," in data:
    words = [w.strip() for w in data.split(",")]
    print("Message type: 8-bit binary words")
else:
    words = to_words(data)
    print("Message type: text message")
print("Data words:", words)

checksum = checksum_1s(words)
print("1's complement checksum:", checksum)

if err == "single":
    # single-bit error: flip one bit of word 2 (simulates a noisy line)
    words[1] = format(int(words[1], 2) ^ 0b00010000, "08b")
    print("Single-bit error introduced in word 2 ->", words[1])
elif err == "multiple":
    # multiple-bit error: flip one bit in every word
    words[0] = format(int(words[0], 2) ^ 0b00100000, "08b")
    words[1] = format(int(words[1], 2) ^ 0b00010000, "08b")
    words[2] = format(int(words[2], 2) ^ 0b00000010, "08b")
    print("Multiple-bit errors introduced ->", words)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 5004))
print("Data and checksum sent to server...")
client.sendall((",".join(words) + "|" + checksum).encode())
reply = client.recv(1024).decode()
print("Server reply:")
print(reply)
client.close()
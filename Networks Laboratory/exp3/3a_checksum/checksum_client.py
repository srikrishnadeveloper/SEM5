import socket

# banking client, computes 1's complement checksum and sends data + checksum

def checksum_1s(words):
    # sum all words, wrap the carry, then invert to get the checksum
    total = 0
    for w in words:
        total += int(w, 2)
    while total > 0xFF:
        total = (total & 0xFF) + (total >> 8)
    return format(total ^ 0xFF, "08b")

def to_words(text):
    # break a text message into 8-bit ascii words
    return [format(b, "08b") for b in text.encode()]

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 5004))

msg = input("Enter data words (8-bit, comma separated) or text message: ")
err = input("Introduce error? (0 none, 1 single-bit, 2 multiple-bit): ")

if "," in msg:
    words = [w.strip() for w in msg.split(",")]
else:
    words = to_words(msg)
print("Data words:", words)

checksum = checksum_1s(words)
print("1's complement checksum:", checksum)

if err == "1":
    # single-bit error: flip one bit of word 2 (simulates a noisy line)
    words[1] = format(int(words[1], 2) ^ 0b00010000, "08b")
    print("Single-bit error introduced in word 2 ->", words[1])
elif err == "2":
    # multiple-bit error: flip one bit in every word
    words[0] = format(int(words[0], 2) ^ 0b00100000, "08b")
    words[1] = format(int(words[1], 2) ^ 0b00010000, "08b")
    words[2] = format(int(words[2], 2) ^ 0b00000010, "08b")
    print("Multiple-bit errors introduced ->", words)

client.sendall((",".join(words) + "|" + checksum).encode())
print("Data and checksum sent to server...")
reply = client.recv(1024).decode()
print("Server reply:")
print(reply)
client.close()
# udp server simulation module: receives packets from the channel, records
# arrival times, computes the delay, and reports statistics for the run

class UDPServer:
    def __init__(self, env, channel, packets, out, detail):
        self.env = env
        self.channel = channel
        self.packets = packets
        self.out = out
        self.detail = detail
        self.received = []
        self.delays = []
        self.expected = 1

    def run(self):
        o = self.out
        print("UDP SERVER SIMULATION", file=o)
        print(f"Total Packets Expected : {self.packets}", file=o)
        print(file=o)
        while True:
            seq, gen_time, size = yield self.channel.get()
            arrival = self.env.now
            delay = arrival - gen_time
            self.received.append(seq)
            self.delays.append(delay)
            if self.detail:
                print(f"Packet {seq} Received", file=o)
                print(f"Generation Time : {gen_time:.2f} s", file=o)
                print(f"Arrival Time    : {arrival:.2f} s", file=o)
                print(f"Delay           : {delay:.2f} s", file=o)
                print(file=o)

    def summary(self):
        o = self.out
        received_count = len(self.received)
        lost = self.packets - received_count
        print("Packets Received        :", received_count, file=o)
        print("Packets Lost            :", lost, file=o)
        print("Packet Loss Ratio       :", f"{lost / self.packets * 100:.1f}%", file=o)
        print("Packet Delivery Ratio   :", f"{received_count / self.packets * 100:.1f}%", file=o)
        if self.delays:
            print("Minimum Delay           :", f"{min(self.delays):.2f} s", file=o)
            print("Maximum Delay           :", f"{max(self.delays):.2f} s", file=o)
            print("Average Delay           :", f"{sum(self.delays) / len(self.delays):.2f} s", file=o)
        # detect missing packets from the sequence numbers
        missing = [s for s in range(1, self.packets + 1) if s not in self.received]
        if missing:
            print("Missing Packets         :", " ".join(map(str, missing)), file=o)
        print(file=o)
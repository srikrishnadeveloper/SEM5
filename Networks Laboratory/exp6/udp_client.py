import random

# udp client simulation module: generates packets periodically with a simpy
# process, applies a random transmission delay, and loses packets based on
# the loss probability before putting them on the communication channel

class UDPClient:
    def __init__(self, env, channel, packets, size, interval, loss_prob,
                 delay_min, delay_max, out, detail):
        self.env = env
        self.channel = channel
        self.packets = packets
        self.size = size
        self.interval = interval
        self.loss_prob = loss_prob
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.out = out
        self.detail = detail
        self.generated = 0
        self.transmitted = 0
        self.lost = 0

    def run(self):
        o = self.out
        print("UDP CLIENT SIMULATION", file=o)
        print(f"Packets to Generate : {self.packets}", file=o)
        print(f"Packet Size         : {self.size} bytes", file=o)
        print(f"Generation Interval : {int(self.interval * 1000)} ms", file=o)
        print(f"Loss Probability    : {int(self.loss_prob * 100)}%", file=o)
        print(f"Delay Range         : {int(self.delay_min * 1000)}-{int(self.delay_max * 1000)} ms", file=o)
        print(file=o)
        for seq in range(1, self.packets + 1):
            yield self.env.timeout(self.interval)
            gen_time = self.env.now
            if self.detail:
                print(f"Packet {seq} Generated at {gen_time:.2f} s", file=o)
            self.generated += 1
            if random.random() < self.loss_prob:
                # packet is lost based on the loss probability
                if self.detail:
                    print(f"Packet {seq} Lost", file=o)
                self.lost += 1
                continue
            # random transmission delay before the packet reaches the channel
            delay = random.uniform(self.delay_min, self.delay_max)
            yield self.env.timeout(delay)
            if self.detail:
                print(f"Packet {seq} Transmitted", file=o)
            self.transmitted += 1
            yield self.channel.put((seq, gen_time, self.size))
            if self.detail:
                print(file=o)
        print(f"Total Packets Generated   : {self.generated}", file=o)
        print(f"Total Packets Transmitted : {self.transmitted}", file=o)
        print(f"Total Packets Lost        : {self.lost}", file=o)
        print(file=o)
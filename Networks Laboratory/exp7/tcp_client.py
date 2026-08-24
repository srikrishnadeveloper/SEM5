# tcp client simulation module: generates segments and transmits them only
# when space is available in the receiver's advertised window

class TCPClient:
    def __init__(self, env, data_ch, ack_ch, total_kb, seg_kb, init_window_kb,
                 sender_rate, delay, buffer_kb, out, detail):
        self.env = env
        self.data_ch = data_ch
        self.ack_ch = ack_ch
        self.total_kb = total_kb
        self.seg_kb = seg_kb
        self.init_window_kb = init_window_kb
        self.sender_rate = sender_rate
        self.delay = delay
        self.buffer_kb = buffer_kb
        self.out = out
        self.detail = detail
        self.transmitted = 0
        self.acks = 0
        self.waiting_events = 0
        self.outstanding = 0          # unacknowledged segments
        self.last_rwnd = buffer_kb    # receiver advertised window from acks

    def handle_ack(self, msg):
        kind, seq, rwnd = msg
        self.last_rwnd = rwnd
        if kind == "ACK":
            self.outstanding -= 1
            self.acks += 1
            if self.detail:
                print(f"ACK Received : S{seq}", file=self.out)
                print(f"Advertised Window : {rwnd} KB", file=self.out)

    def run(self):
        o = self.out
        print("TCP CLIENT SIMULATION", file=o)
        print(f"Total Data        : {self.total_kb} KB", file=o)
        print(f"Segment Size      : {self.seg_kb} KB", file=o)
        print(f"Initial Window    : {self.init_window_kb} KB", file=o)
        print(f"Sender Rate       : {self.sender_rate} KB/s", file=o)
        print(f"Transmission Delay: {self.delay} s", file=o)
        print(file=o)
        for i in range(1, self.total_kb // self.seg_kb + 1):
            # generate the next segment at the sender's rate
            yield self.env.timeout(self.seg_kb / self.sender_rate)
            # transmit only when the advertised window has space
            waited = False
            while self.outstanding * self.seg_kb >= min(self.init_window_kb, self.last_rwnd):
                # one waiting event per time the sender is held back
                if not waited:
                    waited = True
                    self.waiting_events += 1
                    if self.detail:
                        print("Sender Waiting - Receiver Window Full", file=o)
                # wait for an ack or a window update from the receiver
                ack = yield self.ack_ch.get()
                self.handle_ack(ack)
            if self.detail:
                print(f"Segment S{i} Transmitted", file=o)
            self.transmitted += 1
            self.outstanding += 1
            yield self.env.timeout(self.delay)
            yield self.data_ch.put((i, self.seg_kb))
        # drain the remaining acks until everything is acknowledged
        while self.outstanding > 0:
            ack = yield self.ack_ch.get()
            self.handle_ack(ack)
        print("Transmission Completed", file=o)
        print(f"Total Segments Sent    : {self.transmitted}", file=o)
        print(f"Total ACKs Received    : {self.acks}", file=o)
        print(f"Total Waiting Events   : {self.waiting_events}", file=o)
        print(f"Total Transmission Time: {self.env.now:.2f} s", file=o)
        print(f"Throughput             : {self.total_kb / self.env.now:.2f} KB/s", file=o)
        print(file=o)
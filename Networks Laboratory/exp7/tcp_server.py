# tcp server simulation module: receives segments into a receive buffer,
# generates cumulative acks, processes them at the receiver's rate and
# advertises the available window (rwnd). receive and processing run as
# separate processes so the buffer can fill up while data is being processed

class TCPServer:
    def __init__(self, env, data_ch, ack_ch, buffer_kb, recv_rate, out, detail):
        self.env = env
        self.data_ch = data_ch
        self.ack_ch = ack_ch
        self.buffer_kb = buffer_kb
        self.recv_rate = recv_rate
        self.out = out
        self.detail = detail
        self.occupancy = 0
        self.queue = []            # sizes of received but not processed segments
        self.received = 0
        self.acks_generated = 0
        self.full_events = 0
        self.buffer_was_full = False
        self.occupancy_history = []
        self.window_history = []

    def record(self):
        # keep a sample of occupancy and advertised window after each change
        self.occupancy_history.append(self.occupancy)
        self.window_history.append(self.buffer_kb - self.occupancy)

    def run_receive(self):
        o = self.out
        while True:
            seq, size = yield self.data_ch.get()
            self.queue.append(size)
            self.occupancy += size
            self.received += 1
            self.acks_generated += 1
            self.record()
            if self.detail:
                print(f"Segment S{seq} Received", file=o)
                print(f"Buffer Occupancy : {self.occupancy} KB", file=o)
                print(f"Advertised Window : {self.buffer_kb - self.occupancy} KB", file=o)
                print(f"ACK S{seq} Sent", file=o)
            yield self.ack_ch.put(("ACK", seq, self.buffer_kb - self.occupancy))
            if self.occupancy == self.buffer_kb:
                self.full_events += 1
                self.buffer_was_full = True
                if self.detail:
                    print("Receive Buffer Full", file=o)
                    print("Sender Transmission Temporarily Restricted", file=o)

    def run_process(self):
        o = self.out
        while True:
            if not self.queue:
                # idle poll when nothing is waiting to be processed
                yield self.env.timeout(0.01)
                continue
            size = self.queue[0]
            yield self.env.timeout(size / self.recv_rate)
            self.queue.pop(0)
            self.occupancy -= size
            self.record()
            if self.detail:
                print("Processing Data...", file=o)
                print(f"Buffer Occupancy : {self.occupancy} KB", file=o)
                print(f"Advertised Window : {self.buffer_kb - self.occupancy} KB", file=o)
            if self.buffer_was_full:
                self.buffer_was_full = False
                if self.detail:
                    print("Buffer Space Available", file=o)
                    print("Sender Can Continue Transmission", file=o)
            # window update so a waiting sender can resume
            yield self.ack_ch.put(("WND", 0, self.buffer_kb - self.occupancy))

    def summary(self):
        o = self.out
        occ = self.occupancy_history
        wnd = self.window_history
        print("Segments Received            :", self.received, file=o)
        print("Segments Processed           :", self.received, file=o)
        print("ACKs Generated               :", self.acks_generated, file=o)
        if occ:
            print("Average Buffer Occupancy     :", f"{sum(occ) / len(occ):.2f} KB", file=o)
            print("Maximum Buffer Occupancy     :", f"{max(occ)} KB", file=o)
            print("Average Advertised Window    :", f"{sum(wnd) / len(wnd):.2f} KB", file=o)
        print("Full Buffer Events           :", self.full_events, file=o)
        print(file=o)
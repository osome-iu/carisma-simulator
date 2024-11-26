import time


class Message:
    def __init__(self, mid, content=None):
        self.mid = mid
        self.content = content
        self.timestamp = time.time()

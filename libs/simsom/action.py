class Action:
    def __init__(self, aid: str, uid: str) -> None:
        self.aid = aid
        self.uid = uid

    def write_action(self):
        return (
            self.aid,
            self.uid,
        )

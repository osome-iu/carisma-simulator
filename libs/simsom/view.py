from action import Action


class View(Action):
    def __init__(self, vid: str, uid: str, parent_mid: str, parent_uid: str) -> None:
        Action.__init__(self, vid, uid)
        self.parent_mid = parent_mid
        self.parent_uid = parent_uid

    def write_action(self):
        parent_action = super().write_action()
        return (*parent_action, self.parent_mid, self.parent_uid)

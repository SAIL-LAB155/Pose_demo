

disappear_max = 20
exist_max = 500
alarm_cnt = 400


class Region:
    def __init__(self, idx, h, w):
        self.location = idx
        self.height, self.width = h, w
        self.center = (int((idx[0]+0.5)*w), int((idx[1]+0.5)*h))
        self.exists = 0
        self.disappear = 0
        # self.half = 0

    def clear(self):
        self.exists = 0
        self.disappear = 0

    def update(self, flag):
        if flag == 0:
            pass
        elif flag == 1:
            self.update_exist(flag)
        elif flag == 2:
            self.update_exist(flag)
        elif flag == -1:
            self.update_disappear()
            self.update_exist(flag)
        else:
            raise ValueError("Not a right flag")

    def update_exist(self, f):
        if f == 1:
            if self.exists < exist_max:
                self.exists += 1
        elif f == -1:
            if self.exists > -1:
                self.exists -= 1
        # elif f == 0:
        #     if self.half > 10:
        #         self.exists += 1

    def update_disappear(self):
        if self.disappear < disappear_max:
            self.disappear += 1
        else:
            self.clear()

    def if_warning(self):
        return True if self.exists > alarm_cnt else False

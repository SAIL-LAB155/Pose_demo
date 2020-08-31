
disappear_max = 20
exist_max = 150
alarm_cnt = 100


class Region:
    def __init__(self, idx, h, w):
        self.location = idx
        self.height, self.width = h, w
        self.center = (int((idx[0]+0.5)*w), int((idx[1]+0.5)*h))
        self.top, self.bottom, self.left, self.right = int(idx[1]*h), int((idx[1]+1)*h), int(idx[0]*w), int((idx[0]+1)*w)
        self.exists = 0
        self.disappear = 0
        self.disappearing = True

    def clear(self):
        self.exists = 0
        self.disappear = 0

    def update(self, flag):
        if flag == 0:
            pass
        elif flag == 1:
            self.update_exist(flag)
            self.disappear = 0
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
            if self.exists > 0:
                self.exists -= 1

    def update_disappear(self):
        if self.disappear < disappear_max:
            self.disappear += 1
        else:
            self.clear()

    def if_warning(self):
        return True if self.exists > alarm_cnt else False

    def cnt_color(self):
        if self.exists > alarm_cnt:
            return "red"
        return "gold"

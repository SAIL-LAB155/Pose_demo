from .people import Person


class PeopleAnalyser:
    def __init__(self):
        self.PEOPLE = {}
        self.existing = {}
        self.current = {}

    def update(self, id2box, id2kps, img, img_black):
        for k in id2box.keys():
            if k in self.existing:
                self.PEOPLE[k].BOX.append(id2box[k])
                # if k in id2kps

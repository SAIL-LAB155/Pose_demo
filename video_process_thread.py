#-*- coding: utf-8 -*-
from src.human_detection import HumanDetection
try:
    import src.debug.cohgnfig.cfg as config
except:
    import config.config as config
import cv2

from utils.utils import boxdict2str, kpsdict2str, kpsScoredict2str

body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_dict = {name: idx for idx, name in enumerate(body_parts)}

resize_ratio = config.resize_ratio
show_size = config.show_size
store_size = config.store_size

fourcc = cv2.VideoWriter_fourcc(*'XVID')


from threading import Thread
import time
from queue import Queue
#https://github.com/Kjue/python-opencv-gpu-video


class VideoProcessorThread:
    def __init__(self, path, queueSize=3000):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resize_size = (int(self.width * resize_ratio), int(self.height * resize_ratio))
        self.IP = HumanDetection(self.resize_size)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def update(self):
        # keep looping infinitely
        self.IP.init()
        # IP.object_tracker.init_tracker()
        cnt = 0
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.cap.read()
                start = time.time()
                if grabbed:
                    frame = cv2.resize(frame, self.resize_size)
                    kps, boxes, kps_score = self.IP.process_img(frame)
                    img, img_black = self.IP.visualize()
                    cv2.imshow("res", cv2.resize(img, show_size))
                    cv2.waitKey(2)
                    # out.write(res)
                    all_time = time.time() - start
                    print("time is:", all_time)
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)
            else:
                self.Q.queue.clear()


if __name__ == '__main__':
    #VideoProcessor(config.video_path).process_video()
    VideoProcessorThread(config.video_path).update()

from src.detector.visualize import BBoxVisualizer
from src.tracker.visualize import IDVisualizer
from src.estimator.visualize import KeyPointVisualizer
import cv2
from config.config import video_path, frame_size
from utils.utils import str2boxdict, str2kpsdict, str2kpsScoredict

body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_dict = {name: idx for idx, name in enumerate(body_parts)}


class VideoProcessor:
    def __init__(self, video_path):
        # self.BBV = BBoxVisualizer()
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        with open("/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_box.txt", "r") as bf:
            self.box_txt = [line[:-1] for line in bf.readlines()]
        with open("/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_kps.txt", "r") as kf:
            self.kps_txt = [line[:-1] for line in kf.readlines()]
        with open("/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] +
                  "_kps_score.txt", "r") as ksf:
            self.kps_score_txt = [line[:-1] for line in ksf.readlines()]
        self.IDV = IDVisualizer()
        self.KPV = KeyPointVisualizer()

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                frame = cv2.resize(frame, frame_size)
                id2bbox = str2boxdict(self.box_txt.pop(0))
                id2kps = str2kpsdict(self.kps_txt.pop(0))
                id2kpsScore = str2kpsScoredict(self.kps_score_txt.pop(0))
                if id2bbox is not None:
                    frame = self.IDV.plot_bbox_id(id2bbox, frame, with_bbox=True)
                if id2kps is not None:
                    kps_tensor, score_tensor = self.KPV.kpsdic2tensor(id2kps, id2kpsScore)
                    frame = self.KPV.vis_ske(frame, kps_tensor, score_tensor)
                cv2.imshow("res", frame)
                cv2.waitKey(100)
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    # src_folder = "video/push_up"
    # dest_fodler = src_folder + "kps"
    # sub_folder = [os.path.join(src_folder, folder) for folder in os.listdir(src_folder)]
    # sub_dest_folder = [os.path.join(dest_fodler, folder) for folder in os.listdir(src_folder)]

    VideoProcessor(video_path).process_video()

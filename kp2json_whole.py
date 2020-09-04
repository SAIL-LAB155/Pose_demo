from src.human_detection import HumanDetection
import cv2
import os
import json

HP = HumanDetection()


class ImageDetection:
    def __init__(self, src_folder, dest_folder):
        self.src_img_ls = [os.path.join(src_folder, img_name) for img_name in os.listdir(src_folder)]
        self.dest_img_ls = [os.path.join(dest_folder, img_name) for img_name in os.listdir(src_folder)]
        self.idx = 0
        self.keypoints_json = []
        self.bbox = []
        self.result = {}
        self.result_all = {}
        self.result_all['images'] = []
        self.json = open(src_folder+".json", "w")

        self.result = {}
        self.keypoints = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ]
        self.keypoints_style = [
            "#FF0000",
            "#FF7000",
            "#FFFF00",
            "#99FF00",
            "#00FF00",
            "#00FF99",
            "#00FFFF",
            "#0099FF",
            "#0000FF",
            "#9900FF",
            "#FF00FF",
            "#FF0099",
            "#FFAAAA",
            "#FFCCAA",
            "#FFFFAA",
            "#AAFFAA",
            "#AAFFFF"
        ]
        self.categories = [{
            "id": "0",
            "name": "human",
            "supercategory": "human",
            "keypoints": self.keypoints,
            "keypoints_style": self.keypoints_style
        }]
        self.licenses = []
        self.images = []
        self.annotations = []
        self.id_cnt = 0
        self.result_all["categories"] = self.categories
        self.result_all["licenses"] = self.licenses

    def clear(self):
        pass

    def __process_img(self, img_path, dest_path):
        self.clear()
        HP.init_sort()
        frame = cv2.imread(img_path)
        self.writeImageJson(frame)
        kps, boxes, kps_score = HP.process_img(frame)
        img, img_black = HP.visualize()

        cv2.imwrite(dest_path, img)
        self.writeJson(kps, boxes, kps_score)
        img = cv2.resize(img, (720, 540))
        cv2.imshow("res", img)
        cv2.waitKey(2)
        self.id_cnt += 1

    def writeImageJson(self, image):
        file_name = str(self.src_img_ls[self.idx]).split("/")[-1]
        width, height = image.shape[1], image.shape[0]
        url = "http://localhost:8007/" + file_name
        # # annotation data

        self.images.append({"id": self.id_cnt,
                            "file_name": file_name,
                            "width": width,
                            "height": height,
                            "url": url})

    def writeJson(self, kps_dict, bbox_dict, kpScore_dict):
        keypoints_json, bbox, people_cnt = [], [], 0
        for k in kps_dict.keys():
            box, kps, kpScore = bbox_dict[k], kps_dict[k], kpScore_dict[k]
            if len(box) > 0 and len(kps) > 0:
                for i in range(len(kps)):
                    for j in range(len(kps[0])):
                        keypoints_json.append(kps[i][j].item())
                    if kpScore[i] > 0.3:
                        keypoints_json.append(2)
                    else:
                        keypoints_json.append(0)
                for j in range(4):
                    bbox.append(box[j].item())
                self.annotations.append({"image_id": self.id_cnt,
                                         "category_id": 0,
                                         "bbox": box,
                                         "keypoints": keypoints_json,
                                         "id": people_cnt})
            people_cnt += 1

    def process(self):
        for self.idx, (src, dest) in enumerate(zip(self.src_img_ls, self.dest_img_ls)):
            self.__process_img(src, dest)

        self.result_all["images"] = self.images
        self.result_all["annotations"] = self.annotations
        self.json.write(json.dumps(self.result_all))


if __name__ == '__main__':
    src_folder = r"E:\MODELS\Autoannotation_Pose\Img2json\img\UNDERWATER\0612_selected_mul"
    dest_folder = src_folder + "_kps"
    os.makedirs(dest_folder, exist_ok=True)
    ImageDetection(src_folder, dest_folder).process()


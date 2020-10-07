from config import config
import cv2
from src.human_detection import HumanDetection as ImageProcessor
from utils.utils import Utils

IP = ImageProcessor()

video_path = config.squat_up_side_video_path
body_parts = ["Nose", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
nece_point = [-6, -4, -2]


def run(video_path):
    frm_cnt = 0
    cap = cv2.VideoCapture(video_path)
    count = 0
    up_flag = 0
    count_squat = 0
    count_up = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Video/squat.avi', fourcc, 10, (int(cap.get(3)), int(cap.get(4))))

    while True:
        frm_cnt += 1
        print("cnt {}".format(count))
        print("frame {}".format(frm_cnt))
        ret, frame = cap.read()

        if ret:
            kps_dict, boxes, _ = IP.process_img(frame)
            kps = kps_dict[1]
            img, black_img = IP.visualize()
            if frm_cnt == 170:
                a = 1
            if len(img) > 0 and len(kps) > 0:
                coord = [kps[idx] for idx in nece_point]
                angle = Utils.get_angle(coord[1], coord[0], coord[2])
                if angle > 60:
                    count_squat = 0 if count_squat == 0 else count_squat - 1
                    if count_up < 5:
                        count_up += 1
                    else:
                        count_up = 0
                        up_flag = 1
                else:
                    if up_flag == 1:
                        count_up = 0 if count_up == 0 else count_up - 1
                        if count_squat > 4:
                            count += 1
                            up_flag = 0
                            count_squat = 0
                        else:
                            count_squat += 1
                    else:
                        pass

                cv2.putText(img, "Count: {}".format(count), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                print("The angle is {} degree\n".format(angle))
                out.write(img)
                cv2.imshow("result", img)
                cv2.waitKey(2)

            else:
                cv2.putText(img, "Count: {}".format(count), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                cv2.imshow("result", img)
                cv2.waitKey(2)
        else:
            out.release()
            break
    print("The final count is {}".format(count))


if __name__ == "__main__":
    run(video_path)

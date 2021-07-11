import cv2
import torch


class Counter:
    def __init__(self, video_path, show_image=False):
        self.video_path = video_path
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.show_image = show_image

    def calculate(self):
        self.__read_video()

    def __read_video(self):
        capture = cv2.VideoCapture(self.video_path)

        if not capture.isOpened():
            print("Error opening video stream or file")

        i = 0

        batch = []

        while capture.isOpened():
            ret, frame = capture.read()
            i += 1
            # take only every 8th frame
            if i % 8 != 0:
                continue

            if ret:
                if self.show_image:
                    cv2.imshow("frame", frame)
                batch.append(frame)
                # batch in count of 8
                if len(batch) < 8:
                    continue

                results = self.model(batch, size=320)
                results.print()
                batch = []
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        capture.release()
        cv2.destroyAllWindows()

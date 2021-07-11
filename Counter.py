from typing import List

import cv2
import norfair
import numpy as np
import torch
from norfair import Detection, Tracker

max_distance_between_points: int = 100


class Counter:
    def __init__(self, video_path, show_image=False):
        self.video_path = video_path
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.show_image = show_image
        self.tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=max_distance_between_points,
        )

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
                batch.append(frame)
                # batch in count of 8
                if len(batch) < 8:
                    continue

                results = self.model(batch, size=320)
                for i in range(0, 7):
                    detections = self.yolo_detections_to_norfair_detections(results, i)
                    tracked_objects = self.tracker.update(detections=detections)
                    norfair.print_objects_as_table(tracked_objects)
                results.print()
                if self.show_image:
                    cv2.imshow("frame", frame)
                batch = []
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        capture.release()
        cv2.destroyAllWindows()

    def yolo_detections_to_norfair_detections(self,
                                              yolo_detections: torch.tensor,
                                              i: int,
                                              track_points: str = 'centroid',  # bbox or centroid
                                              ) -> List[Detection]:
        """convert detections_as_xywh to norfair detections
        """
        norfair_detections: List[Detection] = []

        if track_points == 'centroid':
            detections_as_xywh = yolo_detections.xywh[i]
            for detection_as_xywh in detections_as_xywh:
                centroid = np.array(
                    [
                        detection_as_xywh[0].item(),
                        detection_as_xywh[1].item()
                    ]
                )
                scores = np.array([detection_as_xywh[4].item()])
                norfair_detections.append(
                    Detection(points=centroid, scores=scores)
                )
        elif track_points == 'bbox':
            detections_as_xyxy = yolo_detections.xyxy[i]
            for detection_as_xyxy in detections_as_xyxy:
                bbox = np.array(
                    [
                        [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                        [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                    ]
                )
                scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
                norfair_detections.append(
                    Detection(points=bbox, scores=scores)
                )

        return norfair_detections

    @staticmethod
    def euclidean_distance(detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

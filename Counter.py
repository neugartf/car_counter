from typing import List

import cv2
import norfair
import numpy as np
import torch
from norfair import Detection, Tracker

max_distance_between_points: int = 100


class Counter:
    class_map = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, video_path, show_image=False):
        self.video_path = video_path
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
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

        counter = {}
        totals = {}

        while capture.isOpened():
            ret, frame = capture.read()
            i += 1
            # take only every 2th frame
            #if i % 2 != 0:
             #   continue

            if ret:
                batch.append(frame)
                # batch in count of 8
                if len(batch) < 8:
                    continue

                results = self.model(batch, size=320)
                detections = self.yolo_detections_to_norfair_detections(results.tolist())

                for detection in detections:
                    tracked_objects = self.tracker.update(detections=detection)
                    for tracked_object in tracked_objects:
                        if tracked_object.id in counter:
                            if counter[tracked_object.id][1] < tracked_object.last_detection.scores[0]:
                                counter[tracked_object.id] = (tracked_object.last_detection.data, tracked_object
                                                              .last_detection
                                                              .scores[0])
                        else:
                            counter[tracked_object.id] = (tracked_object.last_detection.data, tracked_object
                                                          .last_detection
                                                          .scores[0])

                    print(totals)
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

        for key, value in counter.items():
            totals[value[0]] = totals.get(value[0], 0) + 1
        for key in totals:
            print(self.class_map[key] + ":" + str(totals[key]))

    @staticmethod
    def yolo_detections_to_norfair_detections(yolo_detections: []) -> List[List[Detection]]:
        """convert detections_as_xywh to norfair detections
        """
        norfair_detections: List[List[Detection]] = [[]]

        for yolo_detection in yolo_detections:
            tmp: List[Detection] = []
            for xywh in yolo_detection.xywh:
                centroid = np.array(
                    [
                        xywh[0].item(),
                        xywh[1].item()
                    ]
                )
                scores = np.array([xywh[4].item()])
                tmp.append(Detection(points=centroid, scores=scores, data=xywh[5].item()))
            norfair_detections.append(tmp)

        return norfair_detections

    @staticmethod
    def euclidean_distance(detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

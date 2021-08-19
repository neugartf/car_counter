import datetime
import time
from typing import List
import logging
import cv2
import norfair
import numpy as np
import pandas as pd
import torch
from dateutil.parser import parse
from ffprobe import FFProbe
from imutils.video import FileVideoStream
from norfair import Detection, Tracker, draw_tracked_objects
from tqdm import tqdm

max_distance_between_points: int = 200


class Counter:
    class_map = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    counter = {"bicycles": set(), "cars": set(), "motorcycles": set(), "buses": set(), "trucks": set()}
    log = logging.getLogger(__name__)

    current_pos_in_ms = 0

    def __init__(self, video_path, show_image=False):
        self.log.setLevel(logging.ERROR)
        self.video_path = video_path
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.conf = 0.5
        self.model.classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.show_image = show_image
        ffprobe = FFProbe(self.video_path)
        self.creation_time = (parse(ffprobe.metadata['creation_time']))
        self.car_tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=max_distance_between_points,
        )
        self.bus_tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=max_distance_between_points,
        )
        self.bicycle_tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=max_distance_between_points,
        )
        self.motorcycle_tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=max_distance_between_points
        )
        self.truck_tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=max_distance_between_points,
            initialization_delay=0
        )

    def calculate(self):
        self.__read_video()

    def __read_video(self):
        capture = FileVideoStream(self.video_path).start()
        time.sleep(1.0)

        batch = []

        frame0 = None

        frame_length = int(capture.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        pbar = tqdm(total=frame_length, position=0)
        while capture.more():
            frame = capture.read()
            self.current_pos_in_ms = capture.stream.get(cv2.CAP_PROP_POS_MSEC)
            frame_count += 1
            pbar.update()
            self.log.info("Frame %s/%s" % (frame_count, frame_length))
            if frame is None:
                self.log.error("img0 is None")
                continue
            frame = frame[:, :-175]

            if frame is not None and frame0 is not None:
                diff = np.sum(np.absolute(frame - frame0)) / np.size(frame)
                self.log.debug("difference:" + str(diff))
                if diff > 68.:
                    self.log.info('change')
                else:
                    self.log.info('no change')
                    continue

            img = frame[..., ::-1]
            # batch in frame_count of 8
            # if len(batch) < 8:
            #    continue

            # Inference
            t1 = time.time()
            results = self.model(img, 320)
            t2 = time.time()

            # display images with box
            rendered_imgs = results.render()

            if results.n > 0 and results.pred[0].numel() != 0:
                self.log.info(results.pandas().xywh[0])

                tracked_objects = self.yolo_detections_to_norfair_detections(results.xywh[0])
                draw_tracked_objects(rendered_imgs[0], tracked_objects)
            #    if len(tracked_objects):
            #        norfair.print_objects_as_table(tracked_objects)

            for rendered_img in rendered_imgs:
                cv2.imshow("frame", rendered_img)
            if self.show_image:
                for i, det in enumerate(results):
                    for *xyxy, conf, cls in det:
                        # Add bbox to image
                        plot_one_box(xyxy, frame[0], line_thickness=3)
                for frame in batch:
                    cv2.imshow("frame", frame)
            batch = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame0 = frame
        pd.DataFrame.from_dict(data=self.counter, orient='index').transpose().to_csv('dict_file.csv', header=True,
                                                                                     index=True)
        pbar.close()
        capture.stop()
        cv2.destroyAllWindows()

    def yolo_detections_to_norfair_detections(self, yolo_detections) -> List[Detection]:
        """convert detections_as_xywh to norfair detections
        """
        tracked_bikes = tracked_buses = tracked_motos = tracked_trucks = tracked_cars = list()
        norfair_detections: List[Detection] = []
        for xywh in yolo_detections:
            centroid = np.array(
                [
                    xywh[0].item(),
                    xywh[1].item()
                ]
            )
            scores = np.array([xywh[4].item()])
            norfair_detections.append(norfair.Detection(points=centroid, scores=scores, data=xywh[5].item()))

        iterator = filter(self.is_bike, norfair_detections)
        filtered = list(iterator)
        if len(filtered):
            tracked_bikes = self.bicycle_tracker.update(detections=filtered)

        iterator = filter(self.is_car, norfair_detections)
        filtered = list(iterator)
        if len(filtered):
            tracked_cars = self.car_tracker.update(detections=filtered)

        iterator = filter(self.is_moto, norfair_detections)
        filtered = list(iterator)
        if len(filtered):
            tracked_motos = self.motorcycle_tracker.update(detections=filtered)

        iterator = filter(self.is_bus, norfair_detections)
        filtered = list(iterator)
        if len(filtered):
            tracked_buses = self.bus_tracker.update(detections=filtered)

        iterator = filter(self.is_truck, norfair_detections)
        filtered = list(iterator)
        if len(filtered):
            tracked_trucks = self.truck_tracker.update(detections=filtered)

        self.counter["bicycles"] = self.counter["bicycles"] | set(map(
            lambda tracked_bike: MyDetection(tracked_bike.id,
                                             self.creation_time + datetime.timedelta(
                                                 milliseconds=self.current_pos_in_ms)), tracked_bikes))
        self.counter["buses"] = self.counter["buses"] | set(
            map(lambda tracked_bus: MyDetection(tracked_bus.id, self.creation_time + datetime.timedelta(
                milliseconds=self.current_pos_in_ms)),
                tracked_buses))
        self.counter["motorcycles"] = self.counter["motorcycles"] | set(
            map(lambda tracked_motorcycle: MyDetection(tracked_motorcycle.id, self.creation_time + datetime.timedelta(
                milliseconds=self.current_pos_in_ms)),
                tracked_motos))
        self.counter["trucks"] = self.counter["trucks"] | set(
            map(lambda tracked_truck: MyDetection(tracked_truck.id, self.creation_time + datetime.timedelta(
                milliseconds=self.current_pos_in_ms)),
                tracked_trucks))
        self.counter["cars"] = self.counter["cars"] | set(map(lambda tracked_car: MyDetection(tracked_car.id,
                                                                                              self.creation_time + datetime.timedelta(
                                                                                                  milliseconds=self.current_pos_in_ms)),
                                                              tracked_cars))

        self.log.debug(self.counter)
        return tracked_bikes + tracked_cars + tracked_motos + tracked_buses + tracked_trucks

    @staticmethod
    def is_bike(detection: Detection):
        return int(detection.data) == 1

    @staticmethod
    def is_car(detection: Detection):
        return int(detection.data) == 2

    @staticmethod
    def is_moto(detection: Detection):
        return int(detection.data) == 3

    @staticmethod
    def is_bus(detection: Detection):
        return int(detection.data) == 5

    @staticmethod
    def is_truck(detection: Detection):
        return int(detection.data) == 7

    @staticmethod
    def euclidean_distance(detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class MyDetection:
    def __init__(self, _id, detection_ts):
        self._id = _id
        self.detection_ts = detection_ts

    def __eq__(self, obj):
        return isinstance(obj, MyDetection) and obj._id == self._id

    def __hash__(self):
        return hash(self._id)

    def __repr__(self): return str(self._id) + ":" + str(self.detection_ts)

import datetime
import logging
import time
from datetime import timedelta
from itertools import chain
from typing import List

import cv2
import norfair
import numpy as np
import pandas as pd
import torch
import yaml
from dateutil.parser import parse
from ffprobe import FFProbe
from imutils.video import FileVideoStream
from norfair import Detection, Tracker, draw_tracked_objects
from ssh_pymongo import MongoSession
from tqdm import tqdm

max_distance_between_points: int = 200


class Counter:
    class_map = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    log = logging.getLogger(__name__)

    def __init__(self, video_path, show_image=False):
        with open("config.yaml", "r") as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
        self.log.setLevel(logging.ERROR)
        self.video_path = video_path
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.conf = 0.5
        self.model.classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.show_image = show_image
        self.counter = {"bicycles": set(), "cars": set(), "motorcycles": set(), "buses": set(), "trucks": set()}
        ffprobe = FFProbe(self.video_path)
        self.creation_time = parse(ffprobe.metadata['creation_time'])
        self.current_pos_in_ms = 0
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

        frames = int(capture.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(capture.stream.get(cv2.CAP_PROP_FPS))
        self.duration_in_ms = (frames / fps) * 1000
        batch = []

        frame0 = None

        frame_count = 0
        pbar = tqdm(total=frames, position=0)
        while capture.more():
            frame = capture.read()
            frame_count += 1
            if frame is None:
                self.log.error("img0 is None")
                break

            self.current_pos_in_ms = (frame_count / fps) * 1000
            pbar.update()
            self.log.info("Frame %s/%s" % (frame_count, frames))
            if frame is not None and frame0 is not None:
                diff = np.sum(np.absolute(frame - frame0)) / np.size(frame)
                self.log.debug("difference:" + str(diff))
                if diff > 68.:
                    self.log.info('change')
                else:
                    self.log.info('no change')
                    continue

            img = frame[..., ::-1]

            # Inference
            results = self.model(img, 320)

            # display images with box
            rendered_imgs = results.render()

            if results.n > 0 and results.pred[0].numel() != 0:
                self.log.info(results.pandas().xywh[0])

                tracked_objects = self.yolo_detections_to_norfair_detections(results.xywh[0])
                draw_tracked_objects(rendered_imgs[0], tracked_objects)

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

        self.write_to_db()

        pbar.close()
        capture.stop()
        cv2.destroyAllWindows()

    def write_to_db(self):
        creation_time_masked = self.floor_datetime_to_minutes(self.creation_time)
        logging.info(creation_time_masked)
        video_length_in_min = int(self.duration_in_ms / 1000 / 60)
        logging.info(video_length_in_min)
        dti = pd.date_range(creation_time_masked, periods=video_length_in_min + 1, freq="1min")
        df = pd.DataFrame(index=dti, columns=["bicycles", "cars", "motorcycles", "buses", "trucks"])
        df = df.fillna(0)

        counter_as_panda = pd.DataFrame.from_dict(data=self.counter, orient='index')

        for index, row in counter_as_panda.iterrows():
            for occurrence in row.dropna():
                logging.info("detection ts:" + str(occurrence.detection_ts))
                datetime_masked = self.floor_datetime_to_minutes(occurrence.detection_ts)
                df.loc[datetime_masked][index] += 1

        data = df.to_dict("index")

        session = MongoSession(self.cfg["mongodb"]["host"], user=self.cfg["mongodb"]["user"],
                               key_password=self.cfg["mongodb"]["password"],
                               key=self.cfg["mongodb"]["key_path"])

        db = session.connection["vehicleCounterDB"]
        logging.info("Connected to mongodb")
        vehicle_collection = db.get_collection("vehicles")
        for key, value in tqdm(data.items()):
            vehicle_collection.insert_one({"timestamp": key, "count": value})
        logging.info("Finished writing to mongodb")
        logging.info("Wrote " + str(len(data.items())) + " data sets")
        session.stop()

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

        tracked_vehicles = dict(
            zip(list(self.counter.keys()), [tracked_bikes, tracked_cars, tracked_motos, tracked_buses, tracked_trucks]))

        for value in tracked_vehicles.items():
            self.update_counter(value)

        self.log.debug(self.counter)
        return list(chain(*list(tracked_vehicles.values())))

    def update_counter(self, tracked_vehicles):
        key, value = tracked_vehicles
        ms_ = self.creation_time + datetime.timedelta(milliseconds=self.current_pos_in_ms)

        self.counter[key] = self.counter[key] | set(map(
            lambda tracked_vehicle: MyDetection(tracked_vehicle.id,
                                                ms_), value))

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

    @staticmethod
    def floor_datetime_to_minutes(datetime: datetime):
        logging.warning("floor datetime to minutes" + str(datetime))
        return np.datetime64(datetime - timedelta(minutes=datetime.minute % 1,
                                                  seconds=datetime.second,
                                                  microseconds=datetime.microsecond), 'ns')


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

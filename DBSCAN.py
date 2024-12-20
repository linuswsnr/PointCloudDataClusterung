import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

"""
# odometry
timestamp, x_seq, y_seq, yaw_seq, vx, yaw_rate
"""

"""
# radar_data
print(f"Form des Datasets: {radar_dataset.shape}")
print(f"Datentyp des Datasets: {radar_dataset.dtype}")
* timestamp: in micro seconds relative to some arbitrary origin
* sensor_id: integer value, id of the sensor that recorded the detection
* range_sc: in meters, radial distance to the detection, sensor coordinate system
* azimuth_sc: in radians, azimuth angle to the detection, sensor coordinate system
* rcs: in dBsm, radar cross section (RCS value) of the detection
* vr: in m/s. Radial velocity measured for this detection
* vr_compensated in m/s: Radial velocity for this detection but compensated for the ego-motion
* x_cc and y_cc: in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
* x_seq and y_seq in m, position of the detection in the global sequence-coordinate system (origin is at arbitrary start point)
* uuid: unique identifier for the detection. Can be used for association with predicted labels and for debugging
* track_id: id of the dynamic object this detection belongs to. Empty, if it does not belong to any.
* label_id: semantic class id of the object to which this detection belongs. 
            passenger cars (0), 
            large vehicles (like agricultural or construction vehicles) (1), 
            trucks (2), 
            busses (3), 
            trains (4), 
            bicycles (5), 
            motorized two-wheeler (6), 
            pedestrians (7), 
            groups of pedestrian (8), 
            animals (9), 
            all other dynamic objects encountered while driving (10), 
            and the static environment (11)
"""
dataset_path = r"C:\Users\roesc\Hochschule\09_Semester\Sensorik\RadarScenes\data"
sequence_nr = 7

sequence_folder = rf"{dataset_path}\sequence_{sequence_nr}"
# img_folder = rf"{sequence_folder}\camera"
hdf_file = rf"{sequence_folder}\radar_data.h5"
json_file = rf"{sequence_folder}\scenes.json"

with h5py.File(hdf_file, "r") as f:
    radar_dataset = f["radar_data"][:]
    odometry_dataset = f["odometry"][:]

with open(json_file, 'r', encoding='utf-8') as file:
    scenes = json.load(file)["scenes"]

start, end = None, None
sensor_1, sensor_2, sensor_3, sensor_4 = [], [], [], []

epsilon = 0.5
samples = 1
class_id = 0


def dbscan_labels(X):
    db_scan = DBSCAN(eps=epsilon, min_samples=samples)
    db_scan_labels = db_scan.fit_predict(X)

    return db_scan_labels


for timestamp in scenes:
    scene = scenes[timestamp]
    sensor_id = scene["sensor_id"]
    start, end = scene["radar_indices"][0], scene["radar_indices"][1]
    radar_data = radar_dataset[start:end]
    data = radar_data[radar_data["label_id"] == class_id]
    x, y = data["x_cc"], data["y_cc"]

    if len(data) == 0:
        continue

    if sensor_id == 1:
        sensor_1 = [x, y]
    elif sensor_id == 2:
        sensor_2 = [x, y]
    elif sensor_id == 3:
        sensor_3 = [x, y]
    elif sensor_id == 4:
        sensor_4 = [x, y]

    if len(sensor_1) > 0 and len(sensor_2) > 0 and len(sensor_3) > 0 and len(sensor_4):
        x_all_sensors = list(sensor_1[0]) + list(sensor_2[0]) + list(sensor_3[0]) + list(sensor_4[0])
        y_all_sensors = list(sensor_1[1]) + list(sensor_2[1]) + list(sensor_3[1]) + list(sensor_4[1])

        sensor_1, sensor_2, sensor_3, sensor_4 = [], [], [], []

        X = np.array(list(zip(x_all_sensors, y_all_sensors)))

        labels = dbscan_labels(X)

        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.show()

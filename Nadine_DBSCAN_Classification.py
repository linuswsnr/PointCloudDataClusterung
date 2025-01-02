import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
classes = ["passenger cars", "large vehicles", "trucks", "busses", "trains", "bicycles", "motorized two-wheeler",
           "pedestrians", "groups of pedestrian", "animals"]

# epsilons = [1, 2, 2, 1.5, 1, 0.5, 0.8, 0.5, 0.5]

X_total, Y_total = np.array([]), np.array([])


def nearest_neighbors(values, seq, c_id):
    k = 2
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(values)
    distances, indices = neighbors_fit.kneighbors(values)

    distances = np.sort(distances[:, k - 1], axis=0)

    plt.plot(distances)
    plt.title(f"Sequence {seq}, class: {classes[c_id]}")
    plt.xlabel("Datenpunkte sortiert nach Entfernung")
    plt.ylabel("Abstand zum k-ten nächsten Nachbarn")
    plt.show()


def dbscan_labels(values, seq, c_id, plot):
    db_scan = DBSCAN(eps=4, min_samples=1)
    db_scan_labels = db_scan.fit_predict(values)

    if plot:
        plt.scatter(values[:, 0], values[:, 1], c=db_scan_labels)
        plt.xlabel("x_cc [m]")
        plt.ylabel("y_cc [m]")
        plt.xticks(range(-50, 51, 10))
        plt.yticks(range(-50, 51, 10))
        plt.title(f"Sequence {seq}, class: {classes[c_id]}")
        plt.show()

    return db_scan_labels


def dbscan_clustering(class_id):
    global X_total, Y_total

    for sequence_nr in range(1, 159):
        print("Sequence", sequence_nr)
        sequence_folder = rf"{dataset_path}\sequence_{sequence_nr}"
        hdf_file = rf"{sequence_folder}\radar_data.h5"
        json_file = rf"{sequence_folder}\scenes.json"

        with h5py.File(hdf_file, "r") as f:
            radar_dataset = f["radar_data"][:]
            # odometry_dataset = f["odometry"][:]

        if class_id not in np.unique(radar_dataset["label_id"]):
            print("Does not contain class", classes[class_id])
            continue

        with open(json_file, 'r', encoding='utf-8') as file:
            scenes = json.load(file)["scenes"]

        sensor_1, sensor_2, sensor_3, sensor_4 = [], [], [], []

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

                # nearest_neighbors(X, sequence_nr, class_id)
                labels = dbscan_labels(X, sequence_nr, class_id, False)

                X_extended = np.hstack((X, labels.reshape(-1, 1)))
                Y = np.array([class_id] * len(labels))

                if len(X_total) == 0:
                    X_total = X_extended
                    Y_total = Y
                else:
                    X_total = np.concatenate((X_total, X_extended))
                    Y_total = np.concatenate((Y_total, Y))

            # plt.savefig(f"C:\\Users\\roesc\\Hochschule\\09_Semester\\Sensorik\\Ergebnisse\\image_{img_nr}.png")
            # plt.close()


def classification():
    X_train, X_test, y_train, y_test = train_test_split(X_total, Y_total, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


dbscan_clustering(1)
classification()

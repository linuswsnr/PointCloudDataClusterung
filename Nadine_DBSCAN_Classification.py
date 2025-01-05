import h5py
import json
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier

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

seq_train = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34,
             35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 57, 59, 60, 61, 62, 64, 65,
             66, 67, 69, 70, 71, 72, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97,
             98, 100, 101, 102, 103, 104, 105, 106, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
             123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146,
             149, 150, 151, 152, 154, 156, 157, 158]
seq_val = [5, 6, 14, 19, 24, 31, 42, 48, 53, 58, 63, 68, 73, 79, 85, 89, 93, 99, 107, 111, 122, 130, 135, 138, 147,
           148, 153, 155]

X_train, y_train = np.array([]), np.array([])
X_val, Y_val = np.array([]), np.array([])


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


def dbscan_clustering(class_id, train):
    global X_train, y_train, X_val, Y_val

    if train:
        sequences = seq_train
    else:
        sequences = seq_val

    for sequence_nr in sequences:
        print("Sequence", sequence_nr)
        sequence_folder = rf"{dataset_path}\sequence_{sequence_nr}"
        hdf_file = rf"{sequence_folder}\radar_data.h5"
        json_file = rf"{sequence_folder}\scenes.json"

        with h5py.File(hdf_file, "r") as f:
            radar_dataset = f["radar_data"][:]
            # odometry_dataset = f["odometry"][:]

        if class_id not in np.unique(radar_dataset["label_id"]):
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

                if train:
                    if len(X_train) == 0:
                        X_train = X_extended
                        y_train = Y
                    else:
                        X_train = np.concatenate((X_train, X_extended))
                        y_train = np.concatenate((y_train, Y))
                else:
                    if len(X_val) == 0:
                        X_val = X_extended
                        Y_val = Y
                    else:
                        X_val = np.concatenate((X_val, X_extended))
                        Y_val = np.concatenate((Y_val, Y))

            # plt.savefig(f"C:\\Users\\roesc\\Hochschule\\09_Semester\\Sensorik\\Ergebnisse\\image_{img_nr}.png")
            # plt.close()


def classification():
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    dump(clf, 'C:\\Users\\roesc\\Hochschule\\09_Semester\\Sensorik\\random_forest_model.joblib')


def prediction():
    clf = load('C:\\Users\\roesc\\Hochschule\\09_Semester\\Sensorik\\random_forest_model.joblib')

    y_pred = clf.predict(X_val)

    print("Accuracy:", accuracy_score(Y_val, y_pred))


def test(seq, c_id):
    sequence_folder = rf"{dataset_path}\sequence_{seq}"
    hdf_file = rf"{sequence_folder}\radar_data.h5"
    json_file = rf"{sequence_folder}\scenes.json"

    with h5py.File(hdf_file, "r") as f:
        radar_dataset = f["radar_data"][:]

    with open(json_file, 'r', encoding='utf-8') as file:
        scenes = json.load(file)["scenes"]

    sensor_1, sensor_2, sensor_3, sensor_4 = [], [], [], []

    for timestamp in scenes:
        scene = scenes[timestamp]
        sensor_id = scene["sensor_id"]
        start, end = scene["radar_indices"][0], scene["radar_indices"][1]
        radar_data = radar_dataset[start:end]

        data = radar_data[radar_data["label_id"] == c_id]
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
            labels = dbscan_labels(X, seq, c_id, False)

            X_extended = np.hstack((X, labels.reshape(-1, 1)))
            Y = np.array([c_id] * len(labels))

            y_pred = clf.predict(X_extended)

            print("Ground Truth", Y)
            print("Prediction", y_pred)
            print("Accuracy:", accuracy_score(Y, y_pred))
            break


"""
for i in range(10):
    print("Class", i)

    dbscan_clustering(i, train=True)
    dbscan_clustering(i, train=False)
"""

# classification()
# prediction()

clf = load('C:\\Users\\roesc\\Hochschule\\09_Semester\\Sensorik\\random_forest_model.joblib')
test_seq = [7, 8, 7, 10, 2, 7, 10, 7, 7, 97]

for c in range(10):
    print(classes[c])
    test(seq=test_seq[c], c_id=c)

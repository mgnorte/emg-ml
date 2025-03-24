import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from bagpy import bagreader
import re
import pywt
#from nateyFunction import *
from scipy import ndimage
from skimage.feature import peak_local_max
from mpl_toolkits.mplot3d import Axes3D

def plotAllEmg(subset2, emgnames):
  n_rows, n_cols = 2, 5
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 8))
  fig.tight_layout(pad=3.0)
  axes = axes.T.flatten()
  for ax, emg_name in zip(axes.flatten(), emgnames):
      ax.plot(subset2.index.astype(np.int64), subset2[emg_name], label=emg_name)
      ax.set_title(emg_name)
      ax.legend()
  plt.tight_layout()
  plt.show()

def openEmgCsv(filename):
  df = pd.read_csv(filename)
  csvdata = []
  csvtime = df.iloc[:, 0].values
  for i in range(1, len(df.iloc[0])):
    csvdata.append(df.iloc[:, i].values)

  return csvdata, csvtime
def parse_position_orientation(row):
    position_matches = re.findall(r'position: \n  x: (.*?)\n  y: (.*?)\n  z: (.*?)\n', row['poses'])
    orientation_matches = re.findall(r'orientation: \n  x: (.*?)\n  y: (.*?)\n  z: (.*?)\n  w: (.*?)[,\]]', row['poses'])
    if position_matches and orientation_matches:
        positions = [tuple(float(i) for i in match) for match in position_matches]
        orientations = [tuple(float(i) for i in match) for match in orientation_matches]
        return np.array(positions), np.array(orientations)
    else:
        return None, None

def get_skl_data(data):
    skl_positions, skl_orientations = np.zeros((len(data), 51, 3)), np.zeros((len(data), 51, 4))
    skl_time = np.array(data['Time'])
    for t in range(len(skl_time)):
        skl_positions[t], skl_orientations[t] = parse_position_orientation(data.iloc[t])
    return skl_time, skl_positions, skl_orientations

#def extract_and_interpolate():
    b = bagreader("moremuscles1/moremuscles1.bag")
    skl_data = pd.read_csv(b.message_by_topic('/natnet_ros/fullbody/pose'))
    skl_time, skl_positions, skl_orientations = get_skl_data(skl_data)

    emgdata, emgtime = openEmgCsv('moremuscles1/moremuscles1.csv')

    difference = emgtime[-1] - skl_time[-1] + 0.712006037
    skl_time += difference

    temgdata = []

    # syncing time and interpolating
    cutoff = skl_time[0]
    cutindex = 0
    for i in range(0, len(emgtime)):
        if emgtime[i] > cutoff:
            cutindex = i
            break

    newemgtime = emgtime[cutindex:]

    for i in range(0, len(emgdata)):
        temgdata.append(emgdata[i][cutindex:])

    newemgdata = np.array(temgdata).T
    newemgtime = np.array(newemgtime)

    newskl_time = np.array(skl_time)
    newskl_positions = np.array(skl_positions[:, 41, :])

    xyz_df = pd.DataFrame(newskl_positions, columns=['x', 'y', 'z'], index=pd.to_datetime(newskl_time, unit='s'))
    emg_df = pd.DataFrame(newemgdata, columns=[f'emg_{i+1}' for i in range(10)], index=pd.to_datetime(newemgtime, unit='s'))

    common_index = xyz_df.index.union(emg_df.index)
    xyz_df = xyz_df.reindex(common_index).interpolate(method='time')
    emg_df = emg_df.reindex(common_index).interpolate(method='time')

    merged_data = pd.concat([xyz_df, emg_df], axis=1)

    start = pd.Timestamp('1970-01-01 00:01:20.651037931')
    end = pd.Timestamp("1970-01-01 00:01:40.747028589")
    subset = merged_data[(merged_data.index >= start) & (merged_data.index <= end)]

    emgnames = ["emg_1", "emg_2", "emg_3", "emg_4", "emg_5", "emg_6", "emg_7", "emg_8", "emg_9", "emg_10"]

    color3D(subset)
    lmt = getRepStart(subset, "z", "x")
    cutData = plotEmg3D(subset, lmt, True, emgnames)

    return subset, emgnames

def importData(filename):
  df = pd.read_csv(filename)
  df.index = pd.to_datetime(df.index)
  df = df.drop(columns=['Unnamed: 0'])
  return df

def get_windows(signal, windowSize):

    signalLength = len(signal)
    segments = []

    for i in range(signalLength - windowSize):
        segment = (i, i + windowSize)
        segments.append(segment)

    return segments

#Time-dependant factors
def mean_absolute_value(data):
    return np.mean(np.abs(data))

def root_mean_square(data):
    return np.sqrt(np.mean(np.square(data)))

def zero_crossings(data):
    differences = np.diff(np.sign(data))

    zero_crossings = np.sum(differences != 0)

    return zero_crossings

def slope_sign_changes(data):

    slope = np.diff(data)

    slope_sign = np.sign(slope)

    sign_changes = np.diff(slope_sign)

    # Count the number of sign changes
    num_sign_changes = np.sum(sign_changes != 0)

    return num_sign_changes


def calculate_waveform_length(signal):
    # Compute the absolute differences between consecutive samples
    differences = np.abs(np.diff(signal))

    # Sum the absolute differences to get the waveform length
    waveform_length = np.sum(differences)

    return waveform_length

def calculate_integral_absolute_value(data):
    return np.sum(data)

def find_local_max_messy(data):
    data[data < 0] = 0
    data = data ** 2

    # Step 1: Apply Gaussian filter for smoothing
    sigma = 1
    Z_smooth = ndimage.gaussian_filter(data, sigma)

    # Step 2: Find local maxima
    min_distance = 1
    threshold_abs = 500000
    coordinates = peak_local_max(Z_smooth, min_distance=min_distance, threshold_abs=threshold_abs, exclude_border=False, num_peaks=5)

    """
    # Print peak locations
    print(len(coordinates))
    print("Approximate local maxima found at:")
    for coord in coordinates:
        print(coord[0], coord[1])

    
    plt.subplot(1,2,1)
    plt.imshow(np.abs(Z_smooth), aspect='auto',
               cmap='jet')
    plt.subplot(1,2,2)
    plt.imshow(np.abs(data), aspect='auto',
               cmap='jet')
    plt.show()
    """

    return coordinates

def find_local_min_messy(data):


    data[data > 0] = 0

    data = data ** 2

    # Step 1: Apply Gaussian filter for smoothing
    sigma = 1
    Z_smooth = ndimage.gaussian_filter(data, sigma)

    # Step 2: Find local maxima
    min_distance = 1
    threshold_abs = np.max(Z_smooth)/10
    coordinates = peak_local_max(Z_smooth, min_distance=min_distance, threshold_abs=threshold_abs, exclude_border=False, num_peaks=5)

    """
    # Print peak locations
    print(len(coordinates))
    print("Approximate local minima found at:")
    for coord in coordinates:
        print(coord[0], coord[1])

    
    plt.subplot(1,2,1)
    plt.imshow(np.abs(Z_smooth), aspect='auto',
               cmap='jet')
    plt.subplot(1,2,2)
    plt.imshow(np.abs(data), aspect='auto',
               cmap='jet')
    plt.show()
    """

    return coordinates

def find_local_maxima(arr):
    arr = abs(arr)

    # Get the shape of the array
    rows, cols = arr.shape

    # Create a boolean mask for local maxima
    local_max = np.zeros_like(arr, dtype=bool)

    # Iterate through the array, excluding the borders
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Check if the current element is greater than all its neighbors
            if arr[i, j] > max(arr[i - 1:i + 2, j - 1:j + 2].flatten()):
                local_max[i, j] = True

    print(np.where(local_max))
    plt.imshow(arr, aspect='auto',
               cmap='jet')
    plt.show()
    # Return the coordinates of local maxima
    return np.where(local_max)

def cwt_spikes(signalData):

    freqResolution = 1
    scales = np.arange(1, 40, freqResolution)

    coeffs, _ = pywt.cwt(signalData, scales, 'mexh')
    coeffs2 = coeffs.copy()

    coeffMaxima = find_local_max_messy(coeffs)
    maxCoordinates = []
    for i in range(5):
        if i < len(coeffMaxima):
            t = coeffMaxima[i][0]
            f = coeffMaxima[i][1]
            maxCoordinates.append([20, coeffs[t][f], t, f*freqResolution])
        else:
            maxCoordinates.append([-20, 0, 0, 0])

    coeffMinima = find_local_min_messy(coeffs2)
    minCoordinates = []
    for i in range(5):
        if i < len(coeffMinima):
            t = coeffMinima[i][0]
            f = coeffMinima[i][1]
            minCoordinates.append([20, coeffs2[t][f], t, f*freqResolution])
        else:
            minCoordinates.append([-20, 0, 0, 0])

    return maxCoordinates, minCoordinates

def collect_features_2(segments):

    MAVs = []
    RMSs = []
    ZCs = []
    SSCs = []
    WLs = []
    IAVs = []
    wavelet = [[] for i in range(40)]

    for segmentData in segments:

        MAV = mean_absolute_value(segmentData)
        MAVs.append(MAV)
        RMS = root_mean_square(segmentData)
        RMSs.append(RMS)
        ZC = zero_crossings(segmentData)
        ZCs.append(ZC)
        SSC = slope_sign_changes(segmentData)
        SSCs.append(SSC)
        WL = calculate_waveform_length(segmentData)
        WLs.append(WL)
        IAV = calculate_integral_absolute_value(segmentData)
        IAVs.append(IAV)

        #list of coordinates (5 lists of length 4)
        waveletMaxes, waveletMins = cwt_spikes(segmentData)
        for n, coord in enumerate(waveletMaxes):
            for i, Xi in enumerate(coord):
                wavelet[n*4 + i].append(Xi * 45)
        for n, coord in enumerate(waveletMins):
            for i, Xi in enumerate(coord):
                wavelet[20 + n*4 + i].append(Xi * 45)

    features = [MAVs, RMSs, ZCs, SSCs, WLs, IAVs]
    features.extend(wavelet)

    #[46, N]
    return features

def collect_features(signal, segments):

    MAVs = []
    RMSs = []
    ZCs = []
    SSCs = []
    WLs = []
    IAVs = []
    wavelet = [[] for i in range(40)]

    for segment in segments:

        segmentData = signal[segment[0]:segment[1]]
        if len(segmentData) == 0:
            continue

        MAV = mean_absolute_value(segmentData)
        MAVs.append(MAV)
        RMS = root_mean_square(segmentData)
        RMSs.append(RMS)
        ZC = zero_crossings(segmentData)
        ZCs.append(ZC)
        SSC = slope_sign_changes(segmentData)
        SSCs.append(SSC)
        WL = calculate_waveform_length(segmentData)
        WLs.append(WL)
        IAV = calculate_integral_absolute_value(segmentData)
        IAVs.append(IAV)

        #list of coordinates (5 lists of length 4)
        waveletMaxes, waveletMins = cwt_spikes(segmentData)
        for n, coord in enumerate(waveletMaxes):
            for i, Xi in enumerate(coord):
                wavelet[n*4 + i].append(Xi * 3)
        for n, coord in enumerate(waveletMins):
            for i, Xi in enumerate(coord):
                wavelet[20 + n*4 + i].append(Xi * 3)


    features = [MAVs, RMSs, ZCs, SSCs, WLs, IAVs]
    features.extend(wavelet)

    #print(len(features))
    #print(features)
    return features



def find_all_windows(timestamps, window_duration_ms):
    # Convert window duration from milliseconds to seconds
    window_duration_s = window_duration_ms / 1000.0

    # Initialize the result list and start index
    windows = []
    start_index = 0

    # Iterate through the list to find all 300ms windows
    while start_index < len(timestamps):
        start_time = timestamps[start_index]
        end_time = start_time + window_duration_s

        # Find the end index for this window
        end_index = start_index
        while end_index < len(timestamps) and timestamps[end_index] <= end_time:
            end_index += 1

        # Subtract 1 from end_index as it points to the first element outside the window
        end_index -= 1

        # Add the window tuple to the result list if the end index is valid
        if start_index <= end_index:
            windows.append((start_index, end_index))

        # Move to the start index of the next window
        #start_index = end_index + 1

    return windows
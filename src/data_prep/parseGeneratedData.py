import numpy as np
import scipy.misc as smc
from natsort import natsorted as ns
import glob, os
import argparse

parser = argparse.ArgumentParser(description="Create Lidar Dataset Parser file")
parser.add_argument("path", help = "path_to_folder", type = str)
args = parser.parse_args()

dataset_path = args.path

raw_path = os.path.join(dataset_path,'raw/2011_09_26/')
processed_path = os.path.join(dataset_path,'processed_20_10/2011_09_26/')

current_dir = os.getcwd()

os.chdir(processed_path)
#Picking up all sync folders
folder_names = ns(glob.glob("*_sync" + os.path.sep))[:-3]

dataset_array = np.zeros(dtype = str, shape = (1,23))
dataset_array_2 = np.zeros(dtype = str, shape = (1,23))

totalDataset = 0
maxPtCldSize = []

for fn in folder_names:
    print(fn)
    file_names_source = ns(glob.glob(os.path.join(processed_path,fn) + "depth_maps_transformed/*.png"))
    file_names_target = ns(glob.glob(os.path.join(processed_path,fn) + "depth_maps/*.png"))
    file_intensity_source = ns(glob.glob(os.path.join(processed_path,fn) + "intensity_maps_transformed/*.png"))
    file_intensity_target = ns(glob.glob(os.path.join(processed_path,fn) + "intensity_maps/*.png"))
    img_source = ns(glob.glob(os.path.join(raw_path,fn) + "image_02/data/*.png"))
    img_target = ns(glob.glob(os.path.join(raw_path,fn) + "image_03/data/*.png"))
    point_src = ns(glob.glob(os.path.join(raw_path,fn) + "velodyne_points/data/*.bin"))
    transforms_list = np.loadtxt(os.path.join(processed_path,fn) + "angle_list.txt", dtype = str)

    # Read the max size of ptCld
    f = open(os.path.join(processed_path,fn) + "maxPtCldSize.txt","r")
    maxPtCldSize.append(int(f.read()))
    f.close()

    file_names_source = np.array(file_names_source, dtype=str).reshape(-1,1)
    file_names_target = np.array(file_names_target, dtype=str).reshape(-1,1)
    file_intensity_source = np.array(file_intensity_source, dtype=str).reshape(-1,1)
    file_intensity_target = np.array(file_intensity_target, dtype=str).reshape(-1,1)
    img_source = np.array(img_source, dtype=str).reshape(-1,1)
    img_target = np.array(img_target, dtype=str).reshape(-1,1)
    point_src = np.array(point_src, dtype=str).reshape(-1,1)

    min_val = min([file_names_source.shape[0], file_names_target.shape[0], file_intensity_source.shape[0], file_intensity_target.shape[0], img_source.shape[0], img_target.shape[0], point_src.shape[0]])

    try:
        dataset = np.hstack((file_names_source, file_names_target, file_intensity_source, file_intensity_target, img_source, img_target, point_src, transforms_list))
        print(dataset.shape)
        totalDataset+=dataset.shape[0]

        dataset_array = np.vstack((dataset_array, dataset))

        #######################################################################################

        file_names_source_2 = ns(glob.glob(os.path.join(processed_path,fn) + "depth_maps_transformed_2/*.png"))
        file_names_target_2 = ns(glob.glob(os.path.join(processed_path,fn) + "depth_maps_2/*.png"))
        file_intensity_source_2 = ns(glob.glob(os.path.join(processed_path,fn) + "intensity_maps_transformed_2/*.png"))
        file_intensity_target_2 = ns(glob.glob(os.path.join(processed_path,fn) + "intensity_maps_2/*.png"))

        transforms_list_2 = np.loadtxt(os.path.join(processed_path,fn) + "angle_list_2.txt", dtype = str)

        file_names_source_2 = np.array(file_names_source_2, dtype=str).reshape(-1,1)
        file_names_target_2 = np.array(file_names_target_2, dtype=str).reshape(-1,1)
        file_intensity_source_2 = np.array(file_intensity_source_2, dtype=str).reshape(-1,1)
        file_intensity_target_2 = np.array(file_intensity_target_2, dtype=str).reshape(-1,1)

        dataset_2 = np.hstack((file_names_source_2, file_names_target_2, file_intensity_source_2, file_intensity_target_2, img_source, img_target,  point_src, transforms_list_2))
        print(dataset_2.shape)
        totalDataset+=dataset.shape[0]

        dataset_array_2 = np.vstack((dataset_array_2, dataset_2))
    
    except ValueError:
        print("the no of points arent equal to the no of images")






dataset_array = dataset_array[1:]
dataset_array_2 = dataset_array_2[1:]

final_array = np.vstack((dataset_array, dataset_array_2))

os.chdir(current_dir)

np.random.shuffle(final_array)
np.savetxt("parsed_set.txt", final_array, fmt = "%s", delimiter=' ')
f = open("maxPtCldSize.txt","w")
f.write(str(max(maxPtCldSize)))
f.close()

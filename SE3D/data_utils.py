from glob import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
import binvox
import os
import random
import pcd_utils
import open3d as o3d

def compute_data_stats(data_list):
    # check that it's channel last
    assert data_list[0]["image"].shape[-1] == np.min(data_list[0]["image"].shape)
    # compute n channels
    image_data = [e["image"] for e in data_list]
    n_channels = data_list[0]["image"].shape[-1]
    sums = np.zeros([n_channels])
    elem_count = 0
    for elem in tqdm(image_data):
        sums += np.sum(elem, axis=(0,1,2))
        elem_count += np.size(elem)//n_channels
    
    means = sums / elem_count

    deviations = np.zeros([n_channels])
    for elem in tqdm(image_data):
        deviations += np.sum((elem-means)**2, axis=(0,1,2))

    deviations = np.sqrt(deviations / elem_count)
    return means.flatten()/255.0, deviations.flatten()/255.0

def load_BraTS_samples(base_folder):
    samples_folders = glob(str(base_folder) + "/*")

    brats_data = []

    def create_sample(base_path, side):
        sample_stem = base_path.split("/")[-1]
        new_sample = {}
        new_sample["image"] = np.load(Path(base_path) / f"{side}/{sample_stem}_{side}.npz")["data"]
        new_sample["segmentation"] = np.load(Path(base_path) / f"{side}/{sample_stem}_seg_{side}.npz")["data"]
        new_sample["side"] = side
        return new_sample

    for sample in samples_folders:
        # Each folder has a left and right sample
        new_sample = create_sample(sample, "left")
        brats_data.append(new_sample)
        new_sample = create_sample(sample, "right")
        brats_data.append(new_sample)

    return brats_data

def load_KiTS_samples(base_folder):
    # All left samples

    kits_data = []

    def create_sample(filepath, side):
        new_sample = {}
        new_sample["image"] = np.load(filepath)["data"]
        seg_id = filepath.split("/")[-1].split("-")[-1]
        folder_path = "/".join(filepath.split("/")[0:-1])
        new_sample["segmentation"] = np.load(Path(folder_path) / f"segmentation-{seg_id}")["data"]
        new_sample["side"] = side
        return new_sample

    samples_left = glob(str(base_folder) + "/left/volume*")
    for sample in samples_left:
        new_sample = create_sample(sample, "left")
        kits_data.append(new_sample)

    samples_right = glob(str(base_folder) + "/right/volume*")
    for sample in samples_right:
        new_sample = create_sample(sample, "right")
        kits_data.append(new_sample)
    
    return kits_data

def load_lung_samples(base_folder):
    return load_KiTS_samples(base_folder)

def load_shapenet_pair_samples(base_folder, train_classes_ids):
    # creates pairs where one sample is from train classes and one sample is from other classes
    def load_binvox(path):
        numpy_model = binvox.Binvox.read(path, mode='dense').numpy().astype(np.uint8)*255
        return numpy_model

    assert len(train_classes_ids) == 2

    class_folders = os.listdir(base_folder)
    class_folders = [e for e in class_folders if os.path.isdir(base_folder / e)]
    
    train_class_folders = [base_folder / e for e in class_folders if e in train_classes_ids]
    noise_class_folders = [base_folder / e for e in class_folders if e not in train_classes_ids]
    
    def load_all_from_folder_list(folder_list):
        data = []
        for folder in folder_list:
            files = glob(str(folder) + "/**/*.binvox", recursive=True)
            for file in files:
                label = file.split("/")[-3]
                data.append((load_binvox(file), label))
        return data

    train_data = load_all_from_folder_list(train_class_folders)
    noise_data = load_all_from_folder_list(noise_class_folders)

    shapenet_data = []
    for train_sample, label in train_data:
        noise_id = np.random.randint(0, len(noise_data))
        noise_sample = noise_data[noise_id][0]
        left_right = np.random.randint(0,2)
        if left_right == 0:
            cat_sample = np.concatenate((train_sample, noise_sample), axis=0)
            cat_segmask = np.concatenate((train_sample//255, np.zeros_like(noise_sample)), axis=0)
            side = "left"
        else:
            cat_sample = np.concatenate((noise_sample, train_sample), axis=0)
            cat_segmask = np.concatenate((np.zeros_like(noise_sample), train_sample//255), axis=0)
            side = "right"
        
        dict_sample = {}
        dict_sample["image"] = cat_sample
        dict_sample["segmentation"] = cat_segmask
        dict_sample["label"] = np.array(train_classes_ids.index(label), dtype=np.int64)
        dict_sample["side"] = side
        shapenet_data.append(dict_sample)
    
    random.shuffle(shapenet_data)
    return shapenet_data

def load_shapenet_binary(base_folder, train_classes_ids):
    # creates a subset of shapenet for classification over the train_classes_ids
    def load_binvox(path):
        numpy_model = binvox.Binvox.read(path, mode='dense').numpy().astype(np.uint8)*255
        return numpy_model

    assert len(train_classes_ids) == 2

    class_folders = os.listdir(base_folder)
    class_folders = [e for e in class_folders if os.path.isdir(base_folder / e)]
    
    train_class_folders = [base_folder / e for e in class_folders if e in train_classes_ids]
    
    def load_all_from_folder_list(folder_list):
        data = []
        for folder in folder_list:
            files = glob(str(folder) + "/**/*.binvox", recursive=True)
            for file in files:
                label = file.split("/")[-3]
                data.append((load_binvox(file), label))
        return data

    train_data = load_all_from_folder_list(train_class_folders)

    shapenet_data = []
    for train_sample, label in train_data:
        dict_sample = {}
        dict_sample["image"] = train_sample
        dict_sample["segmentation"] = train_sample//255
        dict_sample["label"] = np.array(train_classes_ids.index(label), dtype=np.int64)
        dict_sample["side"] = None
        shapenet_data.append(dict_sample)
    
    random.shuffle(shapenet_data)
    return shapenet_data

def shapenet_class_dict(assignment_file_path):
    with open(assignment_file_path) as f:
        lines = f.readlines()
    
    out = {l.split(" ")[0]: l.split(" ")[1] for l in lines}
    return out

def load_scannet(base_folder, train_classnames, isolated):
    # Get all downloaded scene names
    # All scenes might have more than one scan, but we want unique scenes, so only take the one with "_00"
    scan_paths = glob(str(base_folder) + "/scans/scene*/*_vh_clean_2.labels.ply")
    scan_names = ["_".join(e.split("/")[-1].split("_")[:1]) for e in scan_paths]
    scan_names = set(scan_names)
    scan_names = list(scan_names)
    scan_names = [e + "_00" for e in scan_names]

    scannet_data = []

    # Iterate over all scans
    for scan_name in tqdm(scan_names):
        point_cloud, class_ids, object_classes, vertices_classes, vertices_objects = \
            pcd_utils.load_ply_scene(base_folder, scan_name)
        # Iterate over train classes
        for label, classname in enumerate(train_classnames):
            # Find all object ids in pcd that have class = classname
            try:
                searchclass_objects = \
                    [idx for idx, classid in enumerate(object_classes) 
                     if class_ids[classname] == classid]
            except KeyError:
                # print(f"Scan {scan_name} has no {classname} objects, skipping.")
                continue
            
            # Iterate over object of type classname
            for object in searchclass_objects:
                # Filter point cloud based on object and voxelize it
                filtered_point_cloud = pcd_utils.filter_by_object(point_cloud, vertices_objects, object)
                if len(filtered_point_cloud.points) == 0:
                    # If the object is empty
                    continue
                if not isolated:
                    # Find bounds of object and crop point cloud
                    vertices = np.asarray(filtered_point_cloud.points)
                    min_bound = np.min(vertices, axis=0)
                    max_bound = np.max(vertices, axis=0)
                    mask_point_cloud = o3d.geometry.PointCloud(filtered_point_cloud)
                    filtered_point_cloud = pcd_utils.crop_point_cloud(point_cloud, min_bound, max_bound)
                else:
                    mask_point_cloud = filtered_point_cloud
                    
                voxelized_image = pcd_utils.voxelize_point_cloud(filtered_point_cloud)
                voxelized_mask = pcd_utils.voxelize_point_cloud(mask_point_cloud)
                # rotate to y-up for consistency w/ shapenet
                voxelized_image = np.transpose(voxelized_image, axes=(0,2,1))
                voxelized_mask = np.transpose(voxelized_mask, axes=(0,2,1))

                dict_sample = {}
                dict_sample["image"] = voxelized_image*255
                dict_sample["segmentation"] = voxelized_mask
                dict_sample["label"] = np.array(label, dtype=np.int64)
                dict_sample["side"] = None
                scannet_data.append(dict_sample)

    random.shuffle(scannet_data)
    return scannet_data

def load_scannet_isolated(base_folder, train_classnames):
    return load_scannet(base_folder, train_classnames, isolated=True)

def load_scannet_crop(base_folder, train_classnames):
    return load_scannet(base_folder, train_classnames, isolated=False)
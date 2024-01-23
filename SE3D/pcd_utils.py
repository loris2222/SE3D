import json
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

def load_ply_scene(base_path, scan_name):
    scan_path = base_path / f"scans/{scan_name}/{scan_name}_vh_clean_2.labels.ply"
    segments_path = base_path / f"scans/{scan_name}/{scan_name}_vh_clean_2.0.010000.segs.json"
    segment_class_path = base_path / f"scans/{scan_name}/{scan_name}_vh_clean.aggregation.json"
    
    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    mesh_vertices = np.asarray(mesh.vertices)
    with open(segments_path) as f:
        segments_data = json.load(f)
    with open(segment_class_path) as f:
        segments_class_data = json.load(f)
    
    # This indicates the segment to which each vertex belongs to
    vertex_segment_assignment = segments_data["segIndices"]

    # This is a list of dictionaries, each of which contains "segments" which is a list of segment ids from "segIndices" above
    # and a "label" which identifies the class of the object.
    segment_merge_class_assignment = segments_class_data["segGroups"]

    # Find all classes in the scan
    classes = set()
    for merged_segment in segment_merge_class_assignment:
        classes.update([merged_segment["label"]])
    
    # Create class ids
    class_list = list(classes)
    class_ids = {}
    for idx, classname in enumerate(class_list):
        class_ids[classname] = idx
    
    # Assign a class and object id to each segment
    segment_classes = {}
    segment_objects = {}
    object_classes = []
    for merged_segment in segment_merge_class_assignment:
        object_classes.append(class_ids[merged_segment["label"]] if merged_segment["label"] in class_ids.keys() else -1)
        for segment in merged_segment["segments"]:
            segment_classes[segment] = merged_segment["label"]
            segment_objects[segment] = merged_segment["objectId"]
    object_classes = np.array(object_classes)
    
    # Assign a class to each vertex
    vertices_classes = [class_ids[segment_classes[e]] if e in segment_classes.keys() else -1 for e in vertex_segment_assignment]
    vertices_classes = np.array(vertices_classes)

    # Assign an object id to each vertex
    vertices_objects = [segment_objects[e] if e in segment_objects.keys() else -1 for e in vertex_segment_assignment]
    vertices_objects = np.array(vertices_objects)

    # Create point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(mesh_vertices)
    colors = plt.cm.jet(vertices_classes / 40.0)[:, :3]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud, class_ids, object_classes, vertices_classes, vertices_objects

def filter_by_class(point_cloud, vertices_classes, class_ids, filter_classname):
    new_pcd = o3d.geometry.PointCloud()
    vertices = np.asarray(point_cloud.points)
    new_pcd.points = o3d.utility.Vector3dVector(vertices[vertices_classes == class_ids[filter_classname]])
    return new_pcd

def filter_by_object(point_cloud, vertices_objects, filter_objectid):
    new_pcd = o3d.geometry.PointCloud()
    vertices = np.asarray(point_cloud.points)
    new_pcd.points = o3d.utility.Vector3dVector(vertices[vertices_objects == filter_objectid])
    return new_pcd

def crop_point_cloud(point_cloud, min_bound, max_bound):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return point_cloud.crop(bbox)

def voxelize_point_cloud(point_cloud, grid_scale = 1/31):
    new_pcd = o3d.geometry.PointCloud()
    vertices = np.asarray(point_cloud.points)
    new_pcd.points = o3d.utility.Vector3dVector(vertices)
    # Scale to unit cube
    new_pcd.scale(1 / np.max(new_pcd.get_max_bound() - new_pcd.get_min_bound()),
            center=new_pcd.get_center())

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(new_pcd,
                                                            voxel_size=grid_scale)

    voxels = voxel_grid.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    np_voxels = np.zeros([32,32,32])
    np_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return np_voxels


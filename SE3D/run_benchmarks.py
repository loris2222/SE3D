
# Default data paths
from glob import glob
import numpy as np
from pathlib import Path
import data_utils
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchio as tio

from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.hirescam import HiResCAM
from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.saliency_tubes import SaliencyTubes
from pytorch_grad_cam.respond_cam import RespondCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from wsol_3d_metrics.BBoxEvaluator import BBoxEvaluator
from wsol_3d_metrics.BBoxEvaluator3D import BBoxEvaluator3D
from wsol_3d_metrics.MaskEvaluator import MaskEvaluator
from wsol_3d_metrics.F1Evaluator import F1Evaluator, F1OnlyMaskEvaluator
from wsol_3d_metrics.MassConcentrationEvaluator import MassConcentrationEvaluator
from tqdm import tqdm
import argparse

DATASET = ""  # "BraTS", "KiTS", "decathlon_lungs", 
                             # "shapenet-pairs", "shapenet-binary", "scannet-isolated", 
                             #  "scannet-crop"
METHODS = [GradCAM, HiResCAM, GradCAMPlusPlus, RespondCAM, SaliencyTubes]  # Cam method as a class of pytorch_grad_cam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Training model parameters
HIDDEN_DIM = 128
FC_DIM = 512
BATCH_SIZE = 1
# keep N_FOLDS=5, FOLD=0 to have 80/20 split
N_FOLDS = 5
FOLD = 0
# For GPUs with lower memory, lower this so that large samples are excluded
SIZE_THRESH = (0.85)*1e8
# You can retrain the models instead of using the provided ones
TRAIN_MODELS = False

MODEL_SAVE_BASE_PATH = ""

N_SAMPLES = 100  # Number of samples over which to compute the MaxBoxAcc from the test set
N_THRESH_TAU = 100  # Number of thresholds tau to choose between 0 and 1
DELTA_ACC_THRESH_V1 = [50]  # MaxBoxAcc threshold over which to count the sample as correctly localized
DELTA_ACC_THRESH_V2 = [30,50,70]  # MaxBoxAcc threshold over which to count the sample as correctly localized

OUTPUT_PATH = ""
DATASETS_FOLDER_PATH = ""

def run_benchmark():
    global DATASET, METHODS, HIDDEN_DIM, FC_DIM, BATCH_SIZE, N_FOLDS, FOLD, SIZE_THRESH, TRAIN_MODELS, MODEL_SAVE_BASE_PATH, N_SAMPLES, N_THRESH_TAU, DELTA_ACC_THRESH_V1, DELTA_ACC_THRESH_V2, OUTPUT_PATH, DATASETS_FOLDER_PATH
    print(f"Running benchmark on dataset: {DATASET} for saliency methods: {METHODS}")
    print(f"Outputting to {OUTPUT_PATH}")

    print("Loading data (1/3)", end="")
    # To compute mean, std data_utils.compute_data_stats. Here are pre-computed as it takes time.
    if DATASET == "BraTS":
        DATASET_BASE_PATH = DATASETS_FOLDER_PATH / "BraTS"
        TRAIN_DATASET = DATASET
        AGES = [20]
        MEAN = np.array([0.04844053, 0.06671674, 0.04088918, 0.04963282])
        STD = np.array([0.11645673, 0.15704592, 0.09690811, 0.12228158])
        FILTER_THRESH = 0.003
        # All samples  (240, 120, 155, 4)
    elif DATASET in ["shapenet-pairs", "shapenet-binary"]:
        DATASET_BASE_PATH = DATASETS_FOLDER_PATH / "ShapeNetVox32"
        TRAIN_DATASET = DATASET  # You may test on models trained on different datasets
        AGES = [1]
        MEAN = np.array([0.06027088])
        STD = np.array([0.23798802])
        FILTER_THRESH = 0.000
        # All samples shapenet-pairs (64, 32, 32, 1)
        # All samples shapenet-binary (32, 32, 32, 1)
    elif DATASET in ["scannet-isolated", "scannet-crop"]:
        DATASET_BASE_PATH = DATASETS_FOLDER_PATH / "scannet"
        TRAIN_DATASET = DATASET  # You may test on models trained on different datasets
        AGES = [1]
        MEAN = np.array([0.02899467])
        STD = np.array([0.16779148])
        FILTER_THRESH = 0.000
        # All samples  (32, 32, 32, 1)
    elif DATASET == "KiTS":
        raise NotImplementedError("can't get good KiTS classifier")
        DATASET_BASE_PATH = DATASETS_FOLDER_PATH / "KiTS"
        TRAIN_DATASET = DATASET
        AGES = [20, 10]
        MEAN = np.array([0.18517924])
        STD = np.array([0.18119358])
        FILTER_THRESH = 0.001
        # Samples between (512, 256, 29, 1) and (512, 398, 1059, 1)
    elif DATASET == "decathlon_lungs":
        raise NotImplementedError("Lungs dataset not suitable, not enough non-borderline samples")
        DATASET_BASE_PATH = DATASETS_FOLDER_PATH / "decathlon_lungs"
        TRAIN_DATASET = DATASET
        AGES = [20, 10]
        MEAN = np.array([0.11019484])
        STD = np.array([0.12821062])
        FILTER_THRESH = 0.003
        # Samples between (512, 256, 112, 1)and (512, 256, 636, 1)
    else:
        raise NotImplementedError(f"Invalid dataset {DATASET}")

    # Always use the STD, MEAN of the train dataset
    if DATASET != TRAIN_DATASET:
        if TRAIN_DATASET == "shapenet-binary":
            MEAN = np.array([0.06027088])
            STD = np.array([0.23798802])
        elif TRAIN_DATASET in ["scannet-isolated", "scannet-crop"]:
            MEAN = np.array([0.02899467])
            STD = np.array([0.16779148])
        else:
            raise ValueError(f"Invalid TRAIN_DATASET {TRAIN_DATASET}")

    if DATASET == "BraTS":
        data_list = data_utils.load_BraTS_samples(DATASET_BASE_PATH)
    elif DATASET == "KiTS":
        data_list = data_utils.load_KiTS_samples(DATASET_BASE_PATH)
    elif DATASET == "decathlon_lungs":
        data_list = data_utils.load_lung_samples(DATASET_BASE_PATH)
    elif DATASET == "shapenet-pairs":
        class_assignment = data_utils.shapenet_class_dict(DATASET_BASE_PATH / "class_assignment.txt")
        class_list = ["02691156", "02828884"]  # airplane / bench
        # print(f"{[class_assignment[e] for e in class_list]}")
        data_list = data_utils.load_shapenet_pair_samples(DATASET_BASE_PATH, class_list)
    elif DATASET == "shapenet-binary":
        class_assignment = data_utils.shapenet_class_dict(DATASET_BASE_PATH / "class_assignment.txt")
        class_list = ["03001627", "04379243"]  # chair / table
        # print(f"{[class_assignment[e] for e in class_list]}")
        data_list = data_utils.load_shapenet_binary(DATASET_BASE_PATH, class_list)
    elif DATASET == "scannet-isolated":
        class_list = ["chair", "table"]
        # print(f"{class_list}")
        data_list = data_utils.load_scannet_isolated(DATASET_BASE_PATH, class_list)
    elif DATASET == "scannet-crop":
        class_list = ["chair", "table"]
        # print(f"{class_list}")
        data_list = data_utils.load_scannet_crop(DATASET_BASE_PATH, class_list)

    # print(f"Loaded {len(data_list)} samples")

    print("\rLoading data (2/3)", end="")
    # Add channel dimension if samples don't have one
    for sample in data_list:
        if len(sample["image"].shape) == 4:
            break
        sample["image"] = sample["image"][:,:,:,np.newaxis]
        sample["segmentation"] = sample["segmentation"][:,:,:,np.newaxis]
    # print(f"Last sample shape {sample['image'].shape}")

    # Removing the borderline cases
    from copy import copy, deepcopy
    low_thresh_exclusive = 0
    high_thresh_inclusive = FILTER_THRESH

    # Create new dataset where samples that have a fraction of tumor within (low, high] are removed
    # Also plots the new percents
    filtered_datalist = []
    percents_left = []
    percents_right = []

    for elem in data_list:
        value = np.sum(elem["segmentation"])/elem["segmentation"].size
        if value <= low_thresh_exclusive or value > high_thresh_inclusive:
            filtered_datalist.append(deepcopy(elem))
            if elem["side"] == "left":
                percents_left.append(value)
            else:
                percents_right.append(value)

    percents = percents_left + percents_right

    # Filter dataset based on voxel size (large samples don't fit in memory)
    prev_len = len(filtered_datalist)
    filtered_datalist = [e for e in filtered_datalist if e["image"].size <= SIZE_THRESH]
    # print(f"Filtered elements with size larger than {SIZE_THRESH}, removed {prev_len-len(filtered_datalist)} samples. Remaining {len(filtered_datalist)}")

    print("\rLoading data (3/3)")
    # Create labels for the filtered data
    if DATASET == "BraTS":
        def brats_labeler(elem):
            value = np.sum(elem["segmentation"])/elem["segmentation"].size
            elem["label"] = value > 0
        
        labeler = brats_labeler
    elif DATASET in ["shapenet-pairs", "shapenet-binary", "scannet-isolated", "scannet-crop"]:
        def shapenet_labeler(elem):
            # shapenet and scannet already have labels from loader
            pass

        labeler = shapenet_labeler
    else:
        raise NotImplementedError(f"No labeler for dataset {DATASET}")

    for elem in filtered_datalist:
        labeler(elem)

    # Normalize image data
    n_channels = data_list[0]["image"].shape[-1]
    labels = []
    for elem in filtered_datalist:
        elem["image"] = ((elem["image"].astype(np.double) / 255.0) - MEAN.reshape([1,1,1,n_channels])) / np.sqrt(STD).reshape([1,1,1,n_channels])
        elem["label"] = elem["label"].astype(int)
        labels.append(elem["label"])
    #print(f"Output shape: {elem['image'].shape}")
    #print(f"Labels: {labels}")

    # Utilities for dataset folding
    def fold_data_list(data_list, n_folds, test_fold, split="train"):
        l = len(data_list)
        fold_size = l//n_folds
        if split=="train":
            return data_list[0:test_fold*fold_size] + data_list[(test_fold+1)*fold_size:]
        else:
            return data_list[test_fold*fold_size:(test_fold+1)*fold_size]

    print("Creating model (1/2)", end="")
    # Torch Classification model
    assert FOLD < N_FOLDS
    # Transpose data to channel first for use with torch
    for elem in filtered_datalist:
        if elem["image"].shape[0] == np.min(elem["image"].shape):
            #print("Elements already transposed, skipping")
            break
        elem["image"] = np.transpose(elem["image"], [3,0,1,2])
        if len(elem["segmentation"].shape) < 4:
            elem["segmentation"] = elem["segmentation"][np.newaxis,:,:,:]
        else:
            elem["segmentation"] = np.transpose(elem["segmentation"], [3,0,1,2])
    #print(f"Output image shape: {elem['image'].shape}")
    #print(f"Output segmentation shape: {elem['segmentation'].shape}")

    # Define 3D CNN classifier
    assert (HIDDEN_DIM%4)==0

    class CNN3DModel(nn.Module):
        def __init__(self, in_channels, num_classes):
            super(CNN3DModel, self).__init__()
            
            self.conv1 = nn.Conv3d(in_channels, HIDDEN_DIM//4, kernel_size=3, padding='same')
            self.conv2 = nn.Conv3d(HIDDEN_DIM//4, HIDDEN_DIM//2, kernel_size=3, padding='same')
            self.conv3 = nn.Conv3d(HIDDEN_DIM//2, HIDDEN_DIM, kernel_size=3, padding='same')
            
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
            
            self.fc2 = nn.Linear(HIDDEN_DIM, num_classes)

            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.softmax = nn.Softmax(dim=-1)
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))

            x = self.gap(x)
            x = x.view(-1, HIDDEN_DIM)
            x = self.fc2(x) 
            
            return x

    # Create dataloader
    class CustomDataset(Dataset):
        def __init__(self, data_list, n_folds, test_fold):
            self.data = fold_data_list(data_list, n_folds, test_fold, split="train")
            spatial_transforms = {
                tio.RandomElasticDeformation(): 0.2,
                tio.RandomAffine(): 0.8,
            }
            self.transform = tio.Compose([
                tio.OneOf(spatial_transforms, p=0.5),
                tio.RandomFlip(axes="lr", flip_probability=0.5)
            ])

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = torch.tensor(self.data[idx]["image"], dtype=torch.float)
            image = self.transform(image)
            label = torch.tensor(self.data[idx]["label"], dtype=torch.int64)
            return image, label


    print("\rCreating model (2/2)")
    # Create an instance of CustomDataset
    custom_dataset = CustomDataset(filtered_datalist, N_FOLDS, FOLD)

    # Create a DataLoader
    batch_size = BATCH_SIZE
    train_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Instantiate model
    num_classes = 2

    model = CNN3DModel(n_channels, num_classes).cuda()

    # Compute class weights
    class_counts = {}

    # Iterate through the DataLoader to count occurrences of each class
    for _, labels in train_loader:
        for label in labels.numpy():
            if label not in class_counts:
                class_counts[label] = 1
            else:
                class_counts[label] += 1

    total_samples = sum(class_counts.values())
    class_frequencies = {key: value / total_samples for key, value in class_counts.items()}

    print("Computing class weights (1/1)")
    # Calculate class weights as the inverse of class frequencies
    class_weights = {key: 1.0 / value for key, value in class_frequencies.items()}

    # Normalize weights so that they sum to the number of classes
    sum_weights = sum(class_weights.values())
    class_weights_normalized = {key: value / sum_weights for key, value in class_weights.items()}
    class_weights_list = [1/class_weights_normalized[x] for x in range(num_classes)]
    class_weights_list = [x/sum(class_weights_list) for x in class_weights_list]
    class_weights_list = [class_weights_normalized[x] for x in range(num_classes)]

    #print("Class Counts:", class_counts)
    #print("Class Weights:", class_weights_list)


    # Train or load model
    if DATASET != TRAIN_DATASET or not TRAIN_MODELS:
        print("Loading model (1/1)")
        model.load_state_dict(torch.load(MODEL_SAVE_BASE_PATH / f"model_{TRAIN_DATASET}_fold{FOLD}_torch{HIDDEN_DIM}"))
        model.eval()
    else:
        print("Training model (1/1)")
        # training loop
        ages = AGES

        for idx, epochs in enumerate(ages):
            print(f"Age {idx} with {epochs} epochs")
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights_list).cuda())
            optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # 0.0001
            for epoch in range(epochs):
                step=0
                running_loss = 0
                for inputs, labels in train_loader:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    step += 1
                    print(f'\r Step: {step} Loss: {(running_loss/step):.4f}', end='')
                    
                # scheduler.step()
                #print(f'\r Epoch [{epoch+1}/{epochs}], Loss: {(running_loss/step):.4f}')

    # Save model
    if TRAIN_MODELS:
        save_path_str = str(MODEL_SAVE_BASE_PATH / f"model_{DATASET}_fold{FOLD}_torch{HIDDEN_DIM}")
        print(f"Saving model to {save_path_str}")
        torch.save(model.state_dict(), save_path_str)
    model.eval()

    print("Evaluating model (1/1)")
    # Test on training data (no augmentation obvs)
    ok = 0
    checked = 0
    checked_p = 0
    checked_n = 0
    for sample in fold_data_list(filtered_datalist, N_FOLDS, FOLD, split="test"):
        image_in = torch.tensor(sample["image"], dtype=torch.float).cuda()
        if len(image_in.shape) < 5:
            image_in = image_in.unsqueeze(0)
        model_out = model(image_in).cpu().detach().numpy()
        checked += 1
        checked_p = checked_p + 1 if sample['label'] else checked_p
        checked_n = checked_n + 1 if not sample['label'] else checked_n
        if np.argmax(model_out) == sample['label']:
            ok += 1

    print(f"Accuracy: {ok/checked}")
    print(f"There where {checked_p} positive and {checked_n} negative samples")

    print("Computing metrics (1/1)")
    # Compute evaluation metrics
    MODEL = model  # Model to compute cam from 
    TARGET_LAYER = model.conv3  # Last conv layer to compute cam
    PRED_LAYER = model.fc2

    write_file = open(OUTPUT_PATH, "w")

    for METHOD in METHODS: 
        print()
        print(METHOD)
        write_file.write(f"\nCAM: {METHOD}\n")
        thresholds = list(np.linspace(start=0, stop=1, num=N_THRESH_TAU+1, endpoint=False))[1:]  # From 0 to 1 both exclusive
        if METHOD == SaliencyTubes:
            cam = METHOD(model=MODEL, target_layers=[TARGET_LAYER], pred_layer = PRED_LAYER)
        else:
            cam = METHOD(model=MODEL, target_layers=[TARGET_LAYER])

        MaxBoxAcc_evaluator = BBoxEvaluator(
            iou_threshold_list=DELTA_ACC_THRESH_V1,
            multi_contour_eval=False,
        )
        MaxBoxAccV2_evaluator = BBoxEvaluator(
            iou_threshold_list=DELTA_ACC_THRESH_V2,
            multi_contour_eval=True,
        )
        VxAP_evaluator = MaskEvaluator(  # It's the same also as PxAP
            iou_threshold_list=DELTA_ACC_THRESH_V1,
        )
        Max3DBoxAcc_evaluator = BBoxEvaluator3D(
            iou_threshold_list=DELTA_ACC_THRESH_V1,
            multi_contour_eval=False,
        )
        Max3DBoxAccV2_evaluator = BBoxEvaluator3D(
            iou_threshold_list=DELTA_ACC_THRESH_V2,
            multi_contour_eval=True,
        )
        F1_Evaluator = F1Evaluator()
        F1MaskOnly_Evaluator = F1OnlyMaskEvaluator()
        MassConcentration_Evaluator = MassConcentrationEvaluator()

        evaluators = [
            {
                "name": "MaxBoxAcc",
                "evaluator_f": MaxBoxAcc_evaluator,
                "3D": False,
            },
            {
                "name": "MaxBoxAccV2",
                "evaluator_f": MaxBoxAccV2_evaluator,
                "3D": False,
            },
            {
                "name": "VxAP",
                "evaluator_f": VxAP_evaluator,
                "3D": True,
            },
            {
                "name": "Max3DBoxAcc",
                "evaluator_f": Max3DBoxAcc_evaluator,
                "3D": True,
            },
            {
                "name": "Max3DBoxAccV2",
                "evaluator_f": Max3DBoxAccV2_evaluator,
                "3D": True,
            },
            {
                "name": "F1 (F1Score, tau_f1, (Prec@tau_f1, Rec@tau_f1))",
                "evaluator_f": F1_Evaluator,
                "3D": True,
            },
        ]

        # Mass Concentration is only for shapenet-pairs
        if DATASET == "shapenet-pairs":
            evaluators.append({
                "name": "MassConcentration",
                "evaluator_f": MassConcentration_Evaluator,
                "3D": True,
            })

        # Compute cams for all samples
        all_cams = []
        test_samples = fold_data_list(filtered_datalist, N_FOLDS, FOLD, split="test")
        if N_SAMPLES < 1:
            N_SAMPLES = len(test_samples)
            print(f"Executing on all {N_SAMPLES} test samples")
        if N_SAMPLES > len(test_samples):
            raise ValueError(f"N_SAMPLES={N_SAMPLES} is greater than available samples={len(test_samples)}")
        test_samples = test_samples[:N_SAMPLES]
        for sample in tqdm(test_samples):
            image_in = torch.tensor(sample["image"], dtype=torch.float).cuda()

            target_label = int(sample["label"])

            targets = [ClassifierOutputTarget(target_label)]
            grayscale_cam = cam(input_tensor=image_in, targets=targets).astype(np.float32)
            if np.sum(grayscale_cam==0):
                grayscale_cam[0,0,0] = np.finfo(float).eps
            normalized_cam = (grayscale_cam - np.min(grayscale_cam))/(np.max(grayscale_cam)-np.min(grayscale_cam))
            all_cams.append(normalized_cam)

        # Compute metrics for computed CAMs
        for evaluator in evaluators:
            print(f"Computing {evaluator['name']}")
            # If the evaluator function supports 3D, then call its accumulator function, else call it on each slice
            # NOTE: Due to accumulation over slices, the result may not be a multiple of 1/N_SAMPLES

            if evaluator["3D"]:
                def accumulate(heatmap, mask):
                    evaluator["evaluator_f"].accumulate(heatmap, mask)
            else:
                def accumulate(heatmap, mask):
                    for i in range(heatmap.shape[-1]):
                        evaluator["evaluator_f"].accumulate(heatmap[..., i], mask[..., i])

            for idx, normalized_cam in tqdm(enumerate(all_cams)):
                accumulate(normalized_cam, test_samples[idx]["segmentation"][0].astype(np.float32))

            print(f"{evaluator['name']}: {evaluator['evaluator_f'].compute()}")
            write_file.write(f"{evaluator['name']}: {evaluator['evaluator_f'].compute()}\n")
    write_file.close()


def main():
    global DATASET, METHODS, HIDDEN_DIM, FC_DIM, BATCH_SIZE, N_FOLDS, FOLD, SIZE_THRESH, TRAIN_MODELS, MODEL_SAVE_BASE_PATH, N_SAMPLES, N_THRESH_TAU, DELTA_ACC_THRESH_V1, DELTA_ACC_THRESH_V2, OUTPUT_PATH, DATASETS_FOLDER_PATH
    parser = argparse.ArgumentParser(
                    prog='SE3D',
                    description='Runs the SE3D benchmark')
    
    parser.add_argument('-d', '--dataset', help="The dataset to compute the benchmark on. One of [shapenet-binary, shapenet-pairs, scannet-isolated, scannet-crop, BraTS]", required=True)
    parser.add_argument('-o', '--output_path', help="The path where to save the output file. Defaluts to 'cwd/data/output.txt'", default=Path(os.getcwd()) / "data" / "output.txt")
    parser.add_argument('--datasets_base_path', help="The path where datsets are stored. Refer to the guide for help on how to structure the folder.", default=Path(os.getcwd()) / "datasets")
    parser.add_argument('--models_base_path', help="The path where models are stored. Refer to the guide for help on how to structure the folder.", default=Path(os.getcwd()) / "models")

    args = vars(parser.parse_args())
    DATASET = args["dataset"]
    OUTPUT_PATH = Path(args["output_path"])
    DATASETS_FOLDER_PATH = Path(args["datasets_base_path"])
    MODEL_SAVE_BASE_PATH = Path(args["models_base_path"])

    run_benchmark()

if __name__ == "__main__":
    main()
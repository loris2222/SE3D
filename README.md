# SE3D

This is the official implementation for **SE3D: A FRAMEWORK FOR SALIENCY METHOD EVALUATION IN 3D IMAGING**.

Authors: `Mariusz Wiśniewski, Loris Giulivi, Giacomo Boracchi`

> For more than a decade, deep learning models have been dominating in various 2D imaging tasks. Their application is now extending to 3D imaging, with 3D Convolutional Neural Networks (3D CNNs) being able to process LIDAR, MRI, and CT scans, with significant implications for fields such as autonomous driving and medical imaging.
In these critical settings, explaining the model's decisions is fundamental. Despite recent advances in Explainable Artificial Intelligence, however, little effort has been devoted to explaining 3D CNNs, and many works explain these models via inadequate extensions of 2D saliency methods. 
> One fundamental limitation to the development of 3D saliency methods is the lack of a benchmark to quantitatively assess them on 3D data.
To address this issue, we propose SE3D: a framework for \textbf{S}aliency method \textbf{E}valuation in \textbf{3D} imaging.
We propose modifications to ShapeNet, ScanNet, and BraTS datasets, and evaluation metrics to assess saliency methods for 3D CNNs.
We evaluate both state-of-the-art saliency methods designed for 3D data and extensions of popular 2D saliency methods to 3D. Our experiments show that 3D saliency methods do not provide explanations of sufficient quality, and that there is margin for future improvements and safer applications of 3D CNNs in critical fields.

## Quick Start

### Requirements

Install the necessary libraries:

```sh
pip install -r requirements.txt
```
This codebase is tested on `Python 3.11.4`, `CUDA 12.2`, `Ubuntu 22.04`.

Download the datasets following [this](#datasets-and-preprocessing).

### Code Structure

The code is structured as follows:

```sh
SE3D                    # root directory
├── README.md           # this file
├── evaluate.py         # script for evaluating the saliency methods
├── metadata.csv        # metadata file for the customized BraTS 2020 dataset
├── model.py            # 3D CNN model architecture definition
├── prepare_dataset.py  # script for creating the customized BraTS 2020 dataset
├── requirements.txt    # requirements file
├── results_to_md.py    # script for converting the results to Markdown format
├── run_all.sh          # script for running all experiments
├── tf_saliency_methods # module implementing the saliency methods
├── train.py            # script for training the 3D CNN model
└── wsol_3d_metrics     # module implementing the evaluation metrics
```


## Datasets and Preprocessing

The first step to run the benchmark is to download and pre-process the three datasets `ShapeNet`, `ScanNet`, and `BraTS`. You can choose to download a subset of these if you are only interested on benchmarks on one particular dataset.

Download links and instructions:
> [ShapeNet](https://shapenet.org/)  (Already voxelized version available [here](http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.binvox.zip))
> 
>1. Download the voxelized version or voxelize the original models using [binvox](https://www.patrickmin.com/binvox/).

> [Scannet](http://www.scan-net.org/)
> 
>1. Use the provided download script. Only download the filetypes: `['.aggregation.json', '_vh_clean.aggregation.json', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.labels.ply']`.

> [BraTS](https://www.med.upenn.edu/cbica/brats2020/data.html)
> 
>1. Download the entire dataset.
>2. Use the provided tool to split the sample at the *corpus callosum*. `python prepare_dataset.py --train_data_root <brats original download folder> --output_dir <[...]/SE3D/datasets/BraTS>`

Once downloaded, place the datasets in the `/datasets` folder. Your folder structure should look like this:

```sh
SE3D
├── ...
└── datasets
    ├── BraTS
    │   ├── BraTS2D_Training_001
    │   ├── ...
    ├── scannet
    │   └── scans
    └── ShapeNetVox32
        ├── 02691156
        ├── ...
        └── class_assignment.txt
```

### Adding your dataset

The paper demonstrates the use of the proposed framework on `ShapeNet`, `ScanNet`, and `BraTS` datasets. The framework is however not limited to these, and more datasets can be added. To do so:

1. Write a dataset loader and add it to `data_utils.py`. The dataset loader must be a function that takes as input a string path to the base folder of the dataset, and returs a list of samples. Each sample must be a dictionary with keys `image`, `segmentation`, `label`, and `side`. `image` and `segmentation` are the samples image content and binary label, as a numpy `ndarray`, and of types `float` and `int`, respectively. `label` is the sample's binary label, stored as an integer `{0,1}`. `side` is for paired datasets, and can be `"left"` or `"right"`. Default to `"left"` for datasets that are not subject to pairing.

2. Compute your dataset statistics using `data_utils.compute_data_stats(data_list)` where `data_list` is the output of your data loader.

3. Add a `DATASET` term in `run_benchmarks.py` in this form:
```py
elif DATASET in ["your_dataset"]:
        DATASET_BASE_PATH = DATASETS_FOLDER_PATH / "your_dataset_name"
        TRAIN_DATASET = DATASET
        AGES = [1]  # The training strategy for your dataset
        MEAN = np.array([xxx])  # The mean as computed by 'compute_data_stats'
        STD = np.array([yyy])  # The std as computed by 'compute_data_stats'
        FILTER_THRESH = 0.000  # The hard-sample threshold. All samples with segmentation fraction in 0<p<FILTER_THRESH will be removed.
```

### Additional info regarding BraTS
The dataset is provided in the form of NIfTI files, which can be opened using the [NiBabel](https://nipy.org/nibabel/)
library.

The scans are provided in the form of 4D volumes, with the fourth dimension representing the different MRI modalities.
The modalities are:

- native (**T1**)
- post-contrast T1-weighted (**T1Gd**),
- T2-weighted (**T2**),
- T2 Fluid Attenuated Inversion Recovery (**T2-FLAIR**).

The ground truth segmentation masks are provided in the form of 3D volumes, with each voxel containing a label from the
following set:

- 0: background,
- 1: necrotic and non-enhancing tumor core (**NCR/NET**),
- 2: peritumoral edema (**ED**),
- 4: GD-enhancing tumor (**ET**).

The preprocessing steps are implemented in the `prepare_dataset.py` script. The script takes as input the path to the training folder of the BraTS 2020 dataset, and creates a folder containing the preprocessed dataset.

The following preprocessing steps are performed:

1. Each volume is rotated by $-90^{\circ}$ around $x$ and $y$ axes. This is done to align the volumes with the standard radiological view, where the left side of the brain is on the right side of the image.
2. Each volume is first min-max normalized to the range $[0, 1]$ and then converted to the range $[0, 255]$. This is done to reduce the memory footprint of the dataset, as the original volumes are stored as 32-bit floating point numbers.
3. The segmentation masks are converted to contain only two labels: background (0) and tumor (1). This is done by merging the labels 1, 2 and 4 into a single label. The intuition behind this is that the tumor is the only region of interest for the saliency methods, and the different labels are not relevant for the evaluation.
4. Both volumes and segmentation masks are split into 2 halves along the *corpus callosum* plane. Thanks to this, the resulting dataset contains both volumes with and without the tumor.

To run the preprocessing script, use the following command:

```sh
python prepare_dataset.py --train_data_root <path_to_train_data_root> \
                            --output_dir <path_to_output_dir>
```

## Saliency Methods

The implementation for saliency methods is based on the [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) codebase. We have implemented from scratch and included to the framework the two `Respond-CAM` and `Saliency Tubes`.
The methods are implemented in the `SE3D/pytorch_grad_cam` module.

Available methods:

- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- [Grad-CAM++](https://arxiv.org/abs/1710.11063)
- [HiResCAM](https://arxiv.org/abs/2011.08891)
- [Respondd-CAM](https://arxiv.org/abs/1806.00102)
- [Saliency Tubes](https://arxiv.org/abs/1902.01078)

## Evaluation Metrics

The evaluation metrics are implemented in the `SE3D/wsol_3d_metrics` module. The following metrics are available:

Existing metrics designed for 2D CNN saliency methods:
- `MaxBoxAcc`
- `MaxBoxAccV2`
- `PxAP`

Proposed metrics: 
- `Max3DBoxAcc`
- `Max3DBoxAccV2`
- `VxAP`
- `MaxF1`
- `Prec@`$\tau_{F1}$
- `Rec@`$\tau_{F1}$

### Adding your metric
The framework is flexible to extensions to other metrics. To add a metric, you need to:

1.  Add a metric in `SE3D/wsol_3d_metrics/your_metric.py`. The new class must implement two methods: `accumulate(self, heatmap, mask)` and `compute(self)`.
2. Add the metric in `run_benchmark.py` in this format:
```py
{
    "name": "your_name",
    "evaluator_f": YOUR_CLASS,
    "3D": True,  # Whether to evaluate the metric on slices of the volume (False) or on the entire volume (True).
},
```

## Licenses

This code is distributed under the MIT license.\
Check the datasets' website for their respective licenses, which might differ.

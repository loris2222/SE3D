# from wsol_3d_metrics.LocalizationEvaluator import LocalizationEvaluator, check_heatmap_validity
from copy import deepcopy
from sklearn.metrics import f1_score
import numpy as np
import torch
from tqdm import tqdm 

class F1Evaluator():
    def __init__(self):
        # super().__init__()
        self.cam_list = []
        self.mask_list = []

    def accumulate(self, heatmap, mask):
        self.cam_list.append(heatmap)
        self.mask_list.append(mask)
    
    def compute_f1_score(self, cams, masks, threshold):
        cams_binary = (cams >= threshold).astype(int)

        # cams_binary = torch.from_numpy(cams_binary).cuda()
        # masks = torch.from_numpy(masks).cuda()

        # tp = torch.sum(torch.logical_and(cams_binary, masks))
        # all_predicted = torch.sum(cams_binary)
        # all_positive = torch.sum(masks)
        tp = np.sum(np.logical_and(cams_binary, masks))
        all_predicted = np.sum(cams_binary)
        all_positive = np.sum(masks)

        if all_predicted==0 or all_positive==0 or tp==0:
            return 0
        prec = (tp / all_predicted)  # .cpu().numpy()
        rec = (tp / all_positive)  # .cpu().numpy()

        return (2*prec*rec)/(prec+rec)
    
    def compute_prec_rec_at_t(self, cams, masks, threshold):
        cams_binary = (cams >= threshold).astype(int)

        tp = np.sum(np.logical_and(cams_binary, masks))
        all_predicted = np.sum(cams_binary)
        all_positive = np.sum(masks)

        if all_predicted==0 or all_positive==0 or tp==0:
            return 0, 0
        prec = (tp / all_predicted)  # .cpu().numpy()
        rec = (tp / all_positive)  # .cpu().numpy()

        return prec, rec

    
    def find_optimal_f1(self, cams, masks, threshold_range=(0, 1), num_thresholds=100):
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        avg_f1_scores = []

        print("Thresholds: ")
        for threshold in tqdm(thresholds):
            f1_scores = self.compute_f1_score(cams, masks, threshold)
            avg_f1_scores.append(f1_scores)

        # optimal_threshold = thresholds[np.argmax(avg_f1_scores)]
        max_avg_f1_score = np.nanmax(avg_f1_scores)
        best_thresh_id = np.nanargmax(avg_f1_scores)

        return max_avg_f1_score, thresholds[best_thresh_id], self.compute_prec_rec_at_t(cams, masks, thresholds[best_thresh_id])

    def compute(self):
        return self.find_optimal_f1(np.array(self.cam_list), np.array(self.mask_list))
    

class F1OnlyMaskEvaluator(F1Evaluator):
    def __init__(self):
        super().__init__()
        self.cam_list = []
        self.mask_list = []

    def accumulate(self, heatmap, mask):
        filtered_heatmap = deepcopy(heatmap)
        filtered_heatmap[mask==0] = 0
        super().accumulate(filtered_heatmap, mask)

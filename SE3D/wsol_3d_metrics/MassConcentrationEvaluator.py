from copy import deepcopy
# from wsol_3d_metrics.LocalizationEvaluator import LocalizationEvaluator, check_heatmap_validity
import numpy as np

class MassConcentrationEvaluator():
    def __init__(self):
        # super().__init__()
        self.correct_mass = 0
        self.wrong_mass = 0

    def accumulate(self, heatmap, mask):
        # Transpose heatmap, mask so that the left/right dimension is the first
        left_right_dim = np.argmax(heatmap.shape)
        dim_order = [left_right_dim] + [i for i in range(len(heatmap.shape)) if i != left_right_dim]
        heatmap = heatmap.transpose(dim_order)
        mask = mask.transpose(dim_order)

        # Find the side where there should be mass
        left_right_size = mask.shape[0] // 2
        mask_sum_left = np.sum(mask[:left_right_size,:,:])
        mask_sum_right = np.sum(mask[-left_right_size:,:,:])
        assert not (mask_sum_left>0 and mask_sum_right>0)

        # Add correct/wrong mass based on the side where there should be mass
        if mask_sum_left > 0:
            self.correct_mass += np.sum(heatmap[:left_right_size,:,:])
            self.wrong_mass = np.sum(heatmap[-left_right_size:,:,:])
        elif mask_sum_right > 0:
            self.wrong_mass += np.sum(heatmap[:left_right_size,:,:])
            self.correct_mass = np.sum(heatmap[-left_right_size:,:,:])
        else:
            raise ValueError("Invalid mask is all zeros")

    def compute(self):
        return self.correct_mass / (self.correct_mass + self.wrong_mass)

    
    
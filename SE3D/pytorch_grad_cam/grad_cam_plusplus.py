import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM

# https://arxiv.org/abs/1710.11063


class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(model, target_layers,
                                              reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(
            *list(range(2, len(activations.shape))),))
        eps = 0.000001

        if len(input_tensor.shape) == 4:
            aij = grads_power_2 / (2 * grads_power_2 +
                                   sum_activations[:, :, None, None] * grads_power_3 + eps)
        elif len(input_tensor.shape) == 5:
            aij = grads_power_2 / (2 * grads_power_2 +
                                   sum_activations[:, :, None, None, None] * grads_power_3 + eps)
        else:
            raise Exception("Unsupported shape: {}".format(
                sum_activations.shape))
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(
            *list(range(2, len(activations.shape))),))
        return weights

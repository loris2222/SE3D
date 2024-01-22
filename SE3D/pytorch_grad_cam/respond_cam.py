import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class RespondCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            RespondCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.sum(activations * grads, axis=(*list(range(2, len(activations.shape))),)) / np.sum(activations + 1e-10, axis=(*list(range(2, len(activations.shape))),))

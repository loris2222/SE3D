import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM


class SaliencyTubes(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None, pred_layer=None):
        super(
            SaliencyTubes,
            self).__init__(
            model,
            target_layers,
            reshape_transform)
        if pred_layer is None:
            raise Exception("pred_layer must be specified")
        self.pred_layer = pred_layer

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        if len(target_category) != 1:
            raise Exception("Expected only one target category")

        # get predictions, last convolution output and the weights of the prediction layer
        pred_weights = self.pred_layer.weight.data.detach().cpu().numpy().transpose()
        last_conv_output = activations.transpose(1, 2, 3, 0)
        cam = np.zeros(last_conv_output.shape[0:3], dtype=np.float32)

        for i, w in enumerate(pred_weights[:, target_category[0].category]):
            cam += w * last_conv_output[..., i]
        return cam

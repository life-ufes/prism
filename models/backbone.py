import timm
from torch import nn


class TIMM(nn.Module):
    CONFIG_BY_MODEL = {
        "mobilenet-v3": {
            "weights": "mobilenetv3_small_100.lamb_in1k",
            "n_feat_output": 1024,
        },
        "efficientnet-b0": {
            "weights": "timm/efficientnet_b0.ra_in1k",
            "n_feat_output": 1280,
        },
        "davit_tiny": {
            "weights": "davit_tiny.msft_in1k",
            "n_feat_output": 768,
        },
        "swinv2_tiny": {
            "weights": "swinv2_cr_tiny_ns_224.sw_in1k",
            "n_feat_output": 768,
        },
    }

    def __init__(self, model_name, global_pool=None):
        super().__init__()

        if model_name not in TIMM.CONFIG_BY_MODEL.keys():
            raise Exception(f"The model {model_name} is not available!")

        self.feature_extractor = timm.create_model(
            TIMM.CONFIG_BY_MODEL[model_name]["weights"],
            pretrained=True,
            num_classes=0,
            global_pool=global_pool,
        )

    @staticmethod
    def get_output_size(model_name):
        return TIMM.CONFIG_BY_MODEL[model_name]["n_feat_output"]

    def forward(self, img):
        return self.feature_extractor(img)

import torch

from torch import nn
from models.naivebayes import NaiveBayes
from models.backbone import TIMM
from models.multimodal.factory import MultimodalAdapterFactory
from models.multimodal.base import MultimodalAdapter


class Classifier(nn.Module):
    def __init__(
        self, backbone: nn.Module, feature_fusion: MultimodalAdapter, n_classes: int
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_fusion = feature_fusion
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, img, meta, *args, **kwargs):
        feats = self.backbone(img)
        if self.feature_fusion is not None:
            feats = self.feature_fusion(feats, meta)
        return self.classifier(feats)


class ClassifierFactory:
    @staticmethod
    def get(
        n_classes,
        model_name,
        comb_method=None,
        n_metadata=None,
        checkpoint=None,
        vision_checkpoint=None,
        n_categorical_metadata=None,
        n_numerical_metadata=None,
    ):
        backbone = TIMM(
            model_name, global_pool="" if comb_method == "remixformer" else None
        )

        feature_fusion = None
        if comb_method is not None and comb_method != "naive_bayes":
            feature_fusion = MultimodalAdapterFactory.get(
                comb_method,
                vision_model_output_size=TIMM.get_output_size(model_name),
                n_metadata=n_metadata,
            )

        classifier = Classifier(backbone, feature_fusion, n_classes)

        if checkpoint is not None and comb_method != "naive_bayes":
            checkpoint = torch.load(checkpoint, weights_only=False)
            classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)

        # deal with the Naive Bayes separately because it needs to wrap the classifier
        if comb_method == "naive_bayes":
            classifier = NaiveBayes(
                classifier,
                eps=1e-6,
                laplacian_smoothing=1.0,
                n_classes=n_classes,
                n_categorical_features=n_categorical_metadata,
                n_numerical_features=n_numerical_metadata,
            )
            if checkpoint is not None:
                checkpoint = torch.load(checkpoint, weights_only=False)
                classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
            elif vision_checkpoint is not None:
                vision_checkpoint = torch.load(vision_checkpoint, weights_only=False)
                classifier.vision_model.load_state_dict(
                    vision_checkpoint["model_state_dict"], strict=True
                )

        return classifier

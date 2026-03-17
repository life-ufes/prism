from torch import nn
from models.multimodal.crossattention import CrossAttentionAdapter
from models.multimodal.metablock import MetaBlockAdapter


class MultimodalAdapterFactory(nn.Module):

    def get(fusion: str, vision_model_output_size, n_metadata):
        if fusion == "cross_attention":
            return CrossAttentionAdapter(vision_model_output_size, n_metadata)

        if fusion == "metablock":
            return MetaBlockAdapter(vision_model_output_size, n_metadata)

        if fusion == "remixformer":
            from models.multimodal.remixformer import CrossModalityFusionAdapter

            return CrossModalityFusionAdapter(vision_model_output_size, n_metadata)

        raise ValueError(f'The fusion method "{fusion}" is not implemented!')

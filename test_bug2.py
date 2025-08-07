import open_clip
import transformers

class Config(transformers.PretrainedConfig):
    pass

class Model(transformers.PreTrainedModel):
    config_class = Config

    def __init__(self, config):
        super().__init__(config)
        self.another_model = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")[0]


a = Model(Config())
a.save_pretrained("/tmp/abc")

b = Model.from_pretrained("/tmp/abc")

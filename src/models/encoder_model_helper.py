import logging
import types
from typing import List, Tuple

import torch

from src.models import MLP, EncoderNetwork
from src.typing import EncoderConfigs, EncoderModelConfigs, MinModelConfig

logger = logging.getLogger(__name__)


def fake_train(self, *args, **kwargs):
    return self


class EncoderModelHelper:
    def __init__(
        self, encoder_model_configs: EncoderModelConfigs, current_cv_iteration: int
    ):
        if current_cv_iteration is None:
            raise NotImplementedError("Encoder models are only supported for CV")

        self.encoders_configs = encoder_model_configs["encoders"]
        self.finetuner_configs = encoder_model_configs["finetuning"]
        self.simple_encoders: List[MinModelConfig] = []
        self.current_cv_iteration = current_cv_iteration

    def add_simple_encoder(self, encoder_config: MinModelConfig):
        logger.debug("Adding base model settings")
        self.simple_encoders.append(encoder_config)

    def reset_bases(self):
        self.simple_encoders.clear()

    def _create_base(
        self, model_config: MinModelConfig, input_size: int
    ) -> Tuple[MLP, int]:
        assert model_config["input_dim"] is not None
        assert model_config["output_dim"] is not None

        assert model_config["input_dim"] == input_size

        logger.debug("Creating Base encoder model")

        model = MLP(
            num_layers=model_config["num_layers"],
            input_dim=model_config["input_dim"] or -1,
            hidden_dim=model_config["hidden_dim"],
            output_dim=model_config["output_dim"] or -1,
            use_batch_norm=model_config["use_batch_norm"],
            hidden_layers=model_config["hidden_layers"],
        )

        return model, model.out_features

    def _create_encoder(
        self, encoder_config: EncoderConfigs, input_size: int
    ) -> Tuple[MLP, int]:
        model_config = encoder_config["model_config"]
        model_weights_path = encoder_config["encoder_path"]
        freeze_weights = encoder_config["freeze_encoder"]

        logger.debug("Creating pretrained encoder model")

        model, output_size = self._create_base(
            model_config=model_config, input_size=input_size
        )

        encoder_weights = torch.load(
            model_weights_path.format(self.current_cv_iteration)
        )["model"]
        model.load_state_dict(encoder_weights)

        if freeze_weights:
            model.requires_grad_(False)
            model.eval()

            # ! monkeypatch train method on frozen modules
            model.train = types.MethodType(fake_train, model)

        return model, output_size

    def _create_finetuning_model(
        self, model_config: MinModelConfig, output_size: int, input_size: int
    ) -> MLP:
        logger.debug("Creating finetuning model")

        finetuning_config = {**model_config, "output_dim": output_size}
        model, _ = self._create_base(
            model_config=finetuning_config, input_size=input_size
        )

        return model

    def create_encoder_model(
        self, model_input_size: int, model_output_size: int
    ) -> EncoderNetwork:
        logger.debug(
            "Building encoder model with input "
            f"{model_input_size} and output {model_output_size}"
        )

        simple_encoders = []
        pretrained_encoders = []

        embeddings_size = 0

        for encoder_conf in self.simple_encoders:
            encoder_mlp, output_size = self._create_base(
                model_config=encoder_conf,
                input_size=model_input_size,
            )
            simple_encoders.append(encoder_mlp)
            embeddings_size += output_size

        for encoder_conf in self.encoders_configs:
            encoder_mlp, output_size = self._create_encoder(
                encoder_config=encoder_conf,
                input_size=model_input_size,
            )
            pretrained_encoders.append(encoder_mlp)
            embeddings_size += output_size

        finetuner_module = self._create_finetuning_model(
            model_config=self.finetuner_configs,
            input_size=embeddings_size,
            output_size=model_output_size,
        )

        return EncoderNetwork(
            pretrained_encoders=pretrained_encoders,
            base_encoders=simple_encoders,
            finetuner_module=finetuner_module,
        )

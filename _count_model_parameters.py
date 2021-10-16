import os
from typing import OrderedDict

import torch


def count_parameters(model_dict: OrderedDict) -> int:
    param_count = 0
    for values in model_dict.values():
        if isinstance(values, torch.Tensor):
            param_count += values.numel()
        elif isinstance(values, dict):
            param_count += count_parameters(model_dict=values)

    return param_count


evaluation_path = os.path.join(
    "results",
    "v4",
    "crossfold_raw",
    "40e65407aa",
    "text+encoder_v2+color(norem)",
    "models",
    "NoFilter()-TextSequenceAtomic()-CV-F(True)-ENC[color1024x4,lower512x1+16,upper512x1+16]-FINE[2]-emb4-lstmcellIN8-lstmH8-initTrue-catTrue-drop0-compFalse-d256-32b-0.001lr_cf1.pt",
)

encoder_dict, decoder_dict, _ = torch.load(evaluation_path).values()

for k in encoder_dict.keys():
    print(k)
# print(encoder_dict["finetuner_module.batch_norms.0.weight"].requires_grad)
# print(encoder_dict["pretrained_encoders.0.linears.0.weight"].requires_grad)


encoder_params = count_parameters(encoder_dict)
decoder_params = count_parameters(decoder_dict)

print("encoder", encoder_params)
print("decoder", decoder_params)
print("total", encoder_params + decoder_params)

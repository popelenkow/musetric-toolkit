from pathlib import Path

from huggingface_hub import constants as hf_constants

model_checkpoint_url = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt"
model_config_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml"

models_dir = Path(hf_constants.HF_HUB_CACHE) / "model_bs_roformer_ep_368_sdr_12.9628"

model_checkpoint_path = models_dir / "model.ckpt"
model_config_path = models_dir / "config.yaml"

karaoke_mdx_model_url = (
    "https://huggingface.co/AI4future/RVC/resolve/main/UVR_MDXNET_KARA_2.onnx"
)
karaoke_mdx_models_dir = Path(hf_constants.HF_HUB_CACHE) / "uvr_mdxnet_kara_2"
karaoke_mdx_model_path = karaoke_mdx_models_dir / "UVR_MDXNET_KARA_2.onnx"

mdx_model_data_url = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
    "mdx_model_data/model_data_new.json"
)
mdx_model_data_dir = Path(hf_constants.HF_HUB_CACHE) / "mdx_model_data"
mdx_model_data_path = mdx_model_data_dir / "model_data_new.json"

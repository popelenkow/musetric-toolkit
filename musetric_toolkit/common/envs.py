from pathlib import Path

from platformdirs import PlatformDirs

model_checkpoint_url = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt"
model_config_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml"

models_dir = Path(PlatformDirs("musetric-toolkit", "musetric").user_data_path)

model_checkpoint_path = models_dir / "model_bs_roformer_ep_368_sdr_12.9628.ckpt"
model_config_path = models_dir / "model_bs_roformer_ep_368_sdr_12.9628.yaml"

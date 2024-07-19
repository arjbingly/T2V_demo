import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "mps"
dtype = torch.float16

step = 8  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"  # Choose to your favorite base model.
# base = "Lykon/DreamShaper"

repo = "guoyww/animatediff"
ckpt = "v3_sd15_mm.ckpt"

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))


pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
# Add Motion LoRA

# pipe.load_lora_weights(
#     "guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out"
# )

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

# output = pipe(prompt="A girl smiling", guidance_scale=1.0, num_inference_steps=step)
output = pipe(prompt="A girl smiling", guidance_scale=7.0, num_inference_steps=10)
export_to_gif(output.frames[0], "./animation.gif")

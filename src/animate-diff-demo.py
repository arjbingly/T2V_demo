import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, DDIMScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "mps"

repo = "guoyww/animatediff-motion-adapter-v1-5"
base = "SG161222/Realistic_Vision_V5.1_noVAE"

# Load the motion adapter
adapter = MotionAdapter.from_pretrained(repo).to(device)
# load SD 1.5 based finetuned model
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter).to(device)
scheduler = DDIMScheduler.from_pretrained(
    base, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1
)
pipe.scheduler = scheduler

# enable memory savings
# pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=10,
    guidance_scale=7.5,
    num_inference_steps=10,
)
frames = output.frames[0]
export_to_gif(frames, "animation_v1-5.gif")

import gc

import streamlit as st
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, DDIMScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import base64
import time

st.set_page_config("T2V demo",)
st.title("AnimateDiff T2V Demo")

with st.sidebar:
    st.info('Model Configs')
    st.session_state.model_type = st.selectbox('AnimateDiff Model',
                                  ['lightning', 'original'])

    if st.session_state.model_type == 'lightning':
        st.session_state.step = st.selectbox('Model Distillation Step', [2, 4, 8], index=2)
    elif st.session_state.model_type == 'original':
        st.session_state.version = st.selectbox('Model Version', ['v1', 'v2', 'v3'], index=2)

    st.session_state.device = st.selectbox('Compute Device',
                              ['mps', 'cpu'])
    # dtype = torch.float16
    st.session_state.base = st.selectbox('Base Model',
                            ["emilianJR/epiCRealism",
                             "Lykon/DreamShaper",
                             "SG161222/Realistic_Vision_V6.0_B1_noVAE"])

    submitted = st.button(
        "Load Model", type="primary", use_container_width=True)

    clear_mem = st.button(
        "Clear Memory", type="secondary", use_container_width=True)

    if clear_mem:
        if 'adaptor' in st.session_state:
            st.info('Cleared adaptor')
            del(st.session_state.adapter)
        if 'pipe' in st.session_state:
            st.info('Cleared pipline')
            del(st.session_state.pipe)
        gc.collect()
        torch.mps.empty_cache()
        st.info('Garbage collected and cache emptied')


    if submitted:
        # clearing mem
        if 'pipe' in st.session_state:
            del st.session_state.pipe
        if 'adaptor' in st.session_state:
            del st.session_state.adapter

        dtype = torch.float16
        with st.status("Loading Model"):
            st.write("Loading adapter...")
            if st.session_state.model_type =='lightning':
                repo = "ByteDance/AnimateDiff-Lightning"
                ckpt = f"animatediff_lightning_{st.session_state.step}step_diffusers.safetensors"
                st.session_state.adapter = MotionAdapter().to(st.session_state.device, dtype)
                st.session_state.adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=st.session_state.device))
            elif st.session_state.model_type == 'original':
                # TODO version matching
                version_matcher = {
                    'v1': "guoyww/animatediff-motion-adapter-v1-5",
                    'v2': "guoyww/animatediff-motion-adapter-v1-5-2",
                    'v3': "guoyww/animatediff-motion-adapter-v1-5-3" ,
                }
                repo = version_matcher[st.session_state.version]
                st.session_state.adapter = MotionAdapter.from_pretrained(repo).to(st.session_state.device)

            st.write("Loading pipe...")
            st.session_state.pipe = AnimateDiffPipeline.from_pretrained(st.session_state.base,
                                                                        motion_adapter=st.session_state.adapter,
                                                                        torch_dtype=dtype,
                                                                        ).to(st.session_state.device)
            if st.session_state.model_type == 'lightning':
                st.session_state.pipe.scheduler = EulerDiscreteScheduler.from_config(
                    st.session_state.pipe.scheduler.config,
                    timestep_spacing="trailing",
                    beta_schedule="linear")
            elif st.session_state.model_type == 'original':
                st.session_state.pipe.scheduler = DDIMScheduler.from_pretrained(
                    st.session_state.base,
                    subfolder="scheduler",
                    clip_sample=False,
                    timestep_spacing="linspace",
                    steps_offset=1)


            # # enable memory savings
            # st.session_state.pipe.enable_vae_slicing()
            # st.session_state.pipe.enable_model_cpu_offload()
            st.session_state.pipe.enable_attention_slicing()

        st.success("Model Loaded")

    with st.form('MotionLoRA'):
        st.info('MotionLoRA')
        motion_models = {
            "None": None,
            "pan left" : "guoyww/animatediff-motion-lora-pan-left",
            "pan right" :"guoyww/animatediff-motion-lora-pan-right",
            "roll anticlockwise" : "guoyww/animatediff-motion-lora-rolling-anticlockwise",
            "roll clockwise" :"guoyww/animatediff-motion-lora-rolling-clockwise",
            "tilt down" : "guoyww/animatediff-motion-lora-tilt-down",
            "tilt up" : "guoyww/animatediff-motion-lora-tilt-up",
            "zoom in" : "guoyww/animatediff-motion-lora-zoom-in",
            "zoom out" :"guoyww/animatediff-motion-lora-zoom-out",
        }
        m_model = st.selectbox('Motion Model',
                            motion_models.keys())

        msubmitted = st.form_submit_button(
            "Load MotionLoRA", type="primary", use_container_width=True)

    if msubmitted:
        if 'pipe' in st.session_state:
            if motion_models[m_model]:
                with st.status("Loading Motion Model"):
                    st.session_state.pipe.load_lora_weights(
                        motion_models[m_model], adapter_name=m_model
                    )
        else:
            st.error("No model has been loaded")


with st.form("t2i"):
    prompt = st.text_input("Prompt",
                           value="")
    negative_prompt = st.text_input("Neg. Prompt",
                                    # value="worst quality, low quality, ugly, deformed, blurry")
    )

    guidance_scale = st.slider(label="Gudidance Scale",
                               min_value=0.0,
                               max_value=20.0,
                               step=1.0,
                               value=1.0,
                               help="Controls how closely the model adheres to the provided text prompt during the generation process.")

    num_inference_steps = st.slider(label="Number of denoising steps",
                                    min_value=1,
                                    max_value=50,
                                    step=1,
                                    value=7,
                                    help = "The number of denoising steps. More denoising steps usually lead to a higher quality video")

    num_frames = st.slider(label="Number of frames",
                           min_value=2,
                           max_value=32,
                           step=1,
                           value=10,
                           help="The number of frames generated for the gif.")

    generate = st.form_submit_button(
            "Generate", type="primary", use_container_width=True)
if generate:
    output = None
    if 'pipe' in st.session_state:

        start = time.perf_counter()
        with st.status("Generating..."):
            st.write("Inferencing..")
            output = st.session_state.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
            )
            st.write("Exporting..")
            export_to_gif(output.frames[0], "../animation.gif")
        if output:
            time_taken = (time.perf_counter() - start)
            st.success(f"GIF Generated (Time taken: {time_taken:.2f}sec)")

        file_ = open("../animation.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="generated gif">',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download",
            data=file_,
            file_name="animation.gif",
        )
        file_.close()

    else:
        st.error("No model has been loaded")

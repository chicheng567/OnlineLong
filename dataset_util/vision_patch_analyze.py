import torch
from videollama3.model.videollama3_qwen2 import Videollama3Qwen2ForCausalLM
from videollama3.model.videollama3_encoder.image_processing_videollama3 import Videollama3ImageProcessor
model_local = "../OnlineLong/pretrained_models/videollama3_7b_local"
device = "cuda:0"
vl3 = Videollama3Qwen2ForCausalLM.from_pretrained(
    model_local,
    trust_remote_code=True,
    device_map=device,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
vl3 = vl3.eval()
image_processor = Videollama3ImageProcessor(
    do_convert_rgb=True,
    do_normalize=True,
    do_rescale=True,
    do_resize=True,
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
    max_tokens=16384,
    min_tokens=16,
    patch_size=14,
    resample=3,
    rescale_factor=0.00392156862745098,
    merge_size=1,
    force_size=[448, 448],
)
from videollama3.mm_utils import load_video
video_path = "v__7a80bvsbk8.mp4"
frames, timestamps = load_video(video_path, fps=1, max_frames=200)
vision_encoder = vl3.get_vision_encoder()
# Forward image frames through vision encoder
with torch.no_grad():
    inputs = image_processor(frames, return_tensors="pt")
    # Convert pixel_values to bfloat16 on GPU
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)
    merge_sizes = inputs["merge_sizes"].to(device)  # LongTensor on GPU
    grid_sizes = inputs["grid_sizes"].to(device)    # LongTensor on GPU

    patches = vision_encoder(pixel_values=pixel_values, merge_sizes=merge_sizes, grid_sizes=grid_sizes)
    patches = patches.view(len(frames), -1, patches.size(-1))  # (num_frames, num_patches, hidden_size)
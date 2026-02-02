# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch

from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh


# 与训练 config 一致：传 params 覆盖 hub 默认，即可用更小点数（省显存）
# 不传则用 hub 默认 pc_size=81920
PC_SIZE = 8192
PC_SHARPEDGE_SIZE = 0

vae = ShapeVAE.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    use_safetensors=False,
    variant='fp16',
    pc_size=PC_SIZE,
    pc_sharpedge_size=PC_SHARPEDGE_SIZE,
)

loader = SharpEdgeSurfaceLoader(
    num_sharp_points=PC_SHARPEDGE_SIZE,
    num_uniform_points=PC_SIZE,
)
mesh_demo = 'demos/010.glb'
surface = loader(mesh_demo).to('cuda', dtype=torch.float16)
print(surface.shape)

latents = vae.encode(surface)
latents = vae.decode(latents)
mesh = vae.latents2mesh(
    latents,
    output_type='trimesh',
    bounds=1.01,
    mc_level=0.0,
    num_chunks=20000,
    octree_resolution=256,
    mc_algo='mc',
    enable_pbar=True
)

mesh = export_to_trimesh(mesh)[0]
mesh.export('output.obj')

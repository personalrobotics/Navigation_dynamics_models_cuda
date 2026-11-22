"""Build IGHA* with a model header from this repo.

Search and Node stay in IGHAStar. This only puts our include dir first and
compiles our .cu / _cpu.cpp. `_snm` keeps the torch build dir off the in-tree one.
"""

from pathlib import Path

import torch

from ighastar.scripts.common_utils import (
    BASE_DIR,
    CXX_STD_FLAG,
    _ext_load,
    cuda_cflags,
    env_header_flag,
    kernel_cflags,
)
from ighastar.scripts.fetch_robin_hood import ensure_robin_hood

HERE = Path(__file__).resolve().parent
_CUDA = {
    "kinodynamic": ("kinodynamic.h", "kinodynamic.cu"),
    "kinematic": ("kinematic.h", "kinematic.cu"),
    "holonomic_6D": ("holonomic_6D.h", "holonomic_6D.cu"),
}
_CPU = {
    "kinodynamic": ("kinodynamic_cpu.h", "kinodynamic_cpu.cpp"),
    "kinematic": ("kinematic_cpu.h", "kinematic_cpu.cpp"),
}


def load_model(configs, bidirectional=False):
    node_info = configs["experiment_info_default"]["node_info"]
    if str(node_info.get("map_type", "elevation")).lower() == "esdf_bev":
        node_info["map_type"] = "elevation"
    env_name = node_info["node_type"]
    cuda = torch.cuda.is_available()
    if env_name == "holonomic_6D" and not cuda:
        raise RuntimeError("holonomic_6D requires CUDA; no CPU fallback is available")
    header, src_name = (_CUDA if cuda else _CPU)[env_name]

    src = Path(BASE_DIR) / "src"
    includes = [
        str(HERE / "include"),
        str(src / "Environments" / "include"),
        str(src / "utils"),
        str(src),
    ]
    hash_cflags = []
    hash_tag = "std"
    robin = ensure_robin_hood(quiet=True)
    if robin is not None:
        includes.append(str(robin))
        hash_cflags = ["-DIGHA_USE_ROBIN_HOOD"]
        hash_tag = "rh"
    tag = CXX_STD_FLAG.replace("-std=", "").replace("+", "x")
    mod = f"ighastar_{env_name}_{tag}_{hash_tag}_eh_snm"
    flags = [CXX_STD_FLAG, "-O3", env_header_flag(header), *hash_cflags]
    model_src = HERE / "src" / src_name
    if cuda:
        kernel = _ext_load(
            name=f"{mod}_T{node_info['timesteps']}",
            sources=[str(src / "ighastar.cpp"), str(model_src)],
            extra_include_paths=includes,
            extra_cflags=flags,
            extra_cuda_cflags=cuda_cflags(*hash_cflags, *kernel_cflags(node_info)),
            verbose=True,
        )
    else:
        kernel = _ext_load(
            name=mod,
            sources=[str(src / "ighastar.cpp"), str(model_src)],
            extra_include_paths=includes,
            extra_cflags=[*flags[:2], "-fopenmp", *flags[2:]],
            extra_ldflags=["-fopenmp"],
            verbose=True,
        )
    cls = kernel.BiIGHAStar if bidirectional else kernel.IGHAStar
    return cls(configs, False)

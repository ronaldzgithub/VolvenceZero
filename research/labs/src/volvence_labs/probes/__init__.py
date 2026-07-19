"""内置 probes 的入口。

导入此模块会触发所有内置 probe 的注册（通过各子模块 __init__.py）。
"""

# Stage 0
from . import primitive_5_epistemic_pe  # noqa: F401
from . import frontier_5_r15_formalization  # noqa: F401

# Stage 1 batch 1
from . import primitive_4_multitimescale_memory  # noqa: F401
from . import primitive_3_emergent_switching  # noqa: F401

# Stage 1 batch 2
from . import primitive_7_readonly_monitoring  # noqa: F401
from . import primitive_6_bounded_self_mod  # noqa: F401
from . import primitive_2_latent_controller  # noqa: F401

# Stage 1 batch 3
from . import primitive_1_frozen_substrate  # noqa: F401

# Stage 2: Frontier probes
from . import frontier_1_pe_llm_scale  # noqa: F401
from . import frontier_2_crossmodal_z  # noqa: F401
from . import frontier_3_dual_track_regime  # noqa: F401
from . import frontier_4_credit  # noqa: F401

# Stage 3: New frontier probes
from . import frontier_5_sparse_circuits  # noqa: F401
from . import frontier_6_alpha_evolve  # noqa: F401

# Stage 1 batch 3
from . import primitive_1_frozen_substrate  # noqa: F401

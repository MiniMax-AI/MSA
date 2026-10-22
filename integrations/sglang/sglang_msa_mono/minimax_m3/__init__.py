"""MiniMax-M3.1 TP4 verify on MSA-v1 MegaAttention + token-parallel MoE, and its
DSpark draft on the one-launch draft backbone.

``spec`` holds the target's SGLang contract, ``weights`` the conversion of
SGLang's parameters and the checkpoint's routed experts into the kernels'
layouts, and ``runtime`` the routed forward; ``draft_spec`` and
``draft_runtime`` are the draft's contract and forward.
"""

# Fixed by the kernels (csrc/common/kernel/include/tp4_nvls_layout.hpp,
# csrc/moe_token_parallel/include/sm100/common/layout.hpp and the validators);
# every value is per TP4 rank unless noted.
TP = 4
ROWS = 8  # query rows of one launch: one request's eight verify tokens
HIDDEN = 6144
HEAD_DIM = 128
LOCAL_Q_HEADS = 16
ROTARY_DIM = 64
PAGE = 128  # KV tokens a page
TOPK_BLOCKS = 16
EXPERTS = 128
TOP_K = 4
EXPERT_INTER = 3072  # global intermediate of one routed expert
INTER = EXPERT_INTER // TP  # 768: this rank's slice of every expert
SHARED_INTER = 3072
PROJECTION_ROWS = 2560  # q 2048 | k 128 | v 128 | qi 128 | ki 128
O_PROJ_K = LOCAL_Q_HEADS * HEAD_DIM  # 2048
# tp4_nvls::kIndexerMaxPages: the page-table width MegaAttention accepts
INDEXER_MAX_PAGES = 8192
MAX_CONTEXT = INDEXER_MAX_PAGES * PAGE  # 1048576 tokens
GRID_CTAS = 144  # both persistent grids: Scheduler::kGridCtas and moe kGrid
# The DSpark draft backbone (inference/high_tps/msa_v1/draft/backbone.py)
DRAFT_ROWS = 7  # one request's draft block: the bonus token and six mask tokens
DRAFT_LAYERS = 5
DRAFT_QKV_ROWS = LOCAL_Q_HEADS * HEAD_DIM + 2 * HEAD_DIM  # 2304: q | k | v
DRAFT_FFN_INTER = 3072  # this rank's slice of the 12288 dense FFN
DRAFT_WINDOW = 4096  # keys a row attends, HF sliding_window

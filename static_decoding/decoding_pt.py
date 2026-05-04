# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# 在束搜索的选择中高效从候选池中选取topM序列。torch.gather保持批次和历史维度
def _gather_beams(x: torch.Tensor, beam_indices: torch.Tensor) -> torch.Tensor:
  """Efficiently gathers beam data across a batch during the selection step.

  Uses torch.gather to select the top-M sequences from the candidate pool
  while preserving batch and history dimensions.

  Args:
      x: The source tensor to gather from.
          Shape: (batch_size, old_beam_size, ...).
      beam_indices: The indices of the beams to select.
          Shape: (batch_size, new_beam_size).

  Returns:
      The gathered tensor.
          Shape: (batch_size, new_beam_size, ...).
  """
  """
  # 假设参数
    # x = torch.randn(4, 8, 10, 12)      # (batch_size=4, old_beam_size=8, seq_len=10, hidden=12)
    # beam_indices = torch.tensor([[0, 2], [1, 3], [0, 1], [2, 4]])  # (4, 2)
    # 每行表示该样本要选哪些束：样本0选第0和第2个束，样本1选第1和第3个束...
  """
  batch_size, new_beam_size = beam_indices.shape # (batch_size, beam_size)
  #  x的维度数是[batch_size, beam_width, seq_len, hidden_size]
  # x.dim()-2 = 2. view_shape就是（bs, beam_size, 1, 1） 占位，为后续广播做准备
  view_shape = [batch_size, new_beam_size] + [1] * (x.dim() - 2)
  
  # 最终的形状。（batch_size, new_beam_size, seq_len, hidden_size）
  expand_shape = [batch_size, new_beam_size] + list(x.shape[2:])
  
# beam_indices: (4, 2)
#     ↓ .view([4, 2, 1, 1])
# 中间结果: (4, 2, 1, 1)
#     ↓ .expand([4, 2, 10, 12])  # 广播复制
# indices: (4, 2, 10, 12)

  indices = beam_indices.view(view_shape).expand(expand_shape)
  # gather要求index 和 input形状相同
  # 按照dim=1（束维度收集数据）
  """
  # 按照dim=1（束维度收集数据）, indices表明了要哪些束
  """
  return x.gather(1, indices)

"""
将不规则的指针追逐trie遍历替换为向量化操作
工作流程：
1. 批量读取 CSR 行指针 → 获取每个状态的候选子节点范围
2. 使用固定偏移量矩阵一次性读取所有候选 (Token, NextState) 对
3. 创建有效性掩码处理节点子节点数 < limit 的情况
4. 从模型输出的 logprobs 中 gather 对应 token 的概率
5. 对无效路径应用 -inf 掩码
关键优化：O(1) 延迟，与约束集大小无关（传统 trie 遍历是 O(depth)）。
"""
@torch.inference_mode()
def generate_and_apply_logprobs_mask(
    flat_logprobs: torch.Tensor,
    flat_states: torch.Tensor,
    packed_csr: torch.Tensor,
    csr_indptr: torch.Tensor,
    limit: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Performs vectorized sparse candidate extraction from the STATIC CSR matrix.

  This kernel replaces irregular "pointer-chasing" trie traversals with a single
  vectorized burst-read. It retrieves the log-probabilities for all valid child
  tokens of the current trie states in one coalesced operation, ensuring $O(1)$
  latency relative to the total constraint set size.

  Args:
      flat_logprobs: Model-predicted log-probabilities.
          Shape: (batch_size * beam_size, vocab_size).
      flat_states: Current trie state IDs for each beam.
          Shape: (batch_size * beam_size,).
      packed_csr: The flat transition table [Token ID, Next State ID].
          Shape: (num_transitions + V, 2).
      csr_indptr: The CSR row pointer array identifying segments.
          Shape: (num_states + 2,).
      limit: The maximum branching factor (K) for the current trie depth.
      vocab_size: The total token vocabulary size (V).
      device: The accelerator device (CUDA/TPU).

  Returns:
      A tuple containing:
          - candidate_logprobs: Log-probs for valid children.
              Shape: (batch_size * beam_size, K).
          - candidate_token_ids: Token IDs for valid children.
              Shape: (batch_size * beam_size, K).
          - candidate_next_states: Next state IDs for valid children.
              Shape: (batch_size * beam_size, K).
  """
  # 1. Fetch Sparse Rows (Burst Read)
  # We perform a coalesced read from the CSR by indexing into the row pointers.
  starts = csr_indptr[flat_states.long()]
  actual_lens = csr_indptr[flat_states.long() + 1] - starts

  # Create a grid of offsets to gather exactly 'limit' (K) candidates per state.
  offsets = torch.arange(limit, device=device)
  gather_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)

  # Clamp indices to handle states with fewer than 'limit' children safely.
  max_idx = packed_csr.size(0) - 1
  safe_gather_indices = gather_indices.clamp(max=max_idx)

  # Retrieve [Token, NextState] pairs directly from High-Bandwidth Memory (HBM).
  gathered_vals = packed_csr[safe_gather_indices]
  candidate_token_ids = gathered_vals[..., 0]
  candidate_next_states = gathered_vals[..., 1]

  # 2. Validity Masking
  # Mask out 'padding' slots if a trie node has fewer than 'limit' children.
  valid_mask = offsets.unsqueeze(0) < actual_lens.unsqueeze(1)

  # 3. Logprob Gathering
  # Gather only the specific log-probabilities corresponding to valid tokens.
  safe_token_ids = candidate_token_ids.long().clamp(max=vocab_size - 1)
  candidate_logprobs = flat_logprobs.gather(1, safe_token_ids)

  # Apply -inf mask to invalidate non-existent paths in the prefix tree.
  candidate_logprobs = torch.where(
      valid_mask, candidate_logprobs, torch.tensor(-float('inf'), device=device)
  )

  return candidate_logprobs, candidate_token_ids, candidate_next_states


# 输出随机logits
class RandomModel(nn.Module):
  """A dummy model that acts like a Transformer but outputs random logits.

  Used to benchmark the throughput of the decoding harness without the
  computational overhead of a real neural network.
  """

  def __init__(self, vocab_size: int, device: torch.device):
    super().__init__()
    self.vocab_size = vocab_size
    self.device = device
    self.to(device)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Generates random logits for the next token prediction.

    Args:
        input_ids: Shape (batch_size, seq_len).

    Returns:
        Random logits. Shape: (batch_size, 1, vocab_size).
    """
    batch_size = input_ids.size(0)
    return torch.rand(batch_size, 1, self.vocab_size, device=self.device)


"""
主解码循环：执行完整的自回归constrained beam search

混合策略：
├── Dense 层 (前 d_dense 步)
│   └── 使用预计算的 dense_mask / dense_states 直接索引
│       适合状态空间密集的浅层
└── Sparse 层 (后续步骤)
    └── 使用 CSR 结构
        适合状态空间稀疏的深层

解码流程：

步骤	操作
初始化	用 start_token 获取首 token 的 logits，应用根掩码，选 top-beam 个 token
自回归循环	每步：模型前向 → 应用约束掩码 → 计算累积分数 → 选择 top beams → 更新状态和历史
返回	解码完成的 token 序列 (batch_size, beam_size, max_sample_len)

数据结构
packed_csr: 压缩稀疏行格式，存储 trie 的转移表 [Token ID, Next State ID]
csr_indptr: CSR 行指针，标识每个状态的候选区间
dense_mask: 密集有效性掩码，快速判断 token 是否合法
dense_states: 密集状态转移表，直接映射到下一状态

核心思想
将传统 trie 的逐节点遍历改为批量向量化操作，通过 CSR 结构一次性获取所有有效候选，避免了条件分支和内存不连续访问，适合 GPU/TPU 并行加速。
"""
@torch.inference_mode()
def sparse_transition_torch(
    model: nn.Module,
    batch_size: int,
    beam_size: int,
    tokens_per_beam: int,
    start_token: int,
    max_sample_len: int,
    vocab_size: int,
    max_branch_factors: tuple[int, ...],
    packed_csr: torch.Tensor,
    csr_indptr: torch.Tensor,
    start_mask: torch.Tensor,
    dense_mask: torch.Tensor,
    dense_states: torch.Tensor,
    device: torch.device,
    d_dense: int = 2,
) -> torch.Tensor:
  """Main harness for STATIC constrained beam search using a PyTorch model.

  Executes the full autoregressive decoding loop. Generates logits from the
  provided `model` and applies the STATIC hybrid masking strategy
  (Dense + CSR) to strictly enforce the constraint graph.


                    ┌─────────────────────────────────────┐
                    │         vocab_size (V)              │
                    │    例: 词汇表有50000个token          │
                    └─────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   start_mask    │     │     dense_mask      │     │   dense_states  │
│    Shape: (V,)  │     │  Shape: (V, V)      │     │  Shape: (V, V)  │
│    根节点掩码    │     │  d_dense=2时       │     │  d_dense=2时    │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
                                 │
                                 │ 超过 d_dense 层后
                                 ▼
                    ┌─────────────────────────────────────┐
                    │          packed_csr + csr_indptr   │
                    │          稀疏CSR结构                │
                    │   存储深层trie的稀疏转移             │
                    └─────────────────────────────────────┘
  Args:
      model: A PyTorch model that accepts input_ids of shape (B*M, 1) and
          returns logits of shape (B*M, 1, V). 返回logits。
      batch_size: Number of sequences to decode in parallel (B). 并行解码的序列数
      beam_size: Number of beams to maintain per sequence (M). 每条序列保留的束数量
      tokens_per_beam: Number of candidate tokens to consider per beam. 每个束考虑的候选token数量
      start_token: The token ID used to initiate decoding (e.g., BOS or PAD). 解码开始的token
      max_sample_len: Length (L) of the Semantic IDs being decoded. 要生成的序列长度
      vocab_size: Size of the token vocabulary (V). 
      max_branch_factors: Maximum branching factors per level.
          Length: L.
          
      压缩稀疏转移表 packed_csr[索引] = [token, next_state] 比如状态0： token1 -> 跳转到状态5
      存储trie的所有转移边
      packed_csr: Flattened trie transitions (Sparse Tail).
          Shape: (num_transitions + V, 2).
          
      CSR 行指标，
      状态 i 的候选在 packed_csr[csr_indptr[i] : csr_indptr[i+1]]
        例如：


        csr_indptr = [0, 3, 5, 7, ...]
                    ↑  ↑  ↑
                    │  │  └─ 状态2从索引5开始
                    │  └──── 状态1从索引3开始，有2个转移(3-5)
                    └─────── 状态0从索引0开始，有3个转移(0-3)
      csr_indptr: CSR row pointers. 
          Shape: (num_states + 2,).
       
      根节点掩码，标记从根节点可以走哪些token   
      start_mask: 1D validity mask for the root (Level 0).
          Shape: (V,).
          
      dense_mask — 密集掩码

        Shape: (V,) * d_dense
        d_dense=1: (V,)
        d_dense=2: (V, V)
        用于 trie 浅层的快速查表：
        # d_dense=2 的情况
        dense_mask[parent_token, child_token] = True/False
        # 状态(parent_token) 下，child_token 是否合法    
      dense_mask: d_dense-dimensional dense validity mask (Hot Head).
          Shape: (V,) * d_dense.
          
    # d_dense=2 的情况
dense_states[parent_token, child_token] = next_state_id
# 从状态(parent_token) 经过 token(child_token) 跳转到 next_state
      dense_states: d_dense-dimensional dense state table.
          Shape: (V,) * d_dense.
      device: Device to execute on.
      d_dense: Number of initial dense layers.
          NOTE: In practice, we only support d_dense=1 and d_dense=2
          (recommended).

  Returns:
      The decoded token sequences.
          Shape: (batch_size, beam_size, max_sample_len).
  """
  # --- 1. INITIAL STEP (Codeword 1) ---
  # Use the specific start_token expected by the model (BOS/PAD) --这里就是<SID> 起始标识符
  initial_input = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

  # Get real logits from the model
  initial_logits = model(initial_input)
  raw_logprobs = F.log_softmax(initial_logits[:, 0, :], dim=-1)

  # Apply the root mask to restrict the first token
  initial_logprobs = torch.where(
      start_mask, raw_logprobs, torch.tensor(-float('inf'), device=device)
  )
  # 只允许trie根节点
  top_logprobs, top_tokens = torch.topk(initial_logprobs, beam_size, dim=-1)

  # Initialize decoding buffers
  # 初始化解码缓冲区，创建存储生成序列的张量，并填入第一个token
  token_buffer = torch.full(
      (batch_size, beam_size, max_sample_len),
      start_token,
      dtype=top_tokens.dtype,
      device=device,
  )
  token_buffer[:, :, 0] = top_tokens

  # Map Level-0 tokens to their initial trie state IDs (Token T -> ID T+1)
  current_transition_states = top_tokens + 1
  current_token_scores = top_logprobs

  # --- 2. AUTOREGRESSIVE LOOP (Codewords 2 to L) ---
  for step in range(max_sample_len - 1):
    # Prepare input: Flatten the top tokens from the previous step
    # Shape: (batch_size * beam_size, 1)
    flat_input_ids = top_tokens.view(batch_size * beam_size, 1)

    # Generate next-token logits from the model
    flat_logits = model(flat_input_ids)
    flat_logprobs = F.log_softmax(flat_logits[:, 0, :], dim=-1)

    flat_states = current_transition_states.view(batch_size * beam_size)

    # Apply hybrid dense/sparse masking
    if step < d_dense - 1:
      # --- DENSE SPECIALIZATION ---
      # Reconstruct previous token from state ID (Valid for d_dense=2)
      parent_tokens = (flat_states - 1).long()
      masks = dense_mask[parent_tokens]

      flat_logprobs = torch.where(
          masks, flat_logprobs, torch.tensor(-float('inf'), device=device)
      )

      topk_logprobs, topk_indices = torch.topk(
          flat_logprobs, tokens_per_beam, dim=-1
      )

      # Map winners to next trie states using dense table
      next_state_candidates = dense_states[
          parent_tokens.unsqueeze(1), topk_indices.long()
      ]

      limit = tokens_per_beam
      candidates_logprobs, candidates_indices, candidates_states = (
          topk_logprobs,
          topk_indices,
          next_state_candidates,
      )
    else:
      # --- SPARSE CSR LOOKUP ---
      # Transition to CSR logic once the state space becomes too sparse
      limit = max_branch_factors[step + 1]
      candidates_logprobs, candidates_indices, candidates_states = (
          generate_and_apply_logprobs_mask(
              flat_logprobs,
              flat_states,
              packed_csr,
              csr_indptr,
              limit,
              vocab_size,
              device,
          )
      )

    # --- SCORE & BEAM UPDATE ---
    scores = current_token_scores.unsqueeze(2) + candidates_logprobs.view(
        batch_size, beam_size, limit
    )
    flat_scores = scores.view(batch_size, beam_size * limit)

    # Select the top beams for the next step
    top_scores, flat_top_indices = torch.topk(flat_scores, beam_size, dim=-1)

    # Recover token IDs and state transitions for the selected beams
    top_beam_indices = flat_top_indices // limit
    flat_tokens = candidates_indices.view(batch_size, beam_size * limit)
    flat_next_states = candidates_states.view(batch_size, beam_size * limit)

    top_tokens = _gather_beams(flat_tokens, flat_top_indices)
    current_transition_states = _gather_beams(
        flat_next_states, flat_top_indices
    )

    # Update history and scores
    token_buffer = _gather_beams(token_buffer, top_beam_indices)
    token_buffer[:, :, step + 1] = top_tokens
    current_token_scores = top_scores

  return token_buffer

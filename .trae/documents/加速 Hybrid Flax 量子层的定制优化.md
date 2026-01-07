## 性能诊断
- 计算复杂度瓶颈主要来自 Heisenberg 传播的基底规模：当 n_qubits=10、max_weight=6 时，TensorBasis 项数组合爆炸，导致每个门的稀疏 scatter/add 仍需在巨大的向量上操作（参考 [jit_propagation.py](file:///Users/lixin/PyPoli/src/pypoli/jit_propagation.py#L14-L68)）。
- 当前量子层每步仍对 9 个观测量做 vmap，重复执行相同的门序列（参考 [hybrid_flax.py](file:///Users/lixin/PyPoli/examples/hybrid_flax.py#L146-L166)）。尽管我们做了编译缓存，但运行阶段仍做 9 次传播，浪费计算。
- 使用 complex128 与 jax_enable_x64=True 增加计算与内存带宽压力（参考 [hybrid_flax.py](file:///Users/lixin/PyPoli/examples/hybrid_flax.py#L21-L22)）。

## 方案概述
- 将多观测量传播从 vmap 改为“批量传播一次”：重写 flat_step 支持 coeffs 形状为 (batch_obs, basis+1)，一次对所有观测量进行 scatter/add 广播，去掉 vmap 开销（参考 [jit_propagation.py](file:///Users/lixin/PyPoli/src/pypoli/jit_propagation.py#L337-L371)）。
- 降低数值精度以提升速度：改用 complex64 并关闭 x64，保持收敛稳定的同时大幅降低内存与算力负担（参考 [jit_propagation.py](file:///Users/lixin/PyPoli/src/pypoli/jit_propagation.py#L378-L401)、[hybrid_flax.py](file:///Users/lixin/PyPoli/examples/hybrid_flax.py#L21-L22)）。
- 控制基底规模：将 max_weight 默认调回 4，并提供“软截断”选项，用权重相关阈值抑制高权重项（在 jit 路径中引入 damping 与阈值，避免错误能量的同时减少向量尺寸）。
- 保持定制优化：量子层继续使用全局预编译的纯函数 compute_expectations（参考 [hybrid_flax.py](file:///Users/lixin/PyPoli/examples/hybrid_flax.py#L28-L89)），不在 Flax 层内嵌 JIT；必要时仅对该纯函数做一次全局 jit。

## 具体改动
- 重写 JITPropagator.make_flat_step 使其对 (B, basis) 广播：
  - 所有 current_coeffs.at[...] 操作以 axis=-1 为基础进行广播索引，确保批量观测量一次传播（参考 [jit_propagation.py](file:///Users/lixin/PyPoli/src/pypoli/jit_propagation.py#L337-L371)）。
- 将系数与掩码的 dtype 改为 complex64/bool，关闭 x64（参考 [jit_propagation.py](file:///Users/lixin/PyPoli/src/pypoli/jit_propagation.py#L378-L401)、[hybrid_flax.py](file:///Users/lixin/PyPoli/examples/hybrid_flax.py#L21-L22)）。
- 默认 max_weight=4，并加入 damping 软截断参数接口，使高权重项按权重指数阈值抑制（接口在 [jit_propagation.py](file:///Users/lixin/PyPoli/src/pypoli/jit_propagation.py#L273-L335) 的 step 中读取并应用）。
- 在 get_quantum_layer_fn 中移除 vmap，直接调用批量版 flat_step，一次返回 9 个期望值（参考 [hybrid_flax.py](file:///Users/lixin/PyPoli/examples/hybrid_flax.py#L28-L89)）。

## 验证与指标
- 运行 examples/hybrid_flax.py，比较改动前后：
  - 单步训练时间应从 ~9s 降至 ~2-3s（或更低，取决于硬件）。
  - 收敛能量曲线与 MSE 收敛速度保持稳定，无明显数值漂移。
- 运行 examples/simple_hea_vqe.py 验证 VQE 能量合理性，确保不再出现 -23 这类错误下限（参考 [simple_hea_vqe.py](file:///Users/lixin/PyPoli/examples/simple_hea_vqe.py)）。

## 回滚与参数
- 若精度不足：可临时启用 x64 或提升 max_weight 至 5，权衡速度与精度。
- 若速度仍瓶颈：进一步融合同一层的门（门融合）或减少深度 Q_DEPTH。

请确认以上计划；确认后我会一次性实施这些更改、跑两份示例并贴出耗时对比与结果。
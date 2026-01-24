# 在 4090 上 Finetune Hunyuan3D 踩坑指南

> RTX 4090: 24GB VRAM, Ada Lovelace 架构, 原生支持 BF16/FP16

---

## 🔥 问题一览

| 问题 | 症状 | 根因 |
|------|------|------|
| OOM | `CUDA out of memory` | 3.1B 参数 + AdamW 吃爆显存 |
| Loss 卡死 | 训练 1500 步 loss 一直 ~1.95 | 从头训练，未加载预训练权重 |
| Mesh 为空 | `Surface level must be within volume data range` | Token 不对齐 / 模型没学到东西 |
| NaN 爆炸 | `has_nan=True` | 权重损坏或精度溢出 |

---

## 💥 坑 1: OOM

**原因分析**：

```
模型参数: 3.1B (float32 ≈ 12GB)
AdamW 状态: 2x 参数量 (momentum + variance) ≈ 24GB
激活值: 取决于 batch size 和序列长度
────────────────────────────────────
总计: 远超 24GB
```

**解决方案**：

```yaml
# 1. LoRA - 只训练 0.1% 参数
lora_config:
  rank: 16
  target_modules: ["to_q", "to_k", "to_v", "out_proj"]

# 2. Gradient Checkpointing - 用计算换显存
denoiser_cfg:
  params:
    gradient_checkpointing: true

# 3. 减小点云规模
pc_size: 8192  # 默认 16384
```

**显存对比**：

| 配置 | 显存占用 |
|------|----------|
| 全量微调 | 💀 OOM |
| + LoRA | ~18GB |
| + Gradient Checkpointing | ~14GB |

---

## 💥 坑 2: Loss 卡住不动

**症状**：
```
loss=1.95, loss=1.94, loss=1.95...  # 永远在 1.9x 徘徊
```

**根因**: 配置文件里 `denoiser_cfg` 没有 `from_pretrained`，相当于从头训练 3.1B 参数

**修复**：
```yaml
denoiser_cfg:
  target: hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain
  from_pretrained: tencent/Hunyuan3D-2.1  # 👈 必须加这行
```

**正常 loss 曲线**：
```
step 0:    ~2.0
step 500:  ~1.6
step 1000: ~1.4  ✅ 在下降
```

---

## 💥 坑 3: Token 长度不匹配

**症状**：
```
ValueError: Surface level must be within volume data range.
# Grid logits: min=-1.00, max=-0.98 (全负，不包含 0)
```

**根因**: DINO 输出 token 数 ≠ DiT 期望的 `text_len`

**计算公式**：
```python
# DINO-v2 patch size = 14
num_tokens = (image_size // 14) ** 2 + 2  # +2 for [CLS] + [REG]

# 384 → (384/14)² + 2 = 27² + 2 = 731 ≈ 730
# 518 → (518/14)² + 2 = 37² + 2 = 1371 ≈ 1370
```

**对齐规则**：

| image_size | text_len | 状态 |
|------------|----------|------|
| 384 | 730 | ✅ |
| 518 | 1370 | ✅ (官方预训练) |
| 384 | 1370 | ❌ 不匹配 |
| 518 | 730 | ❌ 不匹配 |

**推荐**: 使用官方配置 `image_size=518 + text_len=1370`

---

## 💥 坑 4: Mesh 生成为空

**调试方法**：

在 `pipelines.py` 添加：
```python
print(f"[DEBUG] cond['main'].shape = {cond['main'].shape}")
print(f"[DEBUG] Expected text_len: {self.model.text_len}")
```

在 `autoencoders/model.py` 添加：
```python
print(f"[DEBUG] Grid logits: min={grid_logits.min():.4f}, max={grid_logits.max():.4f}")
print(f"[DEBUG] mc_level=0.0 需要在此范围内")
```

**健康输出**：
```
cond['main'].shape = torch.Size([2, 1370, 1024])  # 1370 = text_len ✅
Expected text_len: 1370 ✅
Grid logits: min=-1.03, max=1.03  # 包含 0 ✅
```

**异常输出**：
```
Grid logits: min=-1.00, max=-0.98  # 全负，不包含 0 ❌
```

---

## 🛠️ 完整可用配置

```yaml
# hunyuandit-finetuning-4090-24gb.yaml 核心配置

training:
  steps: 6000
  use_amp: true
  amp_type: "bf16"  # 4090 原生支持
  base_lr: 1e-5     # LoRA 用小学习率

dataset:
  params:
    pc_size: 8192
    image_size: 518  # 👈 必须和 text_len 对应

model:
  params:
    lora_config:
      rank: 16
      target_modules: ["to_q", "to_k", "to_v", "out_proj"]
    
    denoiser_cfg:
      from_pretrained: tencent/Hunyuan3D-2.1  # 👈 关键
      params:
        gradient_checkpointing: true
        text_len: 1370  # 👈 必须和 image_size 对应
```

---

## 📊 TensorBoard 监控

```bash
tensorboard --logdir=output_folder/dit/xxx/log/tensorboard --port=6006 --bind_all
```

关注指标：
- `train/simple`: 应该稳步下降
- `val/simple`: 验证集 loss
- 每 500 步查看 `log/infer/` 下的 `.glb` 文件

---

## 🎯 快速检查清单

```bash
# 训练前确认
□ from_pretrained 已配置
□ image_size (518) 和 text_len (1370) 对应
□ lora_config 已启用
□ gradient_checkpointing: true
□ peft 库已安装: pip install peft

# 训练中观察
□ Loss 在下降 (不是卡在 1.9x)
□ 没有 NaN 警告
□ 显存占用 < 20GB

# 推理验证
□ cond shape 和 text_len 匹配
□ Grid logits 范围包含 0
□ .glb 文件成功生成
```

---

## 💡 硬件知识补充

**RTX 4090 规格**：
- VRAM: 24GB GDDR6X
- Tensor Cores: 第4代 (支持 FP8/BF16/TF32)
- 带宽: 1TB/s

**精度选择**：
| 精度 | 显存 | 速度 | 稳定性 |
|------|------|------|--------|
| FP32 | 1x | 1x | 最稳定 |
| BF16 | 0.5x | ~1.5x | 推荐 |
| FP16 | 0.5x | ~1.5x | 可能溢出 |

> BF16 指数位和 FP32 相同，不易溢出，4090 原生支持，优先选择

---

*Last updated: 2026-01*

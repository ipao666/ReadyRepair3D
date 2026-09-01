# R3DGuard 远程环境记录

更新时间：2026-07-14

## 服务器资源

- Ubuntu 22.04
- 1 × NVIDIA A100 PCIe 40GB
- 90GB 内存
- 200GB 系统盘
- NVIDIA Driver 570.211.01
- CUDA Toolkit / NVCC 12.4

> 实际资源与最初设想的双 A100、500GB 磁盘不同。实验按单卡和 200GB 存储约束执行。

## 连接

本机已配置独立 SSH 密钥和别名：

```powershell
ssh r3dguard
```

密钥登录已验证；项目脚本和配置中未保存 SSH 密码。

## 目录

```text
/root/r3dguard/
├── activate.sh
├── data/
├── logs/
├── models/
├── outputs/
└── repos/TRELLIS.2/
```

进入环境：

```bash
source /root/r3dguard/activate.sh
```

该脚本会激活 `trellis2` Conda 环境，并设置 CUDA、TRELLIS.2、清华 PyPI 与 Hugging Face 镜像相关变量。

## Python 与 CUDA 依赖

- Miniforge：`/opt/conda`
- Conda 环境：`trellis2`
- Python 3.10.20
- PyTorch 2.6.0+cu124
- torchvision 0.21.0+cu124
- diffusers 0.39.0
- FlashAttention 2.7.3
- nvdiffrast 0.4.0
- nvdiffrec_render 0.0.0
- CuMesh 0.0.1
- FlexGEMM 1.0.0
- o-voxel 0.0.1

PyTorch 已完成 A100 FP16 CUDA 矩阵乘法测试；上述 CUDA 扩展均已完成导入测试。

## 已准备模型

| 模型 | 目录 | 用途 | 状态 |
|---|---|---|---|
| TRELLIS.2-4B | `/root/r3dguard/models/TRELLIS.2-4B` | 单图到带纹理 3D | 已下载 |
| TRELLIS image sparse decoder | `/root/r3dguard/models/TRELLIS-image-large` | TRELLIS 稀疏结构解码 | 已下载 |
| SANA 1.5 1.6B | `/root/r3dguard/models/SANA1.5_1.6B_1024px_diffusers` | 质量优先生图 | 已下载并完成 1024px 推理 |
| DINOv2-L | `/root/r3dguard/models/dinov2-large` | Ready3D 图像特征 | 已下载并离线加载 |
| Depth Anything V2-L | `/root/r3dguard/models/Depth-Anything-V2-Large-hf` | 深度与结构特征 | 已下载并离线加载 |
| SAM2 Hiera Large | `/root/r3dguard/models/sam2-hiera-large` | 前景分割 | 已下载并离线加载 |
| BiRefNet | `/root/r3dguard/models/BiRefNet` | 公开抠图备用模型 | 已下载并离线加载 |
| SDXL Base 1.0 FP16 | `/root/r3dguard/models/SDXL-base-1.0` | 生图基线 | 后台下载中 |

SANA 质量样张：`/root/r3dguard/outputs/sample_sana_1024.png`。

## 当前外部阻塞

TRELLIS.2 的图像条件编码器固定使用 `facebook/dinov3-vitl16-pretrain-lvd1689m`。该模型是 Hugging Face/Meta 受限权重，必须由用户接受许可并提供具有访问权限的只读 Hugging Face Token。`briaai/RMBG-2.0` 同样需要审批，当前已用公开 BiRefNet 准备替代路径。

本地 TRELLIS 配置位于：

```text
/root/r3dguard/models/TRELLIS.2-4B/pipeline.local.json
```

DINOv3 获批后下载到 `/root/r3dguard/models/dinov3-vitl16-pretrain-lvd1689m`，即可继续做 TRELLIS 完整端到端推理测试。

## 下载日志

```text
/root/r3dguard/logs/model_download.log
/root/r3dguard/logs/sdxl_download.log
```

# G0 正式光纤分割工程

本目录只有一条可运行的正式链路：原始视频 → 固定两网 Raw2 → 固定内部前处理 → G0 无提案回退几何 → 原生分辨率标签掩膜与同步对比视频。

内部前处理是现有 G0 结果的必要输入，不作为独立算法、对照臂或输出暴露。工程没有 RGB/R/S、弱蓝对照、旧版 G0 回退等可选分支。

## 运行

在本目录执行：

```powershell
python -m pip install -r requirements.txt
python run_g0.py videos/video_1.mp4 outputs/video_1
python run_g0.py videos/video_2.mp4 outputs/video_2
```

默认优先使用 CUDA；没有 CUDA 时可显式使用 `--device cpu`。输出目录必须不存在或为空。

## 输出

- `masks/000000.png`：逐帧单通道标签图，`0=背景、1=结石、2=光纤`。
- `frames.jsonl`：逐帧掩膜哈希、G0 状态、原因和时间记录。
- `comparison.mp4`：1920×820 的原视频、Raw2、G0 三栏同步对比；G0 新增/删除区域以 Raw2 为基准填色，并显示滚动实测帧率。
- `manifest.json`：输入、模型、配置哈希、运行环境及对比视频哈希记录。

对比视频使用输入视频报告的平均帧率进行恒定帧率编码；它与逐帧掩膜一一对应，但不声明保留源文件的可变 PTS。

## 固定资产

- `models/`：G0 所需的两份原始冻结检查点；运行时只构建两个网络。
- `configs/`：复现现有 G0 输入和无回退几何所需的冻结参数。
- `videos/`：项目中的两段原始视频及其哈希清单。
- `tools/`：G0 主线的内部实现；没有其他实验 runner、训练代码或评估分支入口。

`requirements.txt` 记录当前验证环境的直接运行依赖。模型和视频在加载/审计时使用 SHA-256 校验。

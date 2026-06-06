# TransSAM 第五个实验数据级复现

对应论文：
- `6.5 架构消融与关键参数敏感性分析`
- `6.5.1 核心组件消融验证`
- `6.5.2 跨域泛化与少样本适应能力分析`
- `Table 7`
- `Table 8`

本目录只负责第五个实验，不和 [project_1](D:/lunwen/project_1)、[project_2](D:/lunwen/project_2)、[project_3](D:/lunwen/project_3) 或 [project_4](D:/lunwen/project_4) 混用输出。

## 当前实现

- 五个真实数据集参与评估
- `6.5.1` 组件消融：
  - `TransSAM (Ours)`
  - `w/o SAM (1D Seq)`
  - `w/o ViT (ResNet)`
  - `w/o LWED`
- `6.5.1` 指标：
  - `F1`
  - `AUC`
- `6.5.1` 条件：
  - `Clean`
  - `Obfuscated-0.9`
- `6.5.2` 跨域少样本适应：
  - `shot = 1 / 3 / 5 / 10`
  - `metric = F1`
  - `method = TransSAM`

## 运行

```powershell
cd D:\lunwen\project_5
python project_5.py --quick
```

标准版：

```powershell
cd D:\lunwen\project_5
python project_5.py
```

## 输出

- [project_5_results.scv](D:/lunwen/project_5/project_5_results.scv)
- [project_5_results.json](D:/lunwen/project_5/project_5_results.json)
- [project_5_manifest.json](D:/lunwen/project_5/project_5_manifest.json)

## 使用边界

这是第五个实验的 `real-data proxy reproduction`。

推荐表述：

`We implement a real-data proxy reproduction of Experiment 5, including the component ablation study and the cross-dataset few-shot adaptation analysis, on the five public datasets available in the workspace.`

不建议表述：

`We exactly reproduced the official training pipelines of all architectural variants.`

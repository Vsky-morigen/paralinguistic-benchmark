# 数据格式

`configs/dataset.yaml` 默认从 `data/audio/` 读取录音，从 `data/metadata/` 读取题目和参考标注。所有路径相对于项目根目录。

| 文件 | 内容 |
|---|---|
| samples.jsonl | 每行一条音频的文本和参考标签 |
| choices.json | 固定类别选项，以及按台词编号分组的情境选项 |
| manifest.json | 数据版本、数量、语言分布、音频格式和文件校验值 |

## 样本字段

| 字段 | 含义 |
|---|---|
| schema_version | 当前为 1 |
| sample_id / language | 唯一编号及 zh / en |
| text | 完整台词，仅重音、停顿任务向模型提供其无标记版本 |
| targets.gender / emotion / context | 声音性别、情绪和完整情境意图 |
| targets.speed / speed_verified | 语速类别及是否验证；未验证不评分 |
| targets.stress | status、display、spans；参考重音字符区间 |
| targets.pause | status、display、positions；参考句内停顿边界 |
| reference_audio / reference_sha256 | 相对音频文件名及 SHA-256 |
| provenance.text_id / version | 台词编号及表达版本 |
| provenance.sentence_boundaries | 不计分的句间边界，仅用于评分 |

参考标签只供评分。`status: unknown` 不评分；`status: annotated` 且数组为空表示明确没有。字符坐标规则见[评分规则](evaluation_protocol.md)。

## 候选集合

`choices.json` 的 `labels` 包含 `gender`、`emotion`、`speaking_rate` 类别数组，`context` 按 `text_id` 提供不同表达意图。每个集合至少有两个不同选项，并包含对应参考答案；语速标签在验证后须与其类别集合一致。英文题的基础类别由任务代码映射为英文标签。

情境选项应对应同一台词下合理但不同的意图，避免仅凭不同话题即可排除。修改选项后需重新核查题目，并更新清单中的校验值。

`parabench dataset validate` 检查文件完整性、题号、计数、位置范围、音频格式和候选集合。划分开发集和测试集时按 `text_id` 分组。

## 转换自己的录音

```bash
python -m pip install -e ".[audio]" -c requirements-dev.lock.txt
python scripts/normalize_audio.py --input-dir data/raw --output-dir data/converted --sample-rate 48000
```

转换输出为单声道 PCM16 WAV，不裁剪静音、不降噪、不调整响度；双声道左右各 0.5 混合。输出目录须为空。替换音频后建立新的数据版本并更新音频校验值。

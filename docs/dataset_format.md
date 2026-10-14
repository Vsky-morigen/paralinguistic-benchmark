# 数据格式

默认标签位于 `data/metadata/`，参考录音位于 `data/audio/`。两者由 `configs/dataset.yaml` 指定，路径相对于项目根目录。

`samples.jsonl` 每行一个 JSON 对象：

| 字段 | 含义 |
|---|---|
| schema_version | 当前为 1 |
| sample_id | 唯一题号，例如 AUDIO_003 |
| language | zh 或 en |
| text | 完整朗读文本，不含表达标记 |
| targets.gender / emotion / context | 声音性别、情绪和情境意图 |
| targets.speed / speed_verified | 语速要求及是否经过验证；未验证时不评分 |
| targets.stress | status、display、spans；spans 为重音字符区间 |
| targets.pause | status、display、positions；positions 为句内停顿位置 |
| reference_audio | 相对于音频目录的文件名 |
| reference_sha256 | 参考音频的 SHA-256 |
| provenance.text_id / version | 台词编号及表达版本 |
| provenance.sentence_boundaries | 不计分的句间边界 |

## 位置坐标

规范化依次执行 Unicode NFKC、保留字母数字、转小写。中英文均采用规范化后的**字符位置**，不按词切分。

重音 `spans` 使用左闭右开区间 `[start,end)`。停顿 `positions` 表示位于第几个规范化字符之后。例如“你好，再见。”规范化为“你好再见”，重音 `[0,2]` 对应“你好”，停顿 `2` 对应“你好｜再见”。

`status: annotated` 且位置数组为空表示明确没有；`status: unknown` 表示未知，不评分。句间边界和首尾静音不计入停顿评分。

## 校验与更新

`manifest.json` 保存数据版本、题目数量、语言分布、标签文件校验值和音频格式。`parabench dataset validate` 检查题号、计数、位置范围及音频完整性。修改数据时建立新版本，同时更新清单；划分开发集和测试集时按 `text_id` 分组，避免相同台词跨集合。

需要转换自己的参考录音时：

```bash
python -m pip install -e ".[audio]" -c requirements-dev.lock.txt
python scripts/normalize_audio.py --input-dir data/raw --output-dir data/converted --sample-rate 48000
```

转换器输出单声道 PCM16 WAV，不裁剪静音、不降噪、不调整响度；双声道按左右各 0.5 混合。输出目录必须为空。替换参考音频后须同步建立新的标签与校验清单。

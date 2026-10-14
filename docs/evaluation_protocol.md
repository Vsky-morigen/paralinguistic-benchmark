# 评分说明

模型接收台词和表达要求后生成语音。评分证据必须来自**本次生成音频**，参考标签仅用于比较。默认检查全部六个副语言维度及文本正确性。

## 各维度

| 维度 | 分数 | 通过条件 |
|---|---|---|
| 文本正确性 | 中文字符错误率、英文单词错误率；分数为 `max(0, 1-错误率)` | 错误率 ≤ `content_max_error_rate`，默认 0.1 |
| 声音性别 | 预测标签与目标一致为 1，否则为 0 | 标签一致 |
| 情绪 | 预测标签与目标一致为 1，否则为 0 | 标签一致 |
| 情境意图 | 预测的完整意图标签与目标一致为 1，否则为 0 | 标签一致 |
| 重音 | 规范化文本字符位置的 F1 | 位置完全一致，无多标、无漏标 |
| 句内停顿 | 排除句间及首尾边界后，文本边界位置的 F1 | 位置完全一致，无多标、无漏标 |
| 语速 | 测得语速落在经过验证的目标区间内为 1，否则为 0 | 落在目标区间内；当前数据未验证，暂不评分 |

声音性别、情绪和情境意图可使用外部分类器或音频模型裁判。人类盲听单独计为“符合目标人数／听者总人数”，至少两名不同听者符合才通过。声音性别描述听感。

## 重音和停顿

字符位置的规范化与示例见[数据格式](dataset_format.md)。重音区间展开为字符位置集合，停顿使用文本边界集合；重复或重叠位置只计一次。句间边界来自 `provenance.sentence_boundaries`，不参与停顿评分。

- TP：正确命中的位置数。
- FP：多标的位置数。
- FN：漏标的位置数。

`精确率 = TP / (TP + FP)`，`召回率 = TP / (TP + FN)`，`F1 = 2TP / (2TP + FP + FN)`。

例如，10 个字符中只有 2 个需要重音，把全句标成重音会得到 TP=2、FP=8、FN=0，F1 约为 0.333。目标有 2 处句内停顿，预测全部命中但多出 6 处，F1 为 0.4。

目标和预测都明确没有位置时，精确率、召回率、F1 均为 1；仅一方为空时 F1 为 0，除数为 0 的精确率或召回率记 0。目标未知时不评分。重音采用字符单位；停顿按相同边界严格匹配，不对时长评分，也不设置毫秒容差。

## 证据文件

`evidence.jsonl` 每行对应一条生成音频。`audio_sha256` 取自本次 `records/<sample_id>.json`，其余字段来自外部识别器、对齐器或盲听结果：

```json
{
  "sample_id": "AUDIO_003",
  "audio_sha256": "本次生成音频的64位SHA-256",
  "dimensions": {
    "content": {
      "method": "independent_asr",
      "evaluator_id": "asr-model-version",
      "transcript": "实际识别出的文本"
    },
    "emotion": {
      "method": "classifier",
      "evaluator_id": "emotion-model-version",
      "calibration_id": "validation-version",
      "label": "中性"
    },
    "stress": {
      "method": "acoustic",
      "evaluator_id": "stress-extractor-and-aligner-version",
      "calibration_id": "threshold-version",
      "spans": [[0, 2]]
    },
    "pause": {
      "method": "acoustic",
      "evaluator_id": "pause-detector-and-aligner-version",
      "calibration_id": "threshold-version",
      "positions": [2]
    }
  }
}
```

以上坐标仅演示格式。`gender` 和 `context` 使用与 `emotion` 相同的标签结构。音频模型裁判使用 `method: audio_model_judge`。分类器和声学方法须填写 `calibration_id`；该字段只记录工具或阈值的校准版本。

人类盲听证据放入相应的 `gender`、`emotion` 或 `context` 字段：

```json
{
  "method": "human_blind",
  "evaluator_id": "blind-study-v1",
  "votes": [
    {"rater_id": "listener-a", "label": "中性"},
    {"rater_id": "listener-b", "label": "中性"},
    {"rater_id": "listener-c", "label": "无法判断"}
  ]
}
```

本工程不内置 ASR、分类器、音频裁判或重音／停顿提取器。使用前应固定工具版本、提示词和声学阈值，并验证其准确性。文本正确性仅接受独立 ASR 转写。

## 报告与续跑

- `scored`：证据有效，计算分数。
- `skipped`：缺少证据、标签未知、维度关闭或使用 Mock；分数为空。
- `error`：音频或证据无效；分数为空。

报告按语言、维度、证据方法分别汇总平均分、通过率和已评分覆盖率。重音和停顿先计算每条 F1 再取平均，同时列出精确率、召回率和逐条 TP／FP／FN。未评分和错误不当作 0 分，不输出跨维度总分。

每次运行冻结题目、配置和提示版本；`--resume` 要求这些条件一致。音频重新生成后必须重新获取证据并评分。位置计分版本为 `prosody-position-f1-v2`。

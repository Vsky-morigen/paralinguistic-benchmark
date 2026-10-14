# 副语言理解 Benchmark

评测模型能否理解**输入音频**中的声音性别、情绪、重音、句内停顿、语速和情境意图，支持中文和英文。模型听录音并返回识别结果，程序与经过人类验证的参考标注比较。默认启用全部六个维度。

| 任务 | 模型输入 | 模型输出 |
|---|---|---|
| 声音性别、情绪、语速、情境意图 | 音频＋问题＋候选标签 | 一个候选标签 |
| 重音、句内停顿定位 | 音频＋问题＋无标记台词 | 重音区间或停顿位置数组 |

台词只用于位置定位，已删除重音符号、停顿符号和标点。参考答案不发送给模型；无需先完成转写。

## 安装

需要 Python 3.11 或以上。在项目根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e . -c requirements-dev.lock.txt
```

## 准备音频

把配套压缩包 `paralinguistic-audio-v1.zip` **解压到 `data` 文件夹**，得到 `data/audio/*.wav`。题目和标签已随代码保存在 `data/metadata/`，不要把音频再嵌套一层目录。

```bash
parabench dataset validate
```

音频统一为 48 kHz、单声道、PCM16 WAV。所有配置路径均相对于项目根目录。

## 接入与运行

- **API 服务**：修改 `configs/models/http_example.yaml` 的服务地址和模型名称。密钥通过环境变量 `PARABENCH_API_KEY` 提供；服务须符合[HTTP 接口协议](docs/model_integration.md)。
- **本地模型**：修改 `configs/models/local_example.yaml` 的权重路径、设备和加载函数。参照[本地接入指南](docs/local_models.md)对接模型 SDK。

```bash
# 调用模型理解输入音频
parabench predict --config configs/models/http_example.yaml --run-id model-a-001

# 直接根据模型回答评分
parabench evaluate --run results/model-a-001

# 导出报告
parabench report --run results/model-a-001
```

本地模型把配置替换为 `configs/models/local_example.yaml`。检查流程可使用 `configs/models/mock.yaml`，模拟回答不产生正式成绩。

结果保存在 `results/model-a-001/`：`report.md` 为汇总、`details.csv` 为逐条分数、`records/` 为模型回答、`requests/` 为题目、`inputs/` 为本次输入音频副本。

## 评分与选项

类别识别按标签准确率计分；重音、停顿按位置 F1 计分，多标与漏标都会扣分，句间停顿不计分。位置任务明确属于**有台词辅助的理解测评**。当前数据的语速标签尚未验证，该项显示未评分。

拒答、漏答和无效格式按 0 分；调用失败单列为错误，并按 0 计入有效题目的主分数。未知参考标签、关闭维度和模拟预测不计分。分数按语言、维度分别报告，不输出跨维度总分。[详细评分规则](docs/evaluation_protocol.md)

- `--resume`：断点续跑，保留已完成且校验通过的回答。
- `--language zh` 或 `--language en`：选择语言。
- `--limit 2`：先测试少量音频；全量运行使用新的运行编号。
- `configs/evaluation.yaml`：启用或关闭维度，默认全部启用。

[数据格式](docs/dataset_format.md)

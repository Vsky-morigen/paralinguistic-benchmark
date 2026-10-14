# 副语言表达 Benchmark

评测语音模型能否按要求表达**声音性别、情绪、重音、句内停顿、语速和情境意图**，支持中文和英文。输入台词与表达要求，模型生成语音，结果按语言和维度分别统计。默认启用全部六个副语言维度，并单独检查文本正确性。

## 安装

需要 Python 3.11 或以上。在项目根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e . -c requirements-dev.lock.txt
```

## 准备音频

将配套音频压缩包 `paralinguistic-audio-v1.zip` **解压到项目的 `data` 文件夹中**。解压后的结构如下，不要再嵌套一层压缩包名称：

```text
data/
├── metadata/
│   ├── manifest.json
│   └── samples.jsonl
└── audio/
    ├── AUDIO_003.wav
    └── ...
```

参考音频统一为 48 kHz、单声道、PCM16 WAV。默认从 `data/audio/` 读取，配置路径均相对于项目根目录。检查音频及标签：

```bash
parabench dataset validate
```

## 接入模型

- **API 服务**：修改 `configs/models/http_example.yaml` 的服务地址、模型名称和生成参数；密钥通过环境变量 `PARABENCH_API_KEY` 提供。服务须符合[HTTP 接口协议](docs/model_integration.md)。
- **本地模型**：修改 `configs/models/local_example.yaml`，填写相对权重路径、设备和加载函数。参照[本地模型指南](docs/local_models.md)对接模型 SDK，加载器模板位于 `examples/local_model_loader.py`。

## 运行

```bash
# 生成语音；本地模型换用 configs/models/local_example.yaml
parabench generate --config configs/models/http_example.yaml --run-id model-a-001

# 使用本次生成音频的识别、对齐或盲听结果评分
parabench evaluate --run results/model-a-001 --evidence results/model-a-001/evidence.jsonl

# 导出汇总和逐条结果
parabench report --run results/model-a-001
```

工程提供生成接口与计分器；ASR、情绪等识别器及重音／停顿检测器需自行接入，将结果写入 `evidence.jsonl`。字段及公式见[评分说明](docs/evaluation_protocol.md)。缺少证据或有效参考标签时显示“未评分”；当前数据的语速标签尚未验证，该项暂不评分。

结果保存在 `results/model-a-001/`：`report.md` 为汇总，`details.csv` 为逐条分数，`audio/` 为模型生成音频。参考录音不作为模型输入。

## 常用选项

- `--resume`：断点续跑。
- `--language zh` 或 `--language en`：选择语言。
- `--limit 2`：先跑少量题目；全量运行使用新的运行编号。
- `configs/evaluation.yaml`：调整评分维度，默认全部启用。
- `configs/models/mock.yaml`：检查流程；生成测试音，不产生正式成绩。

分数范围为 0–1。重音和停顿按位置 F1 计分，多标和漏标都会扣分；句间停顿不计分。不同评分方法分别报告，不混算总分。数据字段见[数据格式](docs/dataset_format.md)。

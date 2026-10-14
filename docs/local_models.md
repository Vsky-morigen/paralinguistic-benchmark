# 本地模型接入

`adapter: local` 在本机Python进程中加载模型权重并推理，不需要部署HTTP服务。由于语音模型SDK不同，使用者提供一个加载函数和一个统一的生成方法；无需修改benchmark核心代码。

## 配置与运行

复制并修改 `configs/models/local_example.yaml`：

```yaml
schema_version: 1
adapter: local
model: my-speech-model
model_path: weights/my-model
loader: my_loader.py:load_model
device: cuda:0
concurrency: 1
max_attempts: 1
load_kwargs: {}
generation:
  temperature: 0
  seed: 20261014
```

`model_path` 为已有权重文件或非空目录；路径相对于工程根目录。`loader` 支持 `文件.py:函数` 或已安装的 `包.模块:函数`，文件路径也相对于工程根目录。设备值支持auto、cpu、cuda、cuda:N、mps；auto的具体选择由加载器负责。

安装对应模型的SDK、PyTorch及硬件依赖后运行：

```powershell
parabench generate --config configs/models/local_example.yaml --run-id local-a-001
parabench evaluate --run results/local-a-001 --evidence results/local-a-001/evidence.jsonl
parabench report --run results/local-a-001
```

## 加载器接口

以 `examples/local_model_loader.py` 为模板，接上自己模型的SDK。

```python
def load_model(*, model_path: Path, device: str, **load_kwargs):
    # 调用模型SDK，加载权重；返回带generate方法的对象。
    return MyModel(...)


class MyModel:
    def generate(self, request: GenerationRequest) -> LocalAudio:
        # 传入request.text、request.language、request.instructions、request.generation。
        # 不支持表达指令的模型也应明确记录限制，不能假装已使用该指令。
        return LocalAudio(sample_rate=24000, samples=mono_float_audio)
```

加载函数在一次运行中只执行一次。模型推理和资源释放串行运行在同一个工作线程，防止共享模型并发推理。`load_kwargs` 传给加载器，`generation` 随每个请求交给模型；CPU/GPU推理由模型SDK负责。

返回值二选一：

- `LocalAudio(sample_rate=24000, samples=...)`：单声道、有限数值、范围[-1,1]，列表或可转换为列表的一维数组。框架保存为PCM16 WAV。张量应先由加载器转成单声道CPU数组。
- `LocalAudio(sample_rate=24000, audio_path="模型生成的文件.wav")`：已有PCM WAV文件，采样率须与文件头一致。框架校验并复制到运行目录，不修改原文件。

可选实现 `close()` 释放资源。测试音必须返回 `simulated=True`，不会获得正式评分。加载或推理失败会记录为失败，不回退到Mock。

## 运行约束

- 本地配置要求 `concurrency: 1`，多张GPU可使用不同设备和运行编号分别启动进程。
- `timeout_seconds` 的网络超时不强行中断进程内的CPU/GPU推理；需要硬超时或进程隔离时，可通过HTTP适配器调用独立推理服务。
- 续跑比较配置、加载器文件SHA-256及权重目录的文件名／大小／修改时间指纹。权重元数据指纹不是整个权重的内容校验和；使用者仍应固定权重和依赖版本。模块形式的加载器应固定包版本或在配置中增加版本标识。
- 通用接口不自动识别任意权重格式。示例加载器需要接上模型SDK后才能运行，未实现的模板会明确失败。

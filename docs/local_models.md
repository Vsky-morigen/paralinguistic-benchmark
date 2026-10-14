# 本地模型接入

直接在本机加载音频理解模型，一次运行加载一次，在同一工作线程中依次推理各题。

1. 安装模型 SDK 及硬件依赖。
2. 以 `examples/local_model_loader.py` 为模板实现加载器。
3. 修改 `configs/models/local_example.yaml`，运行：

```bash
parabench predict --config configs/models/local_example.yaml --run-id local-a-001
parabench evaluate --run results/local-a-001
parabench report --run results/local-a-001
```

## 加载器

```python
def load_model(*, model_path: Path, device: str, **load_kwargs):
    return YourModel(...)


class YourModel:
    def predict(self, request: UnderstandingRequest) -> object:
        # SDK 读取 request.audio_path，接收 question、choices、parameters。
        # request.transcript 仅在重音、停顿定位时有值。
        # 将模型回答解析为一个候选标签，或位置数组。
        return parsed_answer

    def close(self): ...
```

`model_path` 是相对于项目根目录的权重文件或非空目录。`loader` 支持 `文件.py:函数` 或已安装的 `包.模块:函数`；文件路径也相对于项目根目录。

`device` 支持 `auto`、`cpu`、`cuda`、`cuda:N`、`mps`，具体加载方式由 SDK 决定。`load_kwargs` 传给加载器，`parameters` 随每个问题传入。默认示例需要接上实际 SDK 后才能运行。

本地配置要求 `concurrency: 1`。网络超时参数不强行中断进程内 CPU/GPU 推理；需要隔离和硬超时时，可改用独立 HTTP 推理服务。

续跑检查配置、加载器文件校验值和权重文件元数据指纹。权重元数据指纹包含文件名、大小和修改时间，不等于完整权重内容校验；请固定权重及 SDK 版本。配置和运行记录保留相对权重路径。

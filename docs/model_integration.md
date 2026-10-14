# 模型接入

支持两种方式：远程／本地HTTP服务使用 `adapter: http`；直接加载本地权重使用 `adapter: local`，见[本地模型接入](local_models.md)。通用本地加载器无需在适配器注册表中为每个模型再增加名字。

## Python接口

```python
class SpeechModelAdapter(ABC):
    async def generate(self, request: GenerationRequest, output_path: Path) -> GenerationResult: ...

    async def close(self) -> None: ...
```

`GenerationRequest` 包含 `sample_id`、`text`、`language`、`instructions`、`generation`。不包含参考录音、参考路径、盲听者答案或替代情境选项。

适配器必须把最终音频写到指定 `output_path`；返回实际采样率、耗时、模型名称及可选用量。音频为完整PCM WAV，1或2声道，采样位宽8/16/24/32位，采样率8–192 kHz，时长不超过600秒。适配器负责处理厂商的流式片段或压缩格式；运行器不会偷偷转码或替换成Mock。

临时性失败抛出 `ModelError("已去敏的原因", retryable=True)`；参数错误、鉴权失败和无效音频应设为False。错误消息不应包含密钥或服务原始响应。需要长连接的适配器在 `close()` 释放连接。

新增适配器步骤：实现 `models/<provider>.py`，在 `models/registry.py` 注册，同时在 `config.py` 的允许列表中加入该名字；增加配置、协议测试和README接入说明。无需更改数据、评分和报告逻辑。

## HTTP桥接协议

向配置的 `endpoint` 发送POST JSON；有 `api_key_env` 时，从该环境变量读取密钥并发送 `Authorization: Bearer ...`。

请求：

```json
{
  "request_id": "AUDIO_003",
  "model": "your-model-name",
  "input": {
    "text": "需要朗读的完整台词",
    "language": "zh",
    "instructions": "声音性别、情绪、重音、句内停顿及完整表达意图"
  },
  "generation": {"temperature": 0, "seed": 20261014}
}
```

成功响应：

```json
{
  "format": "wav",
  "audio_base64": "完整WAV文件的Base64字符串",
  "sample_rate": 16000,
  "usage": {"input_tokens": 100, "output_tokens": 200, "audio_seconds": 4.5}
}
```

`sample_rate` 必须与WAV头一致；Base64需合法。只保存允许的数值用量字段，不保存响应原文或任意厂商元数据。默认响应大小上限32 MiB，可配置 `max_response_bytes`。不跟随HTTP重定向，也不下载返回的音频URL。

HTTP 408、429、5xx以及网络超时可重试；其他HTTP失败和无效响应不重试。指数退避由 `retry_backoff_seconds`、`max_attempts` 控制。并发由 `concurrency` 控制。

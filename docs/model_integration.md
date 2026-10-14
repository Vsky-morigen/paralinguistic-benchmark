# 模型接入协议

模型接收已有音频并回答问题。HTTP 服务和本地加载器使用相同任务字段，每条音频的各维度分别发起请求，不共享参考答案或其他题目的回答。

## HTTP

向配置的 `endpoint` 发送 POST JSON：

```json
{
  "model": "your-model",
  "input": {
    "request_id": "AUDIO_003.emotion",
    "dimension": "emotion",
    "language": "zh",
    "audio_sha256": "输入音频的64位SHA-256",
    "audio": {"format": "wav", "data_base64": "完整WAV文件的Base64"},
    "question": "根据实际语气，选择说话者的情绪。只返回一个候选标签。",
    "choices": ["中性", "愉悦", "恼怒", "悲伤", "担忧"],
    "parameters": {"temperature": 0}
  }
}
```

只有 `stress`、`pause` 请求包含 `transcript`，内容为无标记、无标点台词；这两种请求的 `choices` 为空。`question` 指定规范化字符坐标和返回格式。

成功响应必须含 `answer`：

```json
{"answer": "中性", "usage": {"input_tokens": 100, "output_tokens": 5}}
```

| dimension | answer 格式 |
|---|---|
| gender / emotion / speaking_rate / context | 字符串，与一个候选标签一致 |
| stress | JSON 数组，例如 `[[0,2],[5,7]]` |
| pause | JSON 数组，例如 `[2,5]` |

重音或停顿明确没有时返回 `[]`。数组必须作为 JSON 数组返回，不能包成字符串。中文分类题使用中文标签；英文题的基础类别使用英文标签，情境选项使用对应英文描述。均以本次请求的 `choices` 为准。

这是统一桥接协议；厂商原生 API 需要由服务或自定义适配器转换请求和响应，并把模型原始回答解析为上述结构。服务应将实际音频交给模型，不能只提交文件名或用文字替代音频。

配置有 `api_key_env` 时，通过该环境变量读取密钥，发送 `Authorization: Bearer ...`。不自动读取 `.env`。HTTP 408、429、5xx 和网络超时可重试；鉴权错误、其他 HTTP 错误及无效响应不重试。默认响应上限 1 MiB，不跟随重定向。

## Python 适配器

继承 `AudioModelAdapter`，实现：

```python
async def predict(self, request: UnderstandingRequest) -> PredictionResult: ...


async def close(self) -> None: ...
```

`request.audio_path` 是运行时输入文件，`to_payload()` 返回不含文件路径的题目字段。返回实际模型名称、回答、耗时和可选用量。测试适配器必须设置 `simulated=True`。临时错误抛出 `ModelError("简洁原因", retryable=True)`；错误消息不得包含密钥或原始服务响应。

注册新的适配器时，在 `models/registry.py` 与 `config.py` 中加入其名称。

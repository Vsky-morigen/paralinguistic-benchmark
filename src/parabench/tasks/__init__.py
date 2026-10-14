from parabench.schemas import GenerationRequest, Sample

PROMPT_VERSION = "expression_v1"


def build_request(sample: Sample, generation: dict) -> GenerationRequest:
    target = sample.targets
    if sample.language == "en":
        genders = {"男声": "male-sounding voice", "女声": "female-sounding voice"}
        emotions = {
            "中性": "neutral",
            "愉悦": "happy",
            "恼怒": "angry",
            "悲伤": "sad",
            "担忧": "worried",
        }
        speeds = {
            "中速": "moderate",
            "偏慢": "slow",
            "偏快": "fast",
            "中速至偏快": "moderate to fast",
        }
        instructions = (
            "Read the supplied text aloud and express the following through your voice. "
            "Output only the spoken reading.\n"
            f"Voice: {genders.get(target['gender'], target['gender'])}\n"
            f"Emotion: {emotions.get(target['emotion'], target['emotion'])}\n"
            f"Communicative intent: {target['context']}\n"
            f"Stress (emphasize text inside 【】): {target['stress']['display']}\n"
            f"Intrasentence pauses (pause at ｜): {target['pause']['display']}\n"
            f"Speaking rate: {speeds.get(target['speed'], target['speed'])} (currently unscored)\n"
            "Do not read annotation symbols aloud. 无 means none; 无句内停顿 means no intrasentence pauses."
        )
    else:
        instructions = (
            "朗读给定台词，并通过声音表达以下要求。只生成朗读语音。\n"
            f"声音性别：{target['gender']}\n情绪：{target['emotion']}\n"
            f"表达意图：{target['context']}\n"
            f"重音位置（【】内强调）：{target['stress']['display']}\n"
            f"句内停顿（｜处停顿）：{target['pause']['display']}\n"
            f"语速要求：{target['speed']}（该项暂不评分）\n"
            "台词中的标记用于指导表达，不需要读出标记。"
        )
    # No human recording path, rater answers, alternative choices, or validation
    # status is present in the model-facing request.
    return GenerationRequest(
        sample.sample_id, sample.text, sample.language, instructions, generation
    )

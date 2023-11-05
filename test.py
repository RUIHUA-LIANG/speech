import os
import whisper
import torch
import time

# 获取当前目录下的audio.mp3文件
audio_file = "audio.mp3"

# 记录开始时间
start_time = time.time()

# 加载Whisper模型（这里使用medium模型，你可以根据需要选择其他模型）
model = whisper.load_model("medium")

# 检查是否有多个GPU可用
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Found {torch.cuda.device_count()} GPUs. Using all GPUs for inference.")
    
    # 显式设置device_ids参数，使DataParallel使用所有GPU
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

# 将模型移动到GPU上
model = model.cuda()

# 如果使用了DataParallel，获取实际的模型
if isinstance(model, torch.nn.DataParallel):
    model = model.module

# 记录加载模型的时间
load_model_time = time.time()

# 构造输入文件路径
input_path = os.path.abspath(audio_file)

# 使用Whisper进行语音转录
result = model.transcribe(input_path)

# 记录语音转录的时间
transcription_time = time.time()

# 输出识别结果
print(result["text"])

# 输出各个环节的用时
print(f"Model loading time: {load_model_time - start_time} seconds")
print(f"Transcription time: {transcription_time - load_model_time} seconds")
import os
import whisper
import torch
import time
from multiprocessing import Process
import multiprocessing

def transcribe_on_gpu(gpu_id, model, audio_file):
    print(f"Process on GPU {gpu_id} started.")
    
    # 将模型移动到指定的GPU设备
    model = model.to(gpu_id)
    
    # 构造输入文件路径
    input_path = os.path.abspath(audio_file)
    
    # 使用Whisper进行语音转录
    result = model.transcribe(input_path)
    
    # 输出识别结果
    print(f"Transcription result (GPU {gpu_id}): {result['text']}")

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")
    # Your multiprocessing code here
    
    # 获取当前目录下的audio.mp3文件
    audio_file = "audio.mp3"

    # 记录开始时间
    start_time = time.time()
    
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Using all GPUs for inference.")
    
    # 加载Whisper模型并移动到第一个GPU
    model = whisper.load_model("medium")
    model = model.to(0)

    # 复制模型到其他GPU
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=range(1, num_gpus))
    
    # 创建多个进程，每个进程在不同的GPU上运行
    processes = []
    for gpu_id in range(num_gpus):
        process = Process(target=transcribe_on_gpu, args=(gpu_id, model, audio_file))
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()

    # 记录语音转录的时间
    transcription_time = time.time()

    # 输出各个环节的用时
    print(f"Transcription time: {transcription_time - start_time} seconds")
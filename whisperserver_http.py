from flask import Flask, request, jsonify
import whisper
import io
import tempfile

app = Flask(__name__)

# 初始化Whisper模型
model = whisper.load_model("base")

# 用于处理POST请求的端点
@app.route('/speech-recognition', methods=['POST'])
def speech_recognition():
    try:
        # 从POST请求中获取语音文件，使用request.data获取二进制数据
        audio_data = request.data

        # 将音频数据保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
            temp_audio_file.write(audio_data)
            audio_file_path = temp_audio_file.name

        # 使用Whisper模型进行语音识别
        result = model.transcribe(audio_file_path)

        # 返回识别结果
        response = {
            'transcript': result["text"]
        }

        app.logger.debug("Speech recognition request completed successfully")

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': '发生错误：{}'.format(str(e))}, 500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
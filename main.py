import os
import io
import json
import time
import threading
import re
import sys
import pandas as pd
import webview  # 核心库：pip install pywebview
import dashscope
from datetime import datetime
from PIL import Image, ImageGrab
from dashscope import MultiModalConversation
from http import HTTPStatus


# ================= 配置区域 =================
dashscope.api_key = "sk-d819275cca6044a9a541cb1e3d34d3e0"  # 你的API KEY

# 数据保存路径
SAVE_DIR = os.path.join(os.path.expanduser("~"), 'Desktop', 'FocusOS_Data')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ================= 资源路径处理 (打包exe必需) =================
def resource_path(relative_path):
    """ 获取资源的绝对路径，兼容开发环境和 PyInstaller 打包后的环境 """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ================= 核心逻辑 (来自 ipynb) =================
def compress_image(image, target_size_kb=400, max_dimension=1024):
    width, height = image.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    if image.mode in ("RGBA", "P"): image = image.convert("RGB")
    
    target_bytes = target_size_kb * 1024
    quality = 90
    while True:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=quality)
        if img_byte_arr.tell() <= target_bytes: break
        if quality > 20: quality -= 10
        else:
            image = image.resize((int(image.size[0]*0.9), int(image.size[1]*0.9)), Image.Resampling.LANCZOS)
            quality = 60
    return img_byte_arr

def parse_llm_output(raw_text):
    result = {"json_data": [], "score_change": 0, "comment": "保持专注..."}
    try:
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match: result["json_data"] = json.loads(match.group())
    except: pass
    
    score_match = re.search(r'score\s*=\s*(-?\d+)', raw_text)
    if score_match: 
        try: result["score_change"] = int(score_match.group(1))
        except: pass
        
    text_match = re.search(r'text\s*=\s*["\'](.*?)["\']', raw_text)
    if text_match: result["comment"] = text_match.group(1)
    return result

# ================= API 桥接类 =================
class FocusApi:
    def __init__(self):
        self.is_running = False
        self.user_goal = "高效工作"
        self.current_score = 100
        self.ai_comment = "Focus OS 已就绪"
        self.history_data = []

    def start_monitor(self, goal):
        if not self.is_running:
            self.is_running = True
            self.user_goal = goal if goal else "高效工作"
            # 启动后台线程
            t = threading.Thread(target=self._worker_loop)
            t.daemon = True
            t.start()
        return {"status": "started"}

    def stop_monitor(self):
        self.is_running = False
        return {"status": "stopped"}

    def get_status(self):
        return {
            "score": self.current_score,
            "comment": self.ai_comment,
            "running": self.is_running
        }

    def generate_report(self):
        """生成 CSV 报表"""
        if not self.history_data:
            return {"msg": "暂无数据可生成"}
        
        df = pd.DataFrame(self.history_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(SAVE_DIR, f"report_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return {"msg": f"报表已保存至: {csv_path}"}

    def _worker_loop(self):
        """后台监控循环"""
        while self.is_running:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                # 1. 截图处理
                screen = ImageGrab.grab()
                compressed = compress_image(screen)
                
                # (可选) 本地保存截图，为了节省空间这里只存在内存中上传，若需保存请解开注释
                # temp_path = os.path.join(SAVE_DIR, "temp.jpg")
                # with open(temp_path, "wb") as f: f.write(compressed.getvalue())
                
                # 为了 SDK 读取方便，这里还是临时存一下
                temp_path = os.path.join(SAVE_DIR, "_monitor_temp.jpg")
                with open(temp_path, "wb") as f: f.write(compressed.getvalue())
                abs_path = os.path.abspath(temp_path).replace('\\', '/')

                # 2. 调用 Qwen-VL-Plus
                prompt = (
                    f"用户目标:【{self.user_goal}】。\n"
                    "1.识别窗口(忽略任务栏)。2.判断专注度: 专心+1分/分心-2分。3.简短评价。\n"
                    "格式要求: score=1 text=\"评价内容\""
                )
                
                msgs = [{'role': 'user', 'content': [{'image': f"file://{abs_path}"}, {'text': prompt}]}]
                res = MultiModalConversation.call(model='qwen-vl-plus', messages=msgs)

                if res.status_code == HTTPStatus.OK:
                    raw = res.output.choices[0].message.content[0]['text']
                    data = parse_llm_output(raw)
                    
                    # 更新状态 (分数限制 0-200)
                    self.current_score = max(0, min(200, self.current_score + data["score_change"]))
                    self.ai_comment = data["comment"]

                    # 记录历史
                    self.history_data.append({
                        "time": timestamp,
                        "score": self.current_score,
                        "comment": data["comment"],
                        "change": data["score_change"]
                    })
                
            except Exception as e:
                print(f"Loop Error: {e}")
                self.ai_comment = "网络或API连接微弱..."

            # 智能等待 (保持每30秒一次)
            elapsed = time.time() - start_time
            sleep_time = max(0, 30 - elapsed)
            for _ in range(int(sleep_time * 10)):
                if not self.is_running: break
                time.sleep(0.1)

# ================= 主程序启动 =================
if __name__ == '__main__':
    api = FocusApi()
    
    # 创建窗口
    window = webview.create_window(
        'Focus OS', 
        url=resource_path('gui/index.html'), # 加载本地HTML
        js_api=api,
        width=340, height=520,  # 紧凑的 Widget 尺寸
        resizable=False,
        frameless=True,         # 无边框，像悬浮窗
        easy_drag=True,         # 允许拖动整个窗口
        on_top=True,            # 窗口置顶
        transparent=True        # 允许透明背景
    )
    
    webview.start(debug=False)
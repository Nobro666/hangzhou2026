#!/usr/bin/env python
# coding: UTF-8

import cv2
import time
import pyrealsense2 as rs
from pathlib import Path
import numpy as np

class RealSensePhotoCapture:
    def __init__(self, save_dir: str = "captured_photos"):
        # 创建保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 初始化RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)  # 对齐深度图到彩色图
        
        # 配置流
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        # 启动管道
        self.pipeline.start(self.config)

    def capture_photo(self) -> None:
        """捕获并保存一张照片"""
        # 获取帧
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        
        if not color_frame:
            print("无法获取彩色帧")
            return
        
        # 转换为图像
        color_img = np.asanyarray(color_frame.get_data())
        
        # 生成带时间戳的文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.save_dir / f"photo_{timestamp}.jpg"
        
        # 保存图像
        cv2.imwrite(str(filename), color_img)
        print(f"照片已保存: {filename}")
        
        # 显示已拍摄提示
        cv2.putText(color_img, "Captured!", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("RealSense Capture", color_img)
        cv2.waitKey(500)  # 显示500ms

    def start_capture_loop(self, max_count: int = 50) -> None:
        """每3秒捕获一张照片,拍摄指定次数后自动停止，按'q'可提前退出，加入倒计时显示"""
        try:
            print(f"开始拍照,每3秒一张,共拍摄{max_count}张,按'q'可提前退出...")
            count = 0  # 计数器初始化
            while count < max_count:  # 当拍摄次数小于最大次数时继续
                # 倒计时3秒并显示
                for i in range(3, 0, -1):
                    # 获取实时帧用于显示倒计时
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    color_img = np.asanyarray(color_frame.get_data())
                    
                    # 在图像上绘制倒计时和当前进度
                    cv2.putText(color_img, f"Capturing {count+1}/{max_count} in {i}...", (150, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("RealSense Capture", color_img)
                    
                    # 检查退出按键
                    key = cv2.waitKey(1000)  # 等待1秒
                    if key == ord('q'):
                        print("用户手动退出")
                        return
                
                # 倒计时结束，捕获照片
                self.capture_photo()
                count += 1  # 计数器加1
                
                # 检查退出按键
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("用户手动退出")
                    return
            
            print(f"已完成{max_count}张照片的拍摄，自动停止")
                    
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    capture = RealSensePhotoCapture()
    capture.start_capture_loop(50)  # 指定拍摄50次
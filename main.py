#!/usr/bin/env python3
"""
Early Wildfire Detection with Multi-Stage Edge Inference
稳定版：支持视频文件循环播放、CSV日志、警报截图。
用法：python main.py [视频路径或摄像头索引]
"""

import cv2
import time
import sys
import csv
import numpy as np
from datetime import datetime

from config import SystemConfig
from stage1_rapid_scan import RapidScanner
from stage2_cnn_verify import CNNVerifier
from stage3_temporal_decision import TemporalDecisionEngine
from utils.visualizer import Visualizer

class EarlyWildfireDetectionSystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.scanner = RapidScanner(config.stage1)
        self.verifier = CNNVerifier(config.stage2)
        self.decision_engine = TemporalDecisionEngine(config.stage3)
        self.visualizer = Visualizer()

        self.source_path = config.input_source
        self.cap = self._open_video(self.source_path)
        if self.cap is None:
            raise RuntimeError(f"无法打开视频源: {self.source_path}")

        self.frame_count = 0
        self.loop_count = 0

        # ----- 输出设置 -----
        # 1. 视频保存
        self.video_writer = None
        if config.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{config.output_path}wildfire_detection_{timestamp}.avi"
            self.video_writer = cv2.VideoWriter(output_file, fourcc, 20.0,
                                                (config.frame_width, config.frame_height+100))

        # 2. CSV 日志
        self.log_path = f"{config.output_path}detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['frame','time','s1_candidates','s2_fires','max_conf','ema','pos_ratio','alert'])
        self.log_file = open(self.log_path, 'a', newline='', encoding='utf-8')
        self.log_writer = csv.writer(self.log_file)

        print("="*60)
        print("  Early Wildfire Detection System")
        print("  Multi-Stage Edge Inference Pipeline")
        print("="*60)
        print(f"  Input: {self.source_path}")
        print(f"  Backend: {self.verifier.backend or 'fallback'}")
        print(f"  Log: {self.log_path}")
        print("="*60)

    def _open_video(self, source):
        src = source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return None
        return cap

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        self.cap.release()
                        time.sleep(0.5)
                        self.cap = self._open_video(self.source_path)
                        if self.cap is None or not self.cap.isOpened():
                            print("[Main] 重新打开失败，退出。")
                            break
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            print("[Main] 仍然无法读取帧，退出。")
                            break
                    self.loop_count += 1
                    print(f"[Main] 视频循环 (次数: {self.loop_count})")

                proc_frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))

                # 阶段1
                s1_start = time.time()
                candidates, viz_mask, s1_trig = self.scanner.scan(proc_frame)
                s1_time = time.time() - s1_start

                # 阶段2
                s2_time = None
                s2_trig = False
                stage2_results = []
                if s1_trig:
                    s2_start = time.time()
                    stage2_results = self.verifier.verify_batch(proc_frame, candidates[:3])
                    s2_time = time.time() - s2_start
                    s2_trig = any(r['is_fire'] for r in stage2_results)

                # 阶段3
                decision = self.decision_engine.update(stage2_results)

                # 可视化
                disp = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))
                if s1_trig:
                    contours2 = [np.array([[[c[0],c[1]],[c[0]+c[2],c[1]],
                                           [c[0]+c[2],c[1]+c[3]],[c[0],c[1]+c[3]]]]) for c in candidates]
                    disp = self.visualizer.draw_stage1_candidates(disp, contours2)
                for res in stage2_results:
                    if res['class_id'] == 1:
                        color = (0,0,255) if res['is_fire'] else (0,255,0)
                        disp = self.visualizer.draw_stage2_result(disp, res['bbox'],
                                                                  res['class_name'], res['confidence'], color)
                if decision['alert_active']:
                    disp = self.visualizer.draw_alert(disp)

                # ----- 输出：警报截图 -----
                if decision['alert_triggered']:
                    alert_path = f"{self.config.output_path}alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    cv2.imwrite(alert_path, disp)
                    print(f"[ALERT SAVED] {alert_path}")

                # ----- 输出：CSV 日志 -----
                fire_r = [r for r in stage2_results if r['class_id']==1]
                self.log_writer.writerow([
                    self.frame_count,
                    datetime.now().isoformat(),
                    len(candidates),
                    len(fire_r),
                    max([r['confidence'] for r in fire_r]) if fire_r else 0.0,
                    decision['ema_score'],
                    decision['positive_ratio'],
                    decision['alert_active']
                ])

                self.visualizer.update_metrics(s1_time, s2_time, s1_trig, s2_trig)
                metrics = self.visualizer.get_metrics()
                metrics.update({
                    "EMA": f"{decision['ema_score']:.3f}",
                    "Win+%": f"{decision['positive_ratio']:.1%}",
                    "Alerts": decision['total_alerts'],
                    "Backend": self.verifier.backend or "fallback",
                    "Loop": self.loop_count,
                })
                disp = self.visualizer.draw_status_bar(disp, metrics)

                if self.config.enable_display:
                    cv2.imshow("Early Wildfire Detection", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.decision_engine.reset()
                        print("[Main] 决策引擎已重置")

                if self.video_writer:
                    self.video_writer.write(disp)

                elapsed = time.time() - s1_start
                time.sleep(max(0, 1.0/self.config.target_fps - elapsed))
                self.frame_count += 1

        except KeyboardInterrupt:
            print("\n[Main] 用户中断")
        finally:
            # ===== 清理部分（cleanup） =====
            self.cap.release()
            if self.video_writer:
                self.video_writer.release()
            if hasattr(self, 'log_file') and self.log_file:
                self.log_file.close()
                print(f"[LOG SAVED] {self.log_path}")
            cv2.destroyAllWindows()
            print("[Main] 系统已安全退出")

def main():
    config = SystemConfig()
    if len(sys.argv) > 1:
        config.input_source = sys.argv[1]
        print(f"[Main] 使用输入源: {config.input_source}")
    system = EarlyWildfireDetectionSystem(config)
    system.run()

if __name__ == "__main__":
    main()
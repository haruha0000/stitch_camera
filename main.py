import stitching as st
import camera as cm
import cv2
import tkinter as tk
import numpy as np
import time
import os

# Tắt thông báo lỗi và tối ưu khởi động camera
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "1"  # Ưu tiên Media Foundation
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"          # Tắt debug log

def stitch(stack: list):
    if not stack or len(stack) == 1:
        return stack

    center_idx = len(stack) // 2
    base_img = stack[center_idx]

    left_images = stack[:center_idx][::-1] 
    right_images = stack[center_idx + 1:]     

    for img in left_images:
        res = st.image_stitching([img, base_img]) 
        if res is not None:
            base_img = res

    for img in right_images:
        res = st.image_stitching([base_img, img])
        if res is not None:
            base_img = res

    return [base_img]

def preprocess_images_for_stitching(images):
    processed = []
    for img in images:
        # Cân bằng sáng tối nhẹ (tạm thời bỏ tăng contrast và denoise)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        balanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        processed.append(balanced)
    return processed


if __name__ == "__main__":
    # Chuẩn bị bộ đệm FPS
    fps_buffer = []
    fps_buffer_size = 10
    last_time = time.time()

    ids = cm.find_camera_ids()
    print("Available camera IDs:", ids)

    caps = [cv2.VideoCapture(cam_id) for cam_id in ids]
    rows = 3
    columns = 2


    # Đánh dấu khung hình xử lý và bỏ qua
    process_every_n_frames = 1  # Chỉ xử lý 1 frame trong mỗi n frame
    frame_count = 0
    
    # Lưu panorama cuối cùng
    last_panorama = None

    if len(ids) == 0:
        print("Không có camera nào khả dụng.")
    else:
        while True:
            frame_count += 1
            stack = []
            current_time = time.time()
            elapsed = current_time - last_time
            last_time = current_time
            
            # Tính FPS trung bình
            fps_buffer.append(1.0 / elapsed if elapsed > 0 else 0)
            if len(fps_buffer) > fps_buffer_size:
                fps_buffer.pop(0)
            avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
            
            # Hiển thị FPS trên cửa sổ riêng
            fps_display = np.zeros((50, 150, 3), dtype=np.uint8)
            cv2.putText(fps_display, f"FPS: {avg_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("FPS", fps_display)

            ignor = 0
            for idx, cap in enumerate(caps):
                ret, frame = cap.read()

                if idx == ignor:
                    continue

                cm.show_all_cameras(ids, idx, (ret,frame) , rows, columns)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    for cap in caps:
                        cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
                    
                if not ret:
                    continue
                    
                im = frame.copy()
                if np.mean(im) < 30 or np.mean(im) > 230:
                    continue 
                max_size = 1000  # Giảm kích thước tối đa xuống để tăng tốc

                h, w = im.shape[:2]

                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    im = cv2.resize(im, (int(w * scale), int(h * scale)))

                if im is None:
                    print("Cannot read images")
                    continue
                    
                stack.append(im)
                
            # Chỉ xử lý panorama mỗi n frame để tăng tốc độ phản hồi
            if len(stack) > 0 and frame_count % process_every_n_frames == 0:
                # Đo thời gian xử lý panorama
                process_start = time.time()
                
                processed_stack = preprocess_images_for_stitching(stack)
                res = stitch(processed_stack)

                process_time = time.time() - process_start
                print(f"Panorama processing time: {process_time:.3f}s")
                
                if res and res[0] is not None:
                    last_panorama = res[0]
            
            # Hiển thị panorama mới nhất
            if last_panorama is not None and last_panorama.size > 0:
                # Lấy kích thước màn hình
                root = tk.Tk()
                root.withdraw()
                window_width = root.winfo_screenwidth()
                window_height = root.winfo_screenheight()
                root.destroy()
                
                # Thay đổi kích thước panorama để hiển thị đẹp hơn
                h, w = last_panorama.shape[:2]
                if w > window_width // 2:
                    scale = (window_width // 2) / w
                    display_panorama = cv2.resize(last_panorama, 
                                               (int(w * scale), int(h * scale)))
                else:
                    display_panorama = last_panorama.copy()
                
                # Hiển thị panorama
                cv2.imshow("Stitching Image", display_panorama)
                cv2.moveWindow("Stitching Image", window_width // 3, 0)
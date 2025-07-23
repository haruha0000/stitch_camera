import cv2
import numpy as np
import tkinter as tk

def show_all_cameras(ids, idx, cap, rows, columns, ignor = None):
    if idx != ignor:
        root = tk.Tk()
        root.withdraw()
        window_width = root.winfo_screenwidth()
        window_height = root.winfo_screenheight()
        cell_width = window_width // columns
        cell_height = window_height // rows
        
        root.destroy()
        
        ret, frame = cap
        window_name = f"Camera {ids[idx]}"

        if not ret:
            return

        frame_resized = cv2.resize(frame, (cell_width, cell_height))
        cv2.imshow(window_name, frame_resized)

        col = idx % columns
        row = idx // columns

        x = col * cell_width
        y = row * cell_height
        cv2.moveWindow(window_name, x, y)
        

def find_camera_ids(max_tested=10):
    available_ids = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_ids.append(i)
            cap.release()
    return available_ids

if __name__ == "__main__":
    ids = find_camera_ids()
    print("Available camera IDs:", ids)
    caps = [cv2.VideoCapture(cam_id) for cam_id in ids]
    if len(ids) == 0:
        print("Không có camera nào khả dụng.")
    else:
        while True:
            for idx, cap in enumerate(caps):
                ret, frame = cap.read()
                show_all_cameras(ids, idx, (ret,frame) , rows=2, columns=3)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    for cap in caps:
                        cap.release()
                    cv2.destroyAllWindows()
                    break



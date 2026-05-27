import cv2
import time
import threading
import queue
import urllib.request
import os
from ultralytics import YOLO

# =================================================================
# Course: Data Engineering (CSIE, Tamkang University)
# Lab 13: Multi-threading & Frame Dropping
# =================================================================

def download_video():
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    vid_path = "sample_video.mp4"
    if not os.path.exists(vid_path):
        print("[*] Downloading sample video...")
        urllib.request.urlretrieve(url, vid_path)
    return vid_path

# =========================================================
# GLOBAL VARIABLES & QUEUES
# =========================================================
# TODO 1: Initialize a thread-safe queue with maxsize=1
# This ensures we only hold the absolute freshest frame.
# ---------------------------------------------------------
frame_queue = queue.Queue(maxsize=1)

# A thread-safe flag to gracefully stop all threads
stop_event = threading.Event()

def producer_thread(video_path):
    """Producer: Reads frames from the video as fast as possible."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("[Producer] Started reading video...")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # Loop the video if it reaches the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            continue
            
        # ---------------------------------------------------------
        # TODO 2: Frame Dropping Logic (Producer side)
        # Attempt to put the frame into the queue without blocking.
        # If the queue is FULL (queue.Full exception):
        #   1. Remove the old frame using get_nowait()
        #   2. Put the new frame in.
        # Handle exceptions carefully to avoid race conditions!
        # ---------------------------------------------------------
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                frame_queue.get_nowait()  # Discard the stale frame
            except queue.Empty:
                pass # Safe catch: The consumer grabbed it exactly at this microsecond
            
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass # Safe catch: extremely rare race condition
        

        frame_count += 1
        # Simulate camera hardware delay (approx 30 FPS)
        time.sleep(0.03) 
        
    cap.release()
    print(f"\n[Producer] Stopped. Read {frame_count} frames total.")

def consumer_thread():
    """Consumer: Runs YOLO inference on the freshest frame."""
    print("[Consumer] Loading YOLOv10n model...")
    # Initialize model INSIDE the thread to avoid memory sharing issues
    model = YOLO("yolov10n.pt") 
    
    processed_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        # ---------------------------------------------------------
        # TODO 3: Retrieve frame and run inference
        # 1. Get frame from queue using frame_queue.get(timeout=1.0)
        # 2. If queue.Empty is raised, 'continue' the loop.
        # 3. Run model inference: model(frame, verbose=False)
        # ---------------------------------------------------------
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        results = model(frame, verbose=False)
            
        # --- Update & Print Metrics (Do not modify) ---
        processed_count += 1
        if processed_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = processed_count / elapsed
            print(f"[Consumer] Processed {processed_count} frames | Effective AI FPS: {fps:.2f}")
            
    print(f"\n[Consumer] Stopped. Processed {processed_count} frames total.")

if __name__ == "__main__":
    vid_path = download_video()
    
    # Initialize the queue (Ensure TODO 1 is completed)
    if frame_queue is None:
        raise ValueError("Please complete TODO 1 first! (Initialize frame_queue)")
        
    # Start Threads
    t_prod = threading.Thread(target=producer_thread, args=(vid_path,))
    t_cons = threading.Thread(target=consumer_thread)
    
    t_prod.start()
    t_cons.start()
    
    try:
        print("\n[Main] System running. Let it run for 15 seconds...")
        time.sleep(15)
        print("\n[Main] Time's up. Initiating shutdown...")
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user. Shutting down...")
    finally:
        # Signal all threads to stop
        stop_event.set()
        t_prod.join()
        t_cons.join()
        print("[Main] All threads closed cleanly. System exit.")

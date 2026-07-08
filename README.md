# Raspberry Pi AI Camera Person ReID & Tracking

This project contains Python scripts for real-time person tracking and counting using the Sony IMX500 sensor on the Raspberry Pi AI Camera (AITRIOS SDK `modlib`). It integrates **OSNet** (Omni-Scale Network) to perform **Person Re-Identification (ReID)**, allowing the system to maintain consistent tracking IDs even when people temporarily disappear, get occluded, or leave and re-enter the camera frame.

## 📂 Project Structure

*   `track_stream_reid.py`: Script that tracks people, counts total unique visitors using ReID, and displays persistent labels on the stream.
*   `area_count_reid.py`: Script that counts people in designated polygonal regions/areas of interest, displaying their persistent ReID IDs.
*   `requirements.txt`: Python package dependencies.

---

## 🛠️ Installation & Setup

You can automate the environment setup on your Raspberry Pi 5 using the provided shell script:

1. **Run the setup script:**
   ```bash
   chmod +x set.sh
   ./set.sh
   ```
   *This script checks system requirements, creates a Python virtual environment (`venv`), installs `torch` and `torchvision`, upgrades `pip`, and installs all dependencies including `torchreid` and the Sony AITRIOS `modlib` SDK.*

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install OpenCV system dependencies (if needed):**
   If you run into OpenCV shared library errors when starting the demos, run:
   ```bash
   sudo apt update && sudo apt install -y libglib2.0-0 libgl1-mesa-glx
   ```

---

## 🚀 Running the Applications

### 1. Persistent Tracking Demo
To run the standard tracking stream:
```bash
python track_stream_reid.py
```

### 2. Area Counting Demo with ReID
To count people inside specific polygonal regions:
```bash
python area_count_reid.py --stream
```

---

## 📡 Device Registration & Heartbeat Module

We have added a device tracking system consisting of a client daemon (`client.py`) that runs directly on the Raspberry Pi and connects to the PostgreSQL database.

### How it collects Device Details
*   **Device Name:** Retrieved using `socket.gethostname()`.
*   **Model Name:** Read directly from the system file `/proc/device-tree/model` (with emulation fallback).
*   **Serial Number:** Parsed from `/proc/cpuinfo` (extracts the `Serial` line, with MAC-address fallback).
*   **CPU Temperature:** Monitored from `/sys/class/thermal/thermal_zone0/temp` (or fallback to `vcgencmd measure_temp` or random emulation).

### Running the Client Daemon
1.  **Configure environment:** Make sure the credentials in `.env` are set correctly.
2.  **Start the Client Daemon on Raspberry Pi:**
    ```bash
    python client.py
    ```
3.  **Registration:**
    On startup or reconnection, the client connects to PostgreSQL and checks if the device already exists (matching on `serial_no`):
    *   If the device already exists, it updates its `device_name`, `model`, `status = 'ONLINE'`, and `last_seen`.
    *   Otherwise, it inserts a new device entry.
4.  **Heartbeats:**
    Every 30 seconds, the client reads the current CPU temperature and updates the database row setting `status = 'ONLINE'`, `last_seen` timestamp, and current `temperature`.
5.  **Offline Detection:**
    In each heartbeat cycle, the client also executes an update query to transitions other devices to `"OFFLINE"` if they haven't sent a heartbeat for more than 120 seconds.

---


## 💡 How ReID with OSNet Works

1. **Object Tracking:** `BYTETracker` maintains short-term frame-to-frame association using motion dynamics and Kalman filters.
2. **Feature Extraction:** When a new person is tracked, the script crops their bounding box from the frame, processes it, and passes it to the `osnet_x1_0` model (running locally on CPU) to extract a unique 512-dimensional embedding.
3. **Similarity Search:** The extracted embedding is compared against a gallery of previously seen people using **Cosine Similarity**.
4. **ID Association:** 
   - If the similarity is above the threshold (default: `0.70`), the script maps the new tracker ID back to the existing `global_id`.
   - If no match is found, a new `global_id` is created and stored in the gallery.

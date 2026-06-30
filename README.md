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
To count people inside specific polygonal areas, specify a JSON file containing the coordinates of the areas:
```bash
python area_count_reid.py --json-file areas.json
```

#### JSON Areas Format Example
```json
[
    {
        "points": [
            [0.1, 0.3],
            [0.45, 0.3],
            [0.45, 0.7],
            [0.1, 0.7]
        ]
    },
    {
        "points": [
            [0.55, 0.3],
            [0.9, 0.3],
            [0.9, 0.7],
            [0.55, 0.7]
        ]
    }
]
```

---

## 💡 How ReID with OSNet Works

1. **Object Tracking:** `BYTETracker` maintains short-term frame-to-frame association using motion dynamics and Kalman filters.
2. **Feature Extraction:** When a new person is tracked, the script crops their bounding box from the frame, processes it, and passes it to the `osnet_x1_0` model (running locally on CPU) to extract a unique 512-dimensional embedding.
3. **Similarity Search:** The extracted embedding is compared against a gallery of previously seen people using **Cosine Similarity**.
4. **ID Association:** 
   - If the similarity is above the threshold (default: `0.70`), the script maps the new tracker ID back to the existing `global_id`.
   - If no match is found, a new `global_id` is created and stored in the gallery.

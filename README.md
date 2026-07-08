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

### 2. Area Counting & DB Logging (Interactive Zones Mode)
To run the application, configure your zones interactively, trace transitions (IN vs. OUT), and log hourly counts to PostgreSQL:

1. **Configure Environment Variables**:
   Create a `.env` file in the root of the workspace:
   ```env
   DB_HOST="your_rds_or_postgres_host"
   DB_PORT="5432"
   DB_NAME="your_database_name"
   DB_USER="your_user"
   DB_PASSWORD="your_password"
   DB_SSLMODE="require"  # 'prefer', 'require', or 'disable'
   CAMERA_ID=1           # Camera ID identifying records in the DB
   ```

2. **Verify/Run DB Schema**:
   Ensure your database has the following tables:
   ```sql
   -- Zones table (stores coordinates)
   CREATE TABLE IF NOT EXISTS public.zones (
       zone_id bigserial PRIMARY KEY,
       zone_name varchar(100),
       description text,
       camera_id bigint,
       zone_type varchar(20), -- 'in' or 'out'
       points jsonb
   );

   -- Hourly aggregated stats
   CREATE TABLE IF NOT EXISTS public.people_count_hourly (
       id bigserial PRIMARY KEY,
       camera_id bigint,
       report_date date,
       report_hour smallint,
       total_in integer,
       total_out integer,
       peak_occupancy integer,
       avg_occupancy numeric(6,2),
       created_at timestamp without time zone DEFAULT now()
   );
   ```

3. **Start the Demo**:
   Run the area counter with the `--zones` CLI argument:
   ```bash
   python area_count_reid.py --zones --camera-id 1
   ```
   *Note: If no zones are saved in the DB for the specified `--camera-id`, the script opens an interactive OpenCV GUI. Click points to define the vertices of the **IN** polygon, press **Enter**, then click points to define the **OUT** polygon, and press **Enter** to save them to the DB and start tracking.*

   To redraw the zones and overwrite the existing configuration in the DB, run:
   ```bash
   python area_count_reid.py --zones --camera-id 1 --redraw
   ```

   To run with the live HTTP MJPEG stream (accessible on port 8000 by default):
   ```bash
   python area_count_reid.py --zones --camera-id 1 --stream
   ```

### 3. Basic Area Counting (File-Based Mode)
To count people inside static regions specified in a JSON file without database sync:
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

1. **Object Tracking**: `BYTETracker` maintains short-term frame-to-frame association using motion dynamics and Kalman filters.
2. **Feature Extraction**: When a new person is tracked, the script crops their bounding box from the frame, processes it, and passes it to the `osnet_x1_0` model (running locally on CPU) to extract a unique 512-dimensional embedding.
3. **Similarity Search**: The extracted embedding is compared against a gallery of previously seen people using **Cosine Similarity**.
4. **ID Association & State Transition**: 
   - If the similarity is above the threshold (default: `0.58`), the script maps the new tracker ID back to the existing ID.
   - For `--zones` mode, if a person travels from the OUT zone to the IN zone, they trigger an `IN` count. If they travel from the IN zone to the OUT zone, they trigger an `OUT` count.


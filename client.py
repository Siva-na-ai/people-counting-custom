import os
import socket
import time
import random
from datetime import datetime, timedelta
import psycopg2
from dotenv import load_dotenv

# Load env variables on startup
load_dotenv()

# Helper to read unique serial number from /proc/cpuinfo
def get_serial_number():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    parts = line.split(":")
                    if len(parts) > 1:
                        return parts[1].strip()
    except Exception:
        pass
    # Stable fallback based on UUID MAC address for non-Pi local testing
    import uuid
    mac = uuid.getnode()
    return f"SN-EMU-{mac}"

# Helper to read model from /proc/device-tree/model
def get_model():
    try:
        with open("/proc/device-tree/model", "r") as f:
            return f.read().strip().replace('\x00', '')
    except Exception:
        pass
    return "Sony IMX500-A1 (Emulated)"

# Helper to read CPU temperature
def get_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_raw = f.read().strip()
            return round(float(temp_raw) / 1000.0, 2)
    except Exception:
        try:
            import subprocess
            res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True)
            if res.returncode == 0:
                parts = res.stdout.strip().split('=')
                if len(parts) > 1:
                    return round(float(parts[1].replace("'C", "").strip()), 2)
        except Exception:
            pass
    # Emulated random temperature between 35 and 45°C
    return round(random.uniform(35.0, 45.0), 2)

# Helper to establish connection to PostgreSQL
def get_db_connection():
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    DB_SSLMODE = os.getenv("DB_SSLMODE", "require")
    
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        sslmode=DB_SSLMODE
    )
    return conn

# Register device on startup or reconnection
def register_device(conn, device_name, serial_number, model):
    cursor = conn.cursor()
    now = datetime.utcnow()
    try:
        cursor.execute("SELECT device_id FROM devices WHERE serial_no = %s", (serial_number,))
        row = cursor.fetchone()
        
        if row:
            device_id = row[0]
            cursor.execute(
                "UPDATE devices SET device_name = %s, model = %s, status = 'ONLINE', last_seen = %s "
                "WHERE serial_no = %s",
                (device_name, model, now, serial_number)
            )
            print(f"[{now}] Device already registered. Updated device_id: {device_id}")
        else:
            cursor.execute(
                "INSERT INTO devices (device_name, serial_no, model, status, last_seen, created_at) "
                "VALUES (%s, %s, %s, 'ONLINE', %s, %s) "
                "RETURNING device_id",
                (device_name, serial_number, model, now, now)
            )
            device_id = cursor.fetchone()[0]
            print(f"[{now}] Registered new device. Created device_id: {device_id}")
            
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error during registration: {e}")
        raise
    finally:
        cursor.close()

# Send heartbeat and perform inactivity updates
def send_heartbeat(conn, serial_number, temperature):
    cursor = conn.cursor()
    now = datetime.utcnow()
    try:
        # 1. Update heartbeat for current device
        cursor.execute(
            "UPDATE devices SET status = 'ONLINE', last_seen = %s, temperature = %s "
            "WHERE serial_no = %s",
            (now, temperature, serial_number)
        )
        
        # 2. Check for offline devices: mark other devices as OFFLINE if inactive for > 120s
        cutoff = now - timedelta(seconds=120)
        cursor.execute(
            "UPDATE devices SET status = 'OFFLINE' "
            "WHERE status = 'ONLINE' AND last_seen < %s AND serial_no != %s "
            "RETURNING serial_no",
            (cutoff, serial_number)
        )
        offline_devices = cursor.fetchall()
        conn.commit()
        
        print(f"[{now}] Heartbeat sent. Temperature: {temperature}°C")
        if offline_devices:
            for dev in offline_devices:
                print(f"[{now}] Inactivity check: Device {dev[0]} marked OFFLINE.")
                
    except Exception as e:
        conn.rollback()
        print(f"Error sending heartbeat: {e}")
        raise
    finally:
        cursor.close()

def main():
    device_name = socket.gethostname()
    serial_number = get_serial_number()
    model = get_model()
    
    print("=== Raspberry Pi Client Starting (Direct DB Connection) ===")
    print(f"Device Name:   {device_name}")
    print(f"Serial Number: {serial_number}")
    print(f"Model:         {model}")
    
    conn = None
    
    while True:
        try:
            # Reconnect if connection was dropped
            if conn is None or conn.closed != 0:
                print("\nConnecting to database...")
                conn = get_db_connection()
                print("Connected successfully!")
                
                # Perform registration
                register_device(conn, device_name, serial_number, model)
            
            # Send heartbeat
            temp = get_temperature()
            send_heartbeat(conn, serial_number, temp)
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\nStopping client...")
            break
        except Exception as e:
            print(f"Connection or query error: {e}. Retrying in 10 seconds...")
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            time.sleep(10)
            
    if conn:
        try:
            conn.close()
            print("Connection closed.")
        except Exception:
            pass

if __name__ == "__main__":
    main()

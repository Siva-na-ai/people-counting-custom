import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def fix_db():
    print("[*] Connecting to database...")
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode=os.getenv("DB_SSLMODE", "prefer")
        )
        print("[+] Connected successfully.")
    except Exception as e:
        print(f"[-] Database connection failed: {e}")
        return

    try:
        with conn.cursor() as cur:
            # First, check what columns the cameras table has
            print("[*] Inspecting 'cameras' table columns...")
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'cameras';"
            )
            cols = [r[0] for r in cur.fetchall()]
            print(f"[+] Found columns: {cols}")

            if not cols:
                print("[-] Table 'cameras' not found or has no columns.")
                return

            # Check if camera 1 already exists
            cur.execute("SELECT camera_id FROM public.cameras WHERE camera_id = 1;")
            if cur.fetchone():
                print("[+] Camera 1 already exists in 'cameras' table.")
                return

            # Insert camera 1 based on available columns
            if "camera_name" in cols:
                cur.execute("INSERT INTO public.cameras (camera_id, camera_name) VALUES (1, 'Camera 1');")
            elif "name" in cols:
                cur.execute("INSERT INTO public.cameras (camera_id, name) VALUES (1, 'Camera 1');")
            else:
                non_id_cols = [c for c in cols if c != "camera_id" and c != "created_at" and c != "updated_at"]
                if not non_id_cols:
                    cur.execute("INSERT INTO public.cameras (camera_id) VALUES (1);")
                else:
                    col_name = non_id_cols[0]
                    cur.execute(f"INSERT INTO public.cameras (camera_id, {col_name}) VALUES (1, 'Camera 1');")

            conn.commit()
            print("[+] Successfully inserted Camera 1 into 'cameras' table.")
    except Exception as e:
        print(f"[-] Error executing database fix: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    fix_db()

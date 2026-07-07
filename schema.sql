-- PostgreSQL Database Schema for People Counting & Device Management

-- 1. Create Zone Table
-- The zones table stores designated areas or regions of interest.
-- Must be created first as the cameras table references it.
CREATE TABLE IF NOT EXISTS zone (
    zone_id SERIAL PRIMARY KEY,
    zone_name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Create Devices Table
-- The devices table stores physical hardware/edge devices (e.g., Raspberry Pi, Jetson).
-- Must be created before cameras as cameras references it.
CREATE TABLE IF NOT EXISTS devices (
    device_id SERIAL PRIMARY KEY,
    device_name VARCHAR(255) NOT NULL,
    serial_no VARCHAR(100) UNIQUE NOT NULL,
    model VARCHAR(100),
    status VARCHAR(50) DEFAULT 'offline' CHECK (status IN ('online', 'offline', 'maintenance', 'error')),
    last_seen TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create Cameras Table
-- The cameras table represents camera streams, mapped to physical devices and logical zones.
CREATE TABLE IF NOT EXISTS cameras (
    camera_id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES devices(device_id) ON DELETE SET NULL,
    camera_name VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    zone_id INTEGER REFERENCES zone(zone_id) ON DELETE SET NULL,
    ip_address INET, -- Native PostgreSQL type for IPv4/IPv6 addresses
    status VARCHAR(50) DEFAULT 'offline' CHECK (status IN ('active', 'inactive', 'error')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Create People Count Table
-- Time-series data storing occupancy, entry, and exit counts.
CREATE TABLE IF NOT EXISTS people_count (
    id BIGSERIAL PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    entry_count INTEGER NOT NULL DEFAULT 0 CHECK (entry_count >= 0),
    outgoing_count INTEGER NOT NULL DEFAULT 0 CHECK (outgoing_count >= 0),
    current_occupancy INTEGER NOT NULL DEFAULT 0 CHECK (current_occupancy >= 0),
    total_people_detected INTEGER NOT NULL DEFAULT 0 CHECK (total_people_detected >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create Indexes for Query Optimization
-- Indexes speed up queries joining devices, zones, cameras, and time-series counting.

-- Indexes on foreign keys to optimize joins
CREATE INDEX IF NOT EXISTS idx_cameras_device_id ON cameras(device_id);
CREATE INDEX IF NOT EXISTS idx_cameras_zone_id ON cameras(zone_id);
CREATE INDEX IF NOT EXISTS idx_people_count_camera_id ON people_count(camera_id);

-- Composite index on camera_id and timestamp for fast time-series queries
CREATE INDEX IF NOT EXISTS idx_people_count_camera_time ON people_count(camera_id, timestamp DESC);
-- Index on timestamp for general time-based reports
CREATE INDEX IF NOT EXISTS idx_people_count_timestamp ON people_count(timestamp DESC);

-- Add Table/Column comments for documentation in database
COMMENT ON TABLE zone IS 'Designated logical zones or areas in the facility';
COMMENT ON TABLE devices IS 'Edge tracking/counting hardware devices';
COMMENT ON TABLE cameras IS 'Camera streams linked to devices and zones';
COMMENT ON TABLE people_count IS 'Time-series logs of people counting events';

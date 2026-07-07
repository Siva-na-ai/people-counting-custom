import React, { createContext, useState, useContext, useCallback } from 'react';

const DashboardContext = createContext();

const DEFAULT_ZONES = [
  {
    id: 1,
    name: 'Area 1 (Entrance Zone)',
    points: [
      [0.10, 0.25],
      [0.48, 0.25],
      [0.48, 0.75],
      [0.10, 0.75]
    ],
    color: 'rgba(16, 185, 129, 0.3)',
    borderColor: 'rgba(16, 185, 129, 1)',
    count: 0
  },
  {
    id: 2,
    name: 'Area 2 (Display Counter)',
    points: [
      [0.55, 0.25],
      [0.90, 0.25],
      [0.90, 0.75],
      [0.55, 0.75]
    ],
    color: 'rgba(59, 130, 246, 0.3)',
    borderColor: 'rgba(59, 130, 246, 1)',
    count: 0
  }
];

export const DashboardProvider = ({ children }) => {
  const [activeZones, setActiveZones] = useState(DEFAULT_ZONES);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraData, setCameraData] = useState({
    inside: 0,
    outside: 0,
    unique: 0,
    visitors: []
  });
  const [reidDatabase, setReidDatabase] = useState([
    { id: 1, confidence: 0.94, lastSeen: '10s ago', area: 'Area 1', color: '#10b981' },
    { id: 2, confidence: 0.89, lastSeen: '1m ago', area: 'Area 2', color: '#3b82f6' },
    { id: 3, confidence: 0.92, lastSeen: '3m ago', area: 'Area 1', color: '#f59e0b' },
    { id: 4, confidence: 0.91, lastSeen: '5m ago', area: 'Area 2', color: '#8b5cf6' },
    { id: 5, confidence: 0.88, lastSeen: '8m ago', area: 'Area 1', color: '#ec4899' }
  ]);
  
  const [fps, setFps] = useState(30.0);
  const [cpuTemp, setCpuTemp] = useState(42.2);
  const [toasts, setToasts] = useState([]);
  const [activityLogs, setActivityLogs] = useState([
    { id: 'l1', time: '13:40:02', text: 'System booted successfully.', type: 'info' },
    { id: 'l2', time: '13:40:15', text: 'Edge Vision Engine initialized in simulator mode.', type: 'success' },
    { id: 'l3', time: '13:41:10', text: 'Default Area 1 & Area 2 boundaries loaded.', type: 'info' }
  ]);
  
  const [cameraConfig, setCameraConfig] = useState({
    type: 'simulated',
    source: 'AI Simulator (Default)'
  });
  const [isConnecting, setIsConnecting] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [theme, setTheme] = useState('light');

  const addToast = useCallback((message, type = 'success') => {
    const id = Date.now() + Math.random().toString(36).substr(2, 9);
    setToasts(prev => [...prev, { id, message, type }]);
    
    // Auto-remove toast after 4 seconds
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 4000);
  }, []);

  const addActivityLog = useCallback((text, type = 'info') => {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    const id = Date.now() + Math.random().toString(36).substr(2, 9);
    setActivityLogs(prev => [
      { id, time: timeStr, text, type },
      ...prev.slice(0, 49) // Keep last 50 logs
    ]);
  }, []);

  const updateZonePoints = useCallback((zoneId, points) => {
    setActiveZones(prev => prev.map(zone => {
      if (zone.id === zoneId) {
        return { ...zone, points };
      }
      return zone;
    }));
  }, []);

  const resetZones = useCallback(() => {
    setActiveZones(prev => prev.map((zone, idx) => {
      if (idx === 0) {
        return {
          ...zone,
          points: [
            [0.10, 0.25],
            [0.48, 0.25],
            [0.48, 0.75],
            [0.10, 0.75]
          ]
        };
      } else {
        return {
          ...zone,
          points: [
            [0.55, 0.25],
            [0.90, 0.25],
            [0.90, 0.75],
            [0.55, 0.75]
          ]
        };
      }
    }));
    addToast('Restored default counting zones.', 'success');
    addActivityLog('Restored default counting zones.', 'info');
  }, [addToast, addActivityLog]);

  const clearReidGallery = useCallback(() => {
    setReidDatabase([]);
    setCameraData(prev => ({
      ...prev,
      unique: 0
    }));
    addToast('Local visitor profiles directory cleared.', 'warning');
    addActivityLog('Local visitor profiles directory cleared.', 'warning');
  }, [addToast, addActivityLog]);

  const connectCamera = useCallback((type, source) => {
    setIsConnecting(true);
    addToast(`Connecting to ${type.toUpperCase()} camera source: ${source}...`, 'info');
    addActivityLog(`Attempting connection to ${type.toUpperCase()} camera source: ${source}...`, 'info');
    
    setTimeout(() => {
      setIsConnecting(false);
      setCameraConfig({ type, source });
      addToast(`Connected to ${type.toUpperCase()} camera source successfully!`, 'success');
      addActivityLog(`Connected to ${type.toUpperCase()} camera source successfully!`, 'success');
    }, 1800);
  }, [addToast, addActivityLog]);

  return (
    <DashboardContext.Provider
      value={{
        activeZones,
        setActiveZones,
        cameraActive,
        setCameraActive,
        cameraData,
        setCameraData,
        reidDatabase,
        setReidDatabase,
        fps,
        setFps,
        cpuTemp,
        setCpuTemp,
        toasts,
        addToast,
        activityLogs,
        addActivityLog,
        cameraConfig,
        setCameraConfig,
        isConnecting,
        connectCamera,
        showConfigModal,
        setShowConfigModal,
        updateZonePoints,
        resetZones,
        clearReidGallery,
        theme,
        setTheme
      }}
    >
      {children}
    </DashboardContext.Provider>
  );
};

export const useDashboard = () => useContext(DashboardContext);

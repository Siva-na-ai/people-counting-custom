import { useEffect, useRef } from 'react';
import { useDashboard } from '../context/DashboardContext';
import { fetchCameraData } from '../services/api';

export const useCameraData = () => {
  const {
    cameraActive,
    setCameraActive,
    setCameraData,
    reidDatabase,
    setReidDatabase,
    addToast,
    addActivityLog
  } = useDashboard();

  const prevActiveRef = useRef(cameraActive);
  const reidDatabaseRef = useRef(reidDatabase);

  // Sync ref to avoid stale closure issues in intervals
  useEffect(() => {
    reidDatabaseRef.current = reidDatabase;
  }, [reidDatabase]);

  useEffect(() => {
    const poll = async () => {
      try {
        const data = await fetchCameraData();
        const wasActive = prevActiveRef.current;
        const currentActive = !!data.active;
        
        setCameraActive(currentActive);
        prevActiveRef.current = currentActive;

        if (currentActive) {
          setCameraData(data);

          if (!wasActive) {
            addToast('Live connection established: Receiving feed from camera script!', 'success');
            addActivityLog('Live connection established: Receiving feed from camera script!', 'success');
          }

          // Process visitors for ReID database updates
          const visitors = data.visitors || [];
          let reidUpdated = false;
          const newEntries = [];

          visitors.forEach(v => {
            const exists = reidDatabaseRef.current.some(p => p.id === v.id);
            if (!exists) {
              const confidence = typeof v.confidence === 'number' ? v.confidence : 0.95;
              newEntries.push({
                id: v.id,
                confidence: parseFloat(confidence.toFixed(3)),
                lastSeen: 'Just Now',
                area: v.area || 'Main Coverage',
                color: v.color || '#10b981'
              });
              
              addToast(`Profile Recognized: Visitor #${v.id} detected on camera.`, 'success');
              addActivityLog(`Profile Recognized: Visitor #${v.id} detected on camera (confidence: ${(confidence * 100).toFixed(1)}%).`, 'success');
              reidUpdated = true;
            }
          });

          if (reidUpdated) {
            setReidDatabase(prev => {
              const merged = [...newEntries, ...prev];
              return merged.slice(0, 8); // Limit to 8 items for UI space
            });
          }
        } else {
          if (wasActive) {
            addToast('Camera script went offline. Switching to simulation mode.', 'warning');
            addActivityLog('Camera script went offline. Switching to simulation mode.', 'warning');
          }
        }
      } catch (err) {
        if (prevActiveRef.current) {
          setCameraActive(false);
          prevActiveRef.current = false;
          addToast('Camera script went offline. Switching to simulation mode.', 'warning');
          addActivityLog('Camera script went offline. Switching to simulation mode.', 'warning');
        }
      }
    };

    // Poll immediately
    poll();

    // Poll every 500ms
    const interval = setInterval(poll, 500);

    return () => clearInterval(interval);
  }, [setCameraActive, setCameraData, setReidDatabase, addToast, addActivityLog]);
};

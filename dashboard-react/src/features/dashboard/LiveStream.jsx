import React, { useRef, useEffect, useState } from 'react';
import { Video, Settings, Cpu } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import { useShopperSimulation } from '../../hooks/useShopperSimulation';

export const LiveStream = () => {
  const {
    cameraActive,
    fps,
    cpuTemp,
    cameraConfig,
    setShowConfigModal
  } = useDashboard();

  const canvasRef = useRef(null);

  // Initialize the shopper simulation (draws when cameraActive is false)
  useShopperSimulation(canvasRef);

  const getFeedStatusText = () => {
    if (cameraActive) {
      return `LIVE CAMERA ACTIVE: ${cameraConfig.type.toUpperCase()} (${cameraConfig.source})`;
    }
    return 'Feed Simulated from AI Camera Stream';
  };

  const getFeedStatusStyle = () => {
    return cameraActive ? { color: '#10b981' } : { color: '#ffffff' };
  };

  return (
    <div className="panel stream-panel">
      <div className="panel-header">
        <div className="panel-header-title">
          <Video size={18} className="header-icon green" />
          <h3>IMX500 Live Tracking Stream</h3>
        </div>
        <div className="stream-stats">
          <span className="stat-tag">
            <span className={`indicator ${cameraActive ? 'green' : 'green-sim'}`}></span> 
            Analytics Rate: <strong>{fps.toFixed(1)}</strong> Hz
          </span>
          <span className="stat-tag">Engine Load: <strong>{cpuTemp.toFixed(1)}</strong>%</span>
          <button className="btn btn-outline btn-sm" onClick={() => setShowConfigModal(true)} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <Settings size={12} /> Configure
          </button>
        </div>
      </div>
      <div className="panel-body video-container" style={{ position: 'relative', display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0f172a', minHeight: '480px' }}>
        {cameraActive ? (
          <>
            <img 
              src="/api/stream" 
              alt="Live Camera Stream"
              style={{ width: '100%', height: '100%', maxWidth: '640px', maxHeight: '480px', objectFit: 'contain', display: 'block' }}
            />
            <div className="crosshair-overlay" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
              <div style={{ position: 'absolute', left: '50%', top: '20px', bottom: '20px', width: '1px', backgroundColor: 'rgba(255,255,255,0.15)' }}></div>
              <div style={{ position: 'absolute', top: '50%', left: '20px', right: '20px', height: '1px', backgroundColor: 'rgba(255,255,255,0.15)' }}></div>
            </div>
          </>
        ) : (
          <canvas ref={canvasRef} id="live-canvas" width="640" height="480"></canvas>
        )}
        <div className="video-overlay-banner">
          <div className="overlay-left">
            <Cpu size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: '4px' }} /> 
            Edge Vision Intelligence Engine
          </div>
          <div className="overlay-right" id="val-feed-status" style={getFeedStatusStyle()}>
            {getFeedStatusText()}
          </div>
        </div>
      </div>
    </div>
  );
};

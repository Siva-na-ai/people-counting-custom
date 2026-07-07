import React, { useState, useEffect } from 'react';
import { Settings } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import { Modal } from '../../components/Modal';

export const CameraConfigModal = () => {
  const { showConfigModal, setShowConfigModal, connectCamera } = useDashboard();
  const [streamType, setStreamType] = useState('simulated');
  const [streamUri, setStreamUri] = useState('');
  const [cameraIp, setCameraIp] = useState('');
  const [cameraPort, setCameraPort] = useState('80');

  // Reset inputs when modal opens
  useEffect(() => {
    if (showConfigModal) {
      setStreamType('simulated');
      setStreamUri('');
      setCameraIp('');
      setCameraPort('80');
    }
  }, [showConfigModal]);

  const handleSubmit = (e) => {
    e.preventDefault();
    setShowConfigModal(false);

    let sourceDetail = '';
    if (streamType === 'rtsp') {
      sourceDetail = streamUri || 'rtsp://admin:password@192.168.1.100:554/stream1';
    } else if (streamType === 'ip') {
      sourceDetail = `${cameraIp || '192.168.1.100'}:${cameraPort || '80'}`;
    } else if (streamType === 'api') {
      sourceDetail = streamUri || 'http://192.168.1.100:8000/api/frame';
    } else if (streamType === 'webcam') {
      sourceDetail = `Local Webcam Index ${streamUri || '0'}`;
    } else {
      sourceDetail = 'Simulated Vision Engine';
    }

    connectCamera(streamType, sourceDetail);
  };

  const getPlaceholder = () => {
    if (streamType === 'rtsp') return 'rtsp://admin:password@192.168.1.100:554/stream1';
    if (streamType === 'api') return 'http://192.168.1.100:8000/api/frame';
    if (streamType === 'webcam') return '0';
    return '';
  };

  const getUriLabel = () => {
    if (streamType === 'rtsp') return 'RTSP Connection Link';
    if (streamType === 'api') return 'HTTP API / REST Feed URL';
    if (streamType === 'webcam') return 'USB Webcam Index';
    return '';
  };

  return (
    <Modal 
      isOpen={showConfigModal} 
      onClose={() => setShowConfigModal(false)}
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Settings size={18} className="header-icon green" />
          <span>Stream Configuration</span>
        </div>
      }
    >
      <form onSubmit={handleSubmit} id="camera-config-form">
        <div className="form-group">
          <label htmlFor="stream-type">Camera Connection Type</label>
          <select 
            id="stream-type" 
            className="form-select"
            value={streamType}
            onChange={(e) => setStreamType(e.target.value)}
          >
            <option value="rtsp">RTSP Stream (rtsp://...)</option>
            <option value="ip">IP Camera (http://...)</option>
            <option value="api">HTTP API / REST Feed (json/image)</option>
            <option value="webcam">Local USB Webcam</option>
            <option value="simulated">AI Simulator (Default)</option>
          </select>
        </div>

        {streamType !== 'simulated' && streamType !== 'ip' && (
          <div className="form-group" id="uri-group">
            <label htmlFor="stream-uri">{getUriLabel()}</label>
            <input 
              type="text" 
              id="stream-uri" 
              className="form-input" 
              placeholder={getPlaceholder()}
              value={streamUri}
              onChange={(e) => setStreamUri(e.target.value)}
              required
            />
          </div>
        )}

        {streamType === 'ip' && (
          <>
            <div className="form-group" id="ip-group">
              <label htmlFor="camera-ip">IP Address</label>
              <input 
                type="text" 
                id="camera-ip" 
                className="form-input" 
                placeholder="192.168.1.100"
                value={cameraIp}
                onChange={(e) => setCameraIp(e.target.value)}
                required
              />
            </div>
            <div className="form-group" id="port-group">
              <label htmlFor="camera-port">Port</label>
              <input 
                type="number" 
                id="camera-port" 
                className="form-input" 
                placeholder="80"
                value={cameraPort}
                onChange={(e) => setCameraPort(e.target.value)}
                required
              />
            </div>
          </>
        )}

        <div className="modal-footer">
          <button 
            type="button" 
            className="btn btn-outline" 
            onClick={() => setShowConfigModal(false)}
          >
            Cancel
          </button>
          <button type="submit" className="btn btn-primary">
            Connect Stream
          </button>
        </div>
      </form>
    </Modal>
  );
};

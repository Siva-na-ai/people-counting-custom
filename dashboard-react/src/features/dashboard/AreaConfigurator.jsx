import React, { useRef, useEffect, useState } from 'react';
import { Edit3, Clipboard, Check } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';

export const AreaConfigurator = () => {
  const { activeZones, setActiveZones, resetZones, updateZonePoints, addToast, theme } = useDashboard();
  const canvasRef = useRef(null);
  const dragInfoRef = useRef(null); // { zoneIdx, pointIdx }
  const [copied, setCopied] = useState(false);

  // Generate JSON coordinates output
  const jsonOutput = JSON.stringify(
    activeZones.map(zone => ({
      points: zone.points.map(pt => [
        parseFloat(pt[0].toFixed(3)),
        parseFloat(pt[1].toFixed(3))
      ])
    })),
    null,
    4
  );

  // Copy to clipboard function
  const handleCopy = () => {
    navigator.clipboard.writeText(jsonOutput)
      .then(() => {
        setCopied(true);
        addToast('JSON coordinates copied to clipboard!', 'success');
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => {
        addToast('Failed to copy to clipboard.', 'warning');
      });
  };

  // Draw configuration canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const isLight = theme === 'light';
    // Draw background representing camera view
    ctx.fillStyle = isLight ? '#f1f5f9' : '#0f172a';
    ctx.fillRect(0, 0, 400, 300);

    // Draw grid lines
    ctx.strokeStyle = isLight ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 25; i < 400; i += 25) {
      ctx.beginPath();
      ctx.moveTo(i, 0); ctx.lineTo(i, 300);
      ctx.stroke();
    }
    for (let i = 25; i < 300; i += 25) {
      ctx.beginPath();
      ctx.moveTo(0, i); ctx.lineTo(400, i);
      ctx.stroke();
    }

    // Draw active areas
    activeZones.forEach((zone) => {
      ctx.beginPath();
      zone.points.forEach((pt, pIdx) => {
        const px = pt[0] * 400;
        const py = pt[1] * 300;
        if (pIdx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();

      // Fill area
      ctx.fillStyle = zone.color;
      ctx.fill();
      ctx.strokeStyle = zone.borderColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw clickable vertices
      zone.points.forEach((pt) => {
        const px = pt[0] * 400;
        const py = pt[1] * 300;

        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = zone.borderColor;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Vertex label
        ctx.fillStyle = isLight ? '#475569' : '#94a3b8';
        ctx.font = '8px monospace';
        ctx.fillText(`[${pt[0].toFixed(2)}, ${pt[1].toFixed(2)}]`, px + 8, py + 3);
      });

      // Write Zone ID in center
      let sumX = 0, sumY = 0;
      zone.points.forEach(pt => { sumX += pt[0]; sumY += pt[1]; });
      const cx = (sumX / zone.points.length) * 400;
      const cy = (sumY / zone.points.length) * 300;

      ctx.fillStyle = isLight ? '#0f172a' : '#ffffff';
      ctx.font = 'bold 10px "Plus Jakarta Sans", sans-serif';
      ctx.fillText(`Area ${zone.id}`, cx - 15, cy + 3);
    });
  }, [activeZones]);

  // Setup mouse interactions
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMouseDown = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      // Check if cursor clicked a vertex (with 10px buffer)
      for (let zIdx = 0; zIdx < activeZones.length; zIdx++) {
        const zone = activeZones[zIdx];
        for (let pIdx = 0; pIdx < zone.points.length; pIdx++) {
          const pt = zone.points[pIdx];
          const px = pt[0] * 400;
          const py = pt[1] * 300;
          const dist = Math.hypot(mx - px, my - py);
          if (dist <= 10) {
            dragInfoRef.current = { zoneIdx: zIdx, pointIdx: pIdx };
            return;
          }
        }
      }
    };

    const handleMouseMove = (e) => {
      if (!dragInfoRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      // Bound check inside canvas
      const nx = Math.max(0, Math.min(1.0, mx / 400));
      const ny = Math.max(0, Math.min(1.0, my / 300));

      const { zoneIdx, pointIdx } = dragInfoRef.current;
      
      // Update local state directly
      setActiveZones(prev => prev.map((zone, zIdx) => {
        if (zIdx === zoneIdx) {
          const newPoints = [...zone.points];
          newPoints[pointIdx] = [nx, ny];
          return { ...zone, points: newPoints };
        }
        return zone;
      }));
    };

    const handleMouseUp = () => {
      dragInfoRef.current = null;
    };

    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [activeZones, setActiveZones]);

  return (
    <div className="panel zone-config-panel">
      <div className="panel-header">
        <div className="panel-header-title">
          <Edit3 size={18} className="header-icon green" />
          <h3>Visual Area Configurator</h3>
        </div>
        <div className="actions">
          <button className="btn btn-outline btn-sm" onClick={resetZones}>Reset Areas</button>
        </div>
      </div>
      <div className="panel-body poly-config-body">
        <p className="description">Click and drag vertices to redefine counting boundaries. Changes calculate normalized layout coordinates.</p>
        <div className="canvas-editor-container" style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          <div>
            <canvas ref={canvasRef} id="poly-canvas" width="400" height="300" style={{ cursor: 'crosshair', display: 'block', borderRadius: '6px' }}></canvas>
          </div>
          <div style={{ flex: '1', minWidth: '250px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Configuration JSON Output</span>
              <button 
                className="btn btn-outline btn-sm" 
                onClick={handleCopy}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', padding: '4px 8px', fontSize: '11px' }}
              >
                {copied ? <Check size={12} /> : <Clipboard size={12} />}
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
            <pre 
              id="json-output" 
              style={{ 
                margin: 0, 
                padding: '12px', 
                backgroundColor: '#0f172a', 
                color: '#10b981', 
                fontFamily: 'monospace', 
                fontSize: '11px', 
                borderRadius: '6px', 
                overflowX: 'auto',
                flex: '1',
                maxHeight: '260px'
              }}
            >
              {jsonOutput}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};

import { useEffect, useRef } from 'react';
import { useDashboard } from '../context/DashboardContext';

function isPointInPolygon(point, vs) {
  const x = point[0], y = point[1];
  let inside = false;
  for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
    const xi = vs[i][0], yi = vs[i][1];
    const xj = vs[j][0], yj = vs[j][1];
    const intersect = ((yi > y) !== (yj > y))
        && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

class Shopper {
  constructor(id, color) {
    this.id = id;
    this.color = color;
    this.x = Math.random() * 640;
    this.y = Math.random() * 120 + 300; // bottom section walker
    this.vx = (Math.random() - 0.5) * 3;
    this.vy = (Math.random() - 0.5) * 1.5;
    this.w = 55;
    this.h = 130;
    this.history = [];
    this.activeAreaId = null;
  }

  update(activeZones, onZoneChange) {
    this.x += this.vx;
    this.y += this.vy;

    // Bounce on boundaries
    if (this.x < 30 || this.x > 610) this.vx *= -1;
    if (this.y < 160 || this.y > 450) this.vy *= -1;

    // Track path history
    this.history.push({ x: this.x, y: this.y + this.h / 2 });
    if (this.history.length > 25) this.history.shift();

    // Detect Area containment
    const checkPoint = [this.x / 640, (this.y + this.h) / 480];
    let currentInArea = null;

    activeZones.forEach(zone => {
      if (isPointInPolygon(checkPoint, zone.points)) {
        currentInArea = zone.id;
      }
    });

    if (currentInArea !== this.activeAreaId) {
      onZoneChange(this.id, this.activeAreaId, currentInArea);
      this.activeAreaId = currentInArea;
    }
  }

  draw(ctx) {
    // Draw path history
    ctx.beginPath();
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.35;
    for (let i = 0; i < this.history.length; i++) {
      if (i === 0) ctx.moveTo(this.history[i].x, this.history[i].y);
      else ctx.lineTo(this.history[i].x, this.history[i].y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Draw bounding box
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 3;
    ctx.strokeRect(this.x - this.w / 2, this.y - this.h / 2, this.w, this.h);

    // Draw header tag
    ctx.fillStyle = this.color;
    ctx.fillRect(this.x - this.w / 2 - 1, this.y - this.h / 2 - 25, this.w + 2, 25);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px "Plus Jakarta Sans", sans-serif';
    ctx.fillText(`Visitor #${this.id}`, this.x - this.w / 2 + 6, this.y - this.h / 2 - 8);
  }
}

export const useShopperSimulation = (canvasRef) => {
  const {
    activeZones,
    cameraActive,
    reidDatabase,
    setReidDatabase,
    cameraData,
    setCameraData,
    addToast,
    addActivityLog,
    setFps,
    setCpuTemp,
    theme
  } = useDashboard();

  const shoppersRef = useRef([
    new Shopper(1, '#10b981'),
    new Shopper(2, '#3b82f6'),
    new Shopper(3, '#f59e0b')
  ]);

  const nextTrackIdRef = useRef(4);
  const activeZonesRef = useRef(activeZones);
  
  useEffect(() => {
    activeZonesRef.current = activeZones;
  }, [activeZones]);

  // Handle shopper entering/exiting zones
  const handleZoneChange = (shopperId, oldZoneId, newZoneId) => {
    if (newZoneId !== null) {
      const zone = activeZonesRef.current.find(z => z.id === newZoneId);
      if (zone) {
        addToast(`Visitor #${shopperId} entered ${zone.name}`, 'info');
        addActivityLog(`Visitor #${shopperId} entered ${zone.name}`, 'info');
      }
    } else if (oldZoneId !== null) {
      const zone = activeZonesRef.current.find(z => z.id === oldZoneId);
      if (zone) {
        addActivityLog(`Visitor #${shopperId} exited ${zone.name}`, 'info');
      }
    }
  };

  // Run the simulation traffic changes (spawn/despawn) periodically
  useEffect(() => {
    if (cameraActive) return;

    const trafficInterval = setInterval(() => {
      const shoppers = shoppersRef.current;
      
      if (shoppers.length >= 5 && Math.random() < 0.4) {
        // Despawn shopper
        const removed = shoppers.shift();
        if (removed) {
          addToast(`Visitor #${removed.id} exited coverage zone.`, 'info');
          addActivityLog(`Visitor #${removed.id} exited coverage zone.`, 'info');
        }
      } else if (shoppers.length < 6) {
        // Decide if returning visitor (ReID Match) or new
        const isReturning = Math.random() > 0.4 && reidDatabase.length > 0;
        
        if (isReturning) {
          const randomHist = reidDatabase[Math.floor(Math.random() * reidDatabase.length)];
          const alreadyActive = shoppers.some(s => s.id === randomHist.id);
          
          if (!alreadyActive) {
            const simScore = (0.85 + Math.random() * 0.14).toFixed(3);
            const newShopper = new Shopper(randomHist.id, randomHist.color);
            shoppers.push(newShopper);
            
            addToast(`Welcome back visitor #${randomHist.id} (${(simScore * 100).toFixed(1)}% Match Similarity)`, 'success');
            addActivityLog(`ReID Match: Welcome back visitor #${randomHist.id} (Similarity: ${(simScore * 100).toFixed(1)}%)`, 'success');
            
            // Update confidence and last seen in database
            setReidDatabase(prev => prev.map(p => {
              if (p.id === randomHist.id) {
                return { ...p, lastSeen: 'Just Now', confidence: parseFloat(simScore) };
              }
              return p;
            }));
          }
        } else {
          // Spawn new shopper
          const newId = nextTrackIdRef.current++;
          const colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#ef4444', '#06b6d4'];
          const randomColor = colors[newId % colors.length];
          const newShopper = new Shopper(newId, randomColor);
          shoppers.push(newShopper);

          // Add to ReID database
          setReidDatabase(prev => {
            const exists = prev.some(p => p.id === newId);
            if (exists) return prev;
            return [{
              id: newId,
              confidence: 0.95,
              lastSeen: 'Just Now',
              area: 'Area 1',
              color: randomColor
            }, ...prev].slice(0, 8);
          });
          
          addToast(`New profile registered for visitor #${newId}.`, 'success');
          addActivityLog(`New profile registered: Visitor #${newId} detected.`, 'success');

          // Increment unique count
          setCameraData(prev => ({
            ...prev,
            unique: prev.unique + 1
          }));
        }
      }
    }, 12000); // Check every 12 seconds to keep it active but not spammy

    return () => clearInterval(trafficInterval);
  }, [cameraActive, reidDatabase, addToast, addActivityLog, setReidDatabase, setCameraData]);

  // Main animation loops
  useEffect(() => {
    let animationFrameId;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let lastTime = performance.now();
    let frameCount = 0;

    const animate = (time) => {
      // Calculate real FPS
      frameCount++;
      if (time - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = time;
        // Random CPU temp fluctuation
        setCpuTemp(prev => {
          const delta = (Math.random() - 0.5) * 0.8;
          return parseFloat(Math.max(38, Math.min(55, prev + delta)).toFixed(1));
        });
      }

      if (!cameraActive) {
        const isLight = theme === 'light';
        // 1. Draw Simulated Retail Background
        ctx.fillStyle = isLight ? '#f1f5f9' : '#1e293b';
        ctx.fillRect(0, 0, 640, 480);

        // Store shelves sketches
        ctx.fillStyle = isLight ? '#cbd5e1' : '#334155';
        ctx.fillRect(40, 60, 120, 80);
        ctx.fillRect(200, 60, 120, 80);
        ctx.fillRect(480, 60, 120, 120);

        ctx.fillStyle = isLight ? '#475569' : '#94a3b8';
        ctx.font = '11px "Plus Jakarta Sans", sans-serif';
        ctx.fillText('Store Shelf A', 65, 105);
        ctx.fillText('Store Shelf B', 225, 105);
        ctx.fillText('Checkout Counter', 495, 125);

        // 2. Update and draw counting zones
        const shoppers = shoppersRef.current;
        const currentZones = activeZonesRef.current;
        
        currentZones.forEach(zone => {
          let count = 0;
          shoppers.forEach(s => {
            if (s.activeAreaId === zone.id) count++;
          });
          zone.count = count;

          ctx.beginPath();
          zone.points.forEach((pt, idx) => {
            const px = pt[0] * 640;
            const py = pt[1] * 480;
            if (idx === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.closePath();

          ctx.fillStyle = zone.color;
          ctx.fill();
          ctx.strokeStyle = zone.borderColor;
          ctx.lineWidth = 2;
          ctx.stroke();

          // Label
          const startX = zone.points[0][0] * 640;
          const startY = zone.points[0][1] * 480;
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 11px sans-serif';
          ctx.fillText(`${zone.name} (Count: ${count})`, startX + 10, startY + 20);
        });

        // 3. Update & Draw Shoppers
        shoppers.forEach(s => {
          s.update(currentZones, handleZoneChange);
          s.draw(ctx);
        });

        // 4. Update state variables (at slower frequency if needed, or inline)
        const insideCount = shoppers.filter(s => s.activeAreaId !== null).length;
        const outsideCount = shoppers.filter(s => s.activeAreaId === null).length;
        
        // Batch state update
        setCameraData(prev => {
          if (prev.inside !== insideCount || prev.outside !== outsideCount) {
            return {
              ...prev,
              inside: insideCount,
              outside: outsideCount
            };
          }
          return prev;
        });
      }

      // Draw camera crosshair indicator
      ctx.strokeStyle = theme === 'light' ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(320, 20); ctx.lineTo(320, 460);
      ctx.moveTo(20, 240); ctx.lineTo(620, 240);
      ctx.stroke();

      animationFrameId = requestAnimationFrame(animate);
    };

    animationFrameId = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationFrameId);
  }, [cameraActive, setFps, setCpuTemp, setCameraData]);

  return shoppersRef.current;
};

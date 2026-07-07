// MuseTrack AI - Enterprise Dashboard Logic
console.log("dashboard.js file script tag is loaded!");

function initDashboard() {
    console.log("Initializing dashboard...");
    // --- Live Camera Connection State ---
    let cameraActive = false;
    let cameraData = {
        inside: 0,
        outside: 0,
        unique: 0,
        visitors: []
    };

    const liveImage = new Image();
    let liveImageLoaded = false;
    liveImage.onload = () => { liveImageLoaded = true; };
    liveImage.onerror = () => { liveImageLoaded = false; };

    // Fetch live frame at 100ms intervals
    setInterval(() => {
        if (cameraActive) {
            liveImage.src = '/api/frame?t=' + Date.now();
        }
    }, 100);

    // --- State and Configuration ---
    const state = {
        uniqueVisitors: new Set([1, 2, 3, 4, 5]),
        nextTrackId: 6,
        currentOccupancy: 0,
        fps: 30.0,
        cpuTemp: 42.2,
        logs: [],
        activeZones: [
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
        ]
    };

    // Store historical ReID database profiles
    const reidDatabase = [
        { id: 1, confidence: 0.94, lastSeen: '10s ago', area: 'Area 1', color: '#10b981', seed: 45 },
        { id: 2, confidence: 0.89, lastSeen: '1m ago', area: 'Area 2', color: '#3b82f6', seed: 12 },
        { id: 3, confidence: 0.92, lastSeen: '3m ago', area: 'Area 1', color: '#f59e0b', seed: 88 },
        { id: 4, confidence: 0.91, lastSeen: '5m ago', area: 'Area 2', color: '#8b5cf6', seed: 33 },
        { id: 5, confidence: 0.88, lastSeen: '8m ago', area: 'Area 1', color: '#ec4899', seed: 67 }
    ];

    // --- DOM Elements ---
    const kpiUniqueCount = document.getElementById('kpi-unique-count');
    const kpiInsideCount = document.getElementById('kpi-inside-count');
    const kpiOutsideCount = document.getElementById('kpi-outside-count');
    const valFps = document.getElementById('val-fps');
    const valCpu = document.getElementById('val-cpu');
    const reidGalleryList = document.getElementById('reid-gallery-list');
    const btnClearReid = document.getElementById('btn-clear-reid');
    const btnExport = document.getElementById('btn-export');
    const toastContainer = document.getElementById('toast-container');
    const btnResetPoly = document.getElementById('btn-reset-poly');
    const btnCopyJson = document.getElementById('btn-copy-json');
    const jsonOutput = document.getElementById('json-output');

    const liveCanvas = document.getElementById('live-canvas');
    const liveCtx = liveCanvas.getContext('2d');

    const polyCanvas = document.getElementById('poly-canvas');
    const polyCtx = polyCanvas.getContext('2d');

    // Update initial KPIs
    kpiUniqueCount.textContent = state.uniqueVisitors.size;

    // --- Helper: Show Notifications ---
    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `<i data-lucide="info" class="sm-icon"></i> <span>${message}</span>`;
        if (type === 'warning') {
            toast.style.borderLeftColor = '#ef4444';
        }
        toastContainer.appendChild(toast);
        lucide.createIcons();

        setTimeout(() => {
            toast.style.animation = 'toast-in 0.3s cubic-bezier(0.16, 1, 0.3, 1) reverse forwards';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // --- Helper: Generate Stylized SVG Avatars for ReID Gallery ---
    function generateAvatarSVG(seed, color) {
        return `<svg viewBox="0 0 100 100" class="reid-avatar-svg" xmlns="http://www.w3.org/2000/svg">
            <rect width="100" height="100" fill="#f1f5f9"/>
            <circle cx="50" cy="40" r="18" fill="${color}"/>
            <path d="M25 80 C25 60, 75 60, 75 80" fill="${color}"/>
            <circle cx="43" cy="38" r="2.5" fill="#fff"/>
            <circle cx="57" cy="38" r="2.5" fill="#fff"/>
            <path d="M47 48 Q50 51 53 48" stroke="#fff" stroke-width="2" fill="none"/>
            <text x="50" y="90" font-family="'Plus Jakarta Sans', sans-serif" font-size="10" font-weight="700" fill="#64748b" text-anchor="middle">#${seed}</text>
        </svg>`;
    }

    // --- Render ReID Gallery ---
    function renderReidGallery() {
        reidGalleryList.innerHTML = '';
        reidDatabase.forEach(person => {
            const card = document.createElement('div');
            card.className = 'reid-card';
            card.innerHTML = `
                <div class="reid-avatar-wrapper">
                    ${generateAvatarSVG(person.id, person.color)}
                </div>
                <div class="reid-info">
                    <div class="reid-meta-row">
                        <span class="reid-id-tag">ID: #${person.id}</span>
                        <span class="reid-time">${person.lastSeen}</span>
                    </div>
                    <div class="reid-meta-row">
                        <span class="reid-history">Zone: ${person.area}</span>
                        <span class="reid-confidence">Match Confidence: ${(person.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="reid-similarity-bar">
                        <div class="reid-similarity-fill" style="width: ${person.confidence * 100}%"></div>
                    </div>
                </div>
            `;
            reidGalleryList.appendChild(card);
        });
    }
    renderReidGallery();

    // Clear ReID Database
    btnClearReid.addEventListener('click', () => {
        reidDatabase.length = 0;
        state.uniqueVisitors.clear();
        kpiUniqueCount.textContent = 0;
        renderReidGallery();
        showToast('Local visitor profiles directory cleared.', 'warning');
    });

    // Export Data Button
    btnExport.addEventListener('click', () => {
        showToast('Exported visitor analytics log to MuseTrack_report.csv successfully.');
    });


    // --- Live Stream Simulator (Canvas 640x480) ---
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

        update() {
            this.x += this.vx;
            this.y += this.vy;

            // Bounce on boundaries
            if (this.x < 30 || this.x > 610) this.vx *= -1;
            if (this.y < 160 || this.y > 450) this.vy *= -1;

            // Track history path
            this.history.push({ x: this.x, y: this.y + this.h/2 });
            if (this.history.length > 25) this.history.shift();

            // Detect Area containment
            // Check if center bottom of bounding box is inside area polygons
            const checkPoint = [this.x / 640, (this.y + this.h) / 480];
            let currentInArea = null;

            state.activeZones.forEach(zone => {
                if (isPointInPolygon(checkPoint, zone.points)) {
                    currentInArea = zone.id;
                }
            });

            if (currentInArea !== this.activeAreaId) {
                if (currentInArea !== null) {
                    const zone = state.activeZones.find(z => z.id === currentInArea);
                    showToast(`Visitor #${this.id} entered ${zone.name}`);
                }
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
            ctx.strokeRect(this.x - this.w/2, this.y - this.h/2, this.w, this.h);

            // Draw header tag
            ctx.fillStyle = this.color;
            ctx.fillRect(this.x - this.w/2 - 1, this.y - this.h/2 - 25, this.w + 2, 25);

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 11px "Plus Jakarta Sans", sans-serif';
            ctx.fillText(`Visitor #${this.id}`, this.x - this.w/2 + 6, this.y - this.h/2 - 8);
        }
    }

    // Initialize list of active shoppers
    const shoppers = [
        new Shopper(1, '#10b981'),
        new Shopper(2, '#3b82f6'),
        new Shopper(3, '#f59e0b')
    ];

    // Helper: Raycasting algorithm to check point inside polygon
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

    // --- Main Simulator Frame Loop ---
    function animateLiveStream() {
        if (cameraActive && liveImageLoaded) {
            // Draw the live camera frame on the canvas
            liveCtx.drawImage(liveImage, 0, 0, 640, 480);
        } else {
            // 1. Draw Simulated Retail Background
            liveCtx.fillStyle = '#1e293b';
            liveCtx.fillRect(0, 0, 640, 480);

            // Store shelves sketches
            liveCtx.fillStyle = '#334155';
            liveCtx.fillRect(40, 60, 120, 80);
            liveCtx.fillRect(200, 60, 120, 80);
            liveCtx.fillRect(480, 60, 120, 120);
            
            liveCtx.fillStyle = '#475569';
            liveCtx.font = '11px "Plus Jakarta Sans", sans-serif';
            liveCtx.fillText('Store Shelf A', 65, 105);
            liveCtx.fillText('Store Shelf B', 225, 105);
            liveCtx.fillText('Checkout Counter', 495, 125);
        }

        // 2. Draw Counting Zones Outlines
        if (!cameraActive) {
            state.activeZones.forEach(zone => {
                // Count shoppers in zone
                let inCount = 0;
                shoppers.forEach(s => {
                    if (s.activeAreaId === zone.id) inCount++;
                });
                zone.count = inCount;

                liveCtx.beginPath();
                zone.points.forEach((pt, idx) => {
                    const px = pt[0] * 640;
                    const py = pt[1] * 480;
                    if (idx === 0) liveCtx.moveTo(px, py);
                    else liveCtx.lineTo(px, py);
                });
                liveCtx.closePath();
                
                liveCtx.fillStyle = zone.color;
                liveCtx.fill();
                liveCtx.strokeStyle = zone.borderColor;
                liveCtx.lineWidth = 2;
                liveCtx.stroke();

                // Label the Area
                const startX = zone.points[0][0] * 640;
                const startY = zone.points[0][1] * 480;
                liveCtx.fillStyle = '#ffffff';
                liveCtx.font = 'bold 11px sans-serif';
                liveCtx.fillText(`${zone.name} (Count: ${zone.count})`, startX + 10, startY + 20);
            });
        }

        // 3. Update & Draw Shoppers
        if (!cameraActive) {
            shoppers.forEach(s => {
                s.update();
                s.draw(liveCtx);
            });
        }

        // 4. Update Inside/Outside Traffic KPIs
        if (!cameraActive) {
            const insideCount = shoppers.filter(s => s.activeAreaId !== null).length;
            const outsideCount = shoppers.filter(s => s.activeAreaId === null).length;
            kpiInsideCount.textContent = insideCount;
            kpiOutsideCount.textContent = outsideCount;
            document.getElementById('val-feed-status').textContent = 'Feed Simulated from AI Camera Stream';
            document.getElementById('val-feed-status').style.color = '#ffffff';
        } else {
            document.getElementById('val-feed-status').textContent = 'LIVE CAMERA STREAM ACTIVE';
            document.getElementById('val-feed-status').style.color = '#10b981';
        }

        // 5. Update system settings randomly
        state.fps = cameraActive ? (29.9 + Math.random() * 0.2) : (29.5 + Math.random());
        state.cpuTemp = cameraActive ? (44.5 + Math.random() * 0.4) : (41.8 + Math.random() * 0.8);
        valFps.textContent = state.fps.toFixed(1);
        valCpu.textContent = state.cpuTemp.toFixed(1);

        // Trigger dynamic additions/exits occasionally to simulate ReID flow (only if camera is offline)
        if (!cameraActive && Math.random() < 0.003) {
            simulateReidTraffic();
        }

        // Draw camera crosshair indicator
        liveCtx.strokeStyle = 'rgba(255,255,255,0.15)';
        liveCtx.lineWidth = 1;
        liveCtx.beginPath();
        liveCtx.moveTo(320, 20); liveCtx.lineTo(320, 460);
        liveCtx.moveTo(20, 240); liveCtx.lineTo(620, 240);
        liveCtx.stroke();

        requestAnimationFrame(animateLiveStream);
    }

    // --- ReID Matching Simulation ---
    function simulateReidTraffic() {
        if (shoppers.length >= 5) {
            // Remove a shopper
            const removed = shoppers.shift();
            showToast(`Visitor #${removed.id} exited coverage zone.`);
        } else {
            // Introduce a shopper
            // Decide if new or returning (ReID Match)
            const isReturning = Math.random() > 0.4 && reidDatabase.length > 0;
            if (isReturning) {
                const randomHistorical = reidDatabase[Math.floor(Math.random() * reidDatabase.length)];
                // Check if already active
                if (!shoppers.some(s => s.id === randomHistorical.id)) {
                    const newShopper = new Shopper(randomHistorical.id, randomHistorical.color);
                    shoppers.push(newShopper);
                    
                    // Trigger ReID Toast alert with similarity confidence
                    const simScore = (0.85 + Math.random() * 0.14).toFixed(3);
                    showToast(`Welcome back visitor #${randomHistorical.id} (${(simScore * 100).toFixed(1)}% Match Similarity)`, 'success');
                    
                    // Update gallery
                    randomHistorical.lastSeen = 'Just Now';
                    randomHistorical.confidence = parseFloat(simScore);
                    renderReidGallery();
                }
            } else {
                // Completely new person
                const newId = state.nextTrackId++;
                state.uniqueVisitors.add(newId);
                kpiUniqueCount.textContent = state.uniqueVisitors.size;

                const colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#ef4444', '#06b6d4'];
                const randomColor = colors[newId % colors.length];

                const newShopper = new Shopper(newId, randomColor);
                shoppers.push(newShopper);

                // Add to database
                reidDatabase.unshift({
                    id: newId,
                    confidence: 0.95,
                    lastSeen: 'Just Now',
                    area: 'Area 1',
                    color: randomColor
                });
                if (reidDatabase.length > 7) reidDatabase.pop();
                renderReidGallery();

                showToast(`New profile registered for visitor #${newId}.`);
            }
        }
    }

    // Start Live Stream Canvas simulation
    animateLiveStream();


    // --- Visual Area Configurator (Canvas 400x300) ---
    // User can drag nodes of Area 1 and Area 2 to edit `areas.json` coordinates.
    let dragInfo = null; // { zoneIdx, pointIdx }

    function drawPolyConfig() {
        // Draw dark background representing the camera view
        polyCtx.fillStyle = '#0f172a';
        polyCtx.fillRect(0, 0, 400, 300);

        // Grid lines
        polyCtx.strokeStyle = 'rgba(255,255,255,0.05)';
        polyCtx.lineWidth = 1;
        for (let i = 25; i < 400; i += 25) {
            polyCtx.beginPath();
            polyCtx.moveTo(i, 0); polyCtx.lineTo(i, 300);
            polyCtx.stroke();
        }
        for (let i = 25; i < 300; i += 25) {
            polyCtx.beginPath();
            polyCtx.moveTo(0, i); polyCtx.lineTo(400, i);
            polyCtx.stroke();
        }

        // Draw areas
        state.activeZones.forEach((zone, zIdx) => {
            polyCtx.beginPath();
            zone.points.forEach((pt, pIdx) => {
                const px = pt[0] * 400;
                const py = pt[1] * 300;
                if (pIdx === 0) polyCtx.moveTo(px, py);
                else polyCtx.lineTo(px, py);
            });
            polyCtx.closePath();

            // Fill area
            polyCtx.fillStyle = zone.color;
            polyCtx.fill();
            polyCtx.strokeStyle = zone.borderColor;
            polyCtx.lineWidth = 2;
            polyCtx.stroke();

            // Draw clickable vertices
            zone.points.forEach((pt, pIdx) => {
                const px = pt[0] * 400;
                const py = pt[1] * 300;
                
                polyCtx.beginPath();
                polyCtx.arc(px, py, 6, 0, Math.PI * 2);
                polyCtx.fillStyle = '#ffffff';
                polyCtx.fill();
                polyCtx.strokeStyle = zone.borderColor;
                polyCtx.lineWidth = 2;
                polyCtx.stroke();

                // Vertex label
                polyCtx.fillStyle = '#94a3b8';
                polyCtx.font = '8px monospace';
                polyCtx.fillText(`[${pt[0].toFixed(2)}, ${pt[1].toFixed(2)}]`, px + 8, py + 3);
            });

            // Write Zone ID in center
            let sumX = 0, sumY = 0;
            zone.points.forEach(pt => { sumX += pt[0]; sumY += pt[1]; });
            const cx = (sumX / zone.points.length) * 400;
            const cy = (sumY / zone.points.length) * 300;

            polyCtx.fillStyle = '#ffffff';
            polyCtx.font = 'bold 10px "Plus Jakarta Sans", sans-serif';
            polyCtx.fillText(`Area ${zone.id}`, cx - 15, cy + 3);
        });

        // Update Coordinate Output pre
        updateJsonOutput();
    }

    function updateJsonOutput() {
        const output = state.activeZones.map(zone => ({
            points: zone.points.map(pt => [
                parseFloat(pt[0].toFixed(3)),
                parseFloat(pt[1].toFixed(3))
            ])
        }));
        if (jsonOutput) {
            jsonOutput.textContent = JSON.stringify(output, null, 4);
        }
    }

    // Poly Editor Mouse Interactions
    polyCanvas.addEventListener('mousedown', (e) => {
        const rect = polyCanvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Check if cursor clicked a vertex (with 10px buffer)
        for (let zIdx = 0; zIdx < state.activeZones.length; zIdx++) {
            const zone = state.activeZones[zIdx];
            for (let pIdx = 0; pIdx < zone.points.length; pIdx++) {
                const pt = zone.points[pIdx];
                const px = pt[0] * 400;
                const py = pt[1] * 300;
                const dist = Math.hypot(mx - px, my - py);
                if (dist <= 10) {
                    dragInfo = { zoneIdx: zIdx, pointIdx: pIdx };
                    return;
                }
            }
        }
    });

    polyCanvas.addEventListener('mousemove', (e) => {
        if (!dragInfo) return;
        const rect = polyCanvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Bound check inside canvas
        const nx = Math.max(0, Math.min(1.0, mx / 400));
        const ny = Math.max(0, Math.min(1.0, my / 300));

        state.activeZones[dragInfo.zoneIdx].points[dragInfo.pointIdx] = [nx, ny];
        drawPolyConfig();
    });

    window.addEventListener('mouseup', () => {
        dragInfo = null;
    });

    // Reset polygons back to default configuration
    btnResetPoly.addEventListener('click', () => {
        state.activeZones[0].points = [
            [0.10, 0.25],
            [0.48, 0.25],
            [0.48, 0.75],
            [0.10, 0.75]
        ];
        state.activeZones[1].points = [
            [0.55, 0.25],
            [0.90, 0.25],
            [0.90, 0.75],
            [0.55, 0.75]
        ];
        drawPolyConfig();
        showToast('Restored default counting zones.');
    });

    // Copy to clipboard JSON coordinates
    if (btnCopyJson && jsonOutput) {
        btnCopyJson.addEventListener('click', () => {
            navigator.clipboard.writeText(jsonOutput.textContent).then(() => {
                showToast('JSON coordinates copied to clipboard!');
            }).catch(err => {
                showToast('Failed to copy to clipboard.', 'warning');
            });
        });
    }

    // Draw initial configurations
    drawPolyConfig();


    // --- Traffic charts setup ---
    const ctxChart = document.getElementById('trafficChart').getContext('2d');
    const chartData = {
        labels: ['4m ago', '3m ago', '2m ago', '1m ago', 'Active'],
        datasets: [
            {
                label: 'Area 1 (Entrance)',
                data: [1, 2, 0, 1, 2],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            },
            {
                label: 'Area 2 (Display Counter)',
                data: [0, 1, 1, 2, 1],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }
        ]
    };

    const trafficChart = new Chart(ctxChart, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 5,
                    ticks: {
                        stepSize: 1,
                        color: '#94a3b8',
                        font: { size: 10 }
                    },
                    grid: {
                        color: '#f1f5f9'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 10 }
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    // Periodically update chart data to match simulator values
    setInterval(() => {
        if (cameraActive) {
            // Compute counts for active zones from live camera payload
            state.activeZones[0].count = cameraData.visitors.filter(v => v.area && v.area.includes('1')).length;
            state.activeZones[1].count = cameraData.visitors.filter(v => v.area && v.area.includes('2')).length;
        }

        // Shift values
        chartData.datasets[0].data.shift();
        chartData.datasets[0].data.push(state.activeZones[0].count);

        chartData.datasets[1].data.shift();
        chartData.datasets[1].data.push(state.activeZones[1].count);

        trafficChart.update();

        // Update bottom table indicators
        document.getElementById('zone-1-count').textContent = `${state.activeZones[0].count} Persons`;
        document.getElementById('zone-2-count').textContent = `${state.activeZones[1].count} Persons`;
    }, 2000);

    // --- Dynamic HTTP Polling for Camera Stats ---
    function pollCameraData() {
        fetch('/api/data')
            .then(res => res.json())
            .then(data => {
                const wasActive = cameraActive;
                cameraActive = data.active;
                
                if (cameraActive) {
                    cameraData = data;
                    
                    // Directly display camera counters
                    kpiUniqueCount.textContent = data.unique;
                    kpiInsideCount.textContent = data.inside;
                    kpiOutsideCount.textContent = data.outside;

                    if (!wasActive) {
                        showToast('Live connection established: Receiving feed from camera script!', 'success');
                    }

                    // Sync shoppers array with camera visitors
                    const cameraIds = data.visitors.map(v => v.id);
                    
                    // Remove shooters not present in live frame
                    for (let i = shoppers.length - 1; i >= 0; i--) {
                        if (!cameraIds.includes(shoppers[i].id)) {
                            shoppers.splice(i, 1);
                        }
                    }
                    
                    // Spawn/update shopper positions on canvas
                    data.visitors.forEach(v => {
                        let existing = shoppers.find(s => s.id === v.id);
                        if (!existing) {
                            const newColor = v.color || '#10b981';
                            existing = new Shopper(v.id, newColor);
                            shoppers.push(existing);
                        }
                        
                        // Map zone index
                        if (v.area && (v.area.includes('Area 1') || v.area.includes('Area 2'))) {
                            existing.activeAreaId = v.area.includes('1') ? 1 : 2;
                        } else {
                            existing.activeAreaId = null;
                        }
                    });

                    // Prepend new profiles to ReID list
                    data.visitors.forEach(v => {
                        if (!reidDatabase.some(p => p.id === v.id)) {
                            reidDatabase.unshift({
                                id: v.id,
                                confidence: v.confidence,
                                lastSeen: 'Just Now',
                                area: v.area || 'Main Coverage',
                                color: v.color || '#10b981'
                            });
                            if (reidDatabase.length > 7) reidDatabase.pop();
                            renderReidGallery();
                            showToast(`Profile Recognized: Visitor #${v.id} detected on camera.`, 'success');
                        }
                    });
                } else if (wasActive) {
                    showToast('Camera script went offline. Switching to simulation mode.', 'warning');
                }
                updateCameraBadge();
            })
            .catch(() => {
                cameraActive = false;
                updateCameraBadge();
            });
    }

    function updateCameraBadge() {
        const navCamera = document.getElementById('nav-camera-streams');
        if (navCamera) {
            let badge = navCamera.querySelector('.nav-item-badge-live');
            if (cameraActive) {
                if (!badge) {
                    badge = document.createElement('span');
                    badge.className = 'nav-item-badge-live';
                    badge.textContent = 'Live';
                    navCamera.appendChild(badge);
                }
            } else {
                if (badge) {
                    badge.remove();
                }
            }
        }
    }

    // --- Sidebar Navigation Scroll and Active State ---
    const navItems = {
        'nav-control-panel': document.querySelector('.header'),
        'nav-camera-streams': document.querySelector('.stream-panel'),
        'nav-zone-configurator': document.querySelector('.zone-config-panel')
    };

    Object.keys(navItems).forEach(id => {
        const btn = document.getElementById(id);
        const target = navItems[id];
        if (btn && target) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('Navigation clicked:', id);
                
                if (id === 'nav-camera-streams') {
                    // Open the configuration modal!
                    const modal = document.getElementById('camera-config-modal');
                    if (modal) {
                        modal.classList.add('active');
                    }
                    return;
                }

                // Update active state in sidebar
                document.querySelectorAll('.nav-menu .nav-item').forEach(item => {
                    item.classList.remove('active');
                });
                btn.classList.add('active');

                // Smooth scroll to panel using offset calculation
                const y = target.getBoundingClientRect().top + window.pageYOffset - 20;
                window.scrollTo({ top: y, behavior: 'smooth' });
            });
        } else {
            console.warn('Navigation setup failed for ID:', id, 'button:', btn, 'target:', target);
        }
    });

    // --- Stream Configuration Form & Modal Handling ---
    const cameraModal = document.getElementById('camera-config-modal');
    const btnConfigureCamera = document.getElementById('btn-configure-camera');
    const btnCloseConfigModal = document.getElementById('btn-close-config-modal');
    const btnCancelConfig = document.getElementById('btn-cancel-config');
    const cameraConfigForm = document.getElementById('camera-config-form');

    const streamTypeSelect = document.getElementById('stream-type');
    const uriGroup = document.getElementById('uri-group');
    const uriLabel = document.getElementById('uri-label');
    const uriInput = document.getElementById('stream-uri');
    const ipGroup = document.getElementById('ip-group');
    const portGroup = document.getElementById('port-group');

    // Open Modal from sidebar or button
    [btnConfigureCamera].forEach(btn => {
        if (btn) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                if (cameraModal) {
                    cameraModal.classList.add('active');
                }
            });
        }
    });

    // Close Modal
    [btnCloseConfigModal, btnCancelConfig].forEach(btn => {
        if (btn) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                if (cameraModal) {
                    cameraModal.classList.remove('active');
                }
            });
        }
    });

    if (cameraModal) {
        cameraModal.addEventListener('click', (e) => {
            if (e.target === cameraModal) {
                cameraModal.classList.remove('active');
            }
        });
    }

    // Dynamic Select Type logic
    if (streamTypeSelect) {
        streamTypeSelect.addEventListener('change', (e) => {
            const val = e.target.value;
            console.log('Stream connection type switched to:', val);

            // Default state: show URI group, hide IP/Port groups
            uriGroup.style.display = 'flex';
            ipGroup.style.display = 'none';
            portGroup.style.display = 'none';
            uriInput.required = true;

            if (val === 'rtsp') {
                uriLabel.textContent = 'RTSP Connection Link';
                uriInput.placeholder = 'rtsp://admin:password@192.168.1.100:554/stream1';
            } else if (val === 'ip') {
                uriGroup.style.display = 'none';
                uriInput.required = false;
                ipGroup.style.display = 'flex';
                portGroup.style.display = 'flex';
            } else if (val === 'api') {
                uriLabel.textContent = 'HTTP API / REST Feed URL';
                uriInput.placeholder = 'http://192.168.1.100:8000/api/frame';
            } else if (val === 'webcam') {
                uriLabel.textContent = 'USB Webcam Index';
                uriInput.placeholder = '0';
            } else if (val === 'simulated') {
                uriGroup.style.display = 'none';
                uriInput.required = false;
            }
        });
    }

    // Form Submit handling
    if (cameraConfigForm) {
        cameraConfigForm.addEventListener('submit', (e) => {
            e.preventDefault();

            // Close modal
            if (cameraModal) {
                cameraModal.classList.remove('active');
            }

            const type = streamTypeSelect.value;
            let sourceDetail = '';

            if (type === 'rtsp') {
                sourceDetail = uriInput.value;
            } else if (type === 'ip') {
                sourceDetail = document.getElementById('camera-ip').value + ':' + document.getElementById('camera-port').value;
            } else if (type === 'api') {
                sourceDetail = uriInput.value;
            } else if (type === 'webcam') {
                sourceDetail = 'Local Webcam Device ' + uriInput.value;
            } else {
                sourceDetail = 'Simulated Vision Engine';
            }

            showToast(`Connecting to ${type.toUpperCase()} camera source: ${sourceDetail}...`, 'success');

            const feedStatus = document.getElementById('val-feed-status');
            if (feedStatus) {
                feedStatus.textContent = 'CONNECTING TO CAMERA SOURCE...';
                feedStatus.style.color = '#f59e0b';
            }

            setTimeout(() => {
                showToast(`Connected to ${type.toUpperCase()} camera source successfully!`, 'success');
                if (feedStatus) {
                    feedStatus.textContent = `LIVE CAMERA ACTIVE: ${type.toUpperCase()} (${sourceDetail})`;
                    feedStatus.style.color = '#10b981';
                }
            }, 1800);
        });
    }

    // Start polling every 500ms
    setInterval(pollCameraData, 500);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDashboard);
} else {
    initDashboard();
}

import React, { useEffect, useState } from 'react';
import { TrendingUp } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export const TrafficChart = () => {
  const { activeZones, cameraActive, cameraData } = useDashboard();
  const [timeframe, setTimeframe] = useState('realtime');
  
  // Local history state
  const [area1History, setArea1History] = useState([1, 2, 0, 1, 2]);
  const [area2History, setArea2History] = useState([0, 1, 1, 2, 1]);
  const [labels, setLabels] = useState(['4m ago', '3m ago', '2m ago', '1m ago', 'Active']);

  // Periodically update chart history to match current active counts
  useEffect(() => {
    const interval = setInterval(() => {
      let area1Count = 0;
      let area2Count = 0;

      if (cameraActive) {
        // Map from live camera payload
        const visitors = cameraData.visitors || [];
        area1Count = visitors.filter(v => v.area && v.area.includes('1')).length;
        area2Count = visitors.filter(v => v.area && v.area.includes('2')).length;
      } else {
        // Map from local simulated zones
        area1Count = activeZones[0]?.count || 0;
        area2Count = activeZones[1]?.count || 0;
      }

      setArea1History(prev => [...prev.slice(1), area1Count]);
      setArea2History(prev => [...prev.slice(1), area2Count]);
    }, 2000);

    return () => clearInterval(interval);
  }, [activeZones, cameraActive, cameraData]);

  // Chart data configuration
  const data = {
    labels,
    datasets: [
      {
        label: 'Area 1 (Entrance)',
        data: area1History,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#10b981',
        pointBorderColor: '#ffffff',
        pointHoverRadius: 6
      },
      {
        label: 'Area 2 (Display Counter)',
        data: area2History,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#3b82f6',
        pointBorderColor: '#ffffff',
        pointHoverRadius: 6
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: '#0f172a',
        titleFont: { family: "'Plus Jakarta Sans', sans-serif", size: 11 },
        bodyFont: { family: "'Plus Jakarta Sans', sans-serif", size: 11 },
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1
      }
    },
    scales: {
      y: {
        min: 0,
        max: 5,
        ticks: {
          stepSize: 1,
          color: '#94a3b8',
          font: { size: 10, family: "'Plus Jakarta Sans', sans-serif" }
        },
        grid: {
          color: 'rgba(148, 163, 184, 0.1)'
        }
      },
      x: {
        ticks: {
          color: '#94a3b8',
          font: { size: 10, family: "'Plus Jakarta Sans', sans-serif" }
        },
        grid: {
          display: false
        }
      }
    }
  };

  const area1Current = area1History[area1History.length - 1];
  const area2Current = area2History[area2History.length - 1];

  return (
    <div className="panel chart-panel">
      <div className="panel-header">
        <div className="panel-header-title">
          <TrendingUp size={18} className="header-icon green" />
          <h3>Traffic & Flow Analysis</h3>
        </div>
        <div className="actions">
          <select 
            className="select-sm" 
            value={timeframe} 
            onChange={(e) => setTimeframe(e.target.value)}
          >
            <option value="realtime">Real-time (Last 5m)</option>
            <option value="hourly">Hourly Trends</option>
          </select>
        </div>
      </div>
      <div className="panel-body chart-body">
        <div className="chart-wrapper" style={{ height: '200px', position: 'relative' }}>
          <Line data={data} options={options} />
        </div>
        <div className="zone-metrics-details" style={{ marginTop: '16px' }}>
          <div className="zone-row">
            <div className="zone-name">
              <span className="color-indicator" style={{ backgroundColor: 'rgba(16, 185, 129, 1)' }}></span>
              <span>Area 1 (Entrance Zone)</span>
            </div>
            <div className="zone-count" id="zone-1-count">{area1Current} Persons</div>
          </div>
          <div className="zone-row">
            <div className="zone-name">
              <span className="color-indicator" style={{ backgroundColor: 'rgba(59, 130, 246, 1)' }}></span>
              <span>Area 2 (Display Counter)</span>
            </div>
            <div className="zone-count" id="zone-2-count">{area2Current} Persons</div>
          </div>
        </div>
      </div>
    </div>
  );
};

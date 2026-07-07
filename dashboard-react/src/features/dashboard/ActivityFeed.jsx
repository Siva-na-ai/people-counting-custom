import React from 'react';
import { Activity } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';

export const ActivityFeed = () => {
  const { activityLogs } = useDashboard();

  const getIndicatorColor = (type) => {
    switch (type) {
      case 'success':
        return '#10b981'; // Green
      case 'warning':
        return '#ef4444'; // Red
      default:
        return '#3b82f6'; // Blue / Info
    }
  };

  return (
    <div className="panel activity-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="panel-header">
        <div className="panel-header-title">
          <Activity size={18} className="header-icon green" />
          <h3>Real-time Activity Feed</h3>
        </div>
      </div>
      <div className="panel-body activity-container" style={{ flex: '1', overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
        <div className="activity-desc" style={{ marginBottom: '12px', fontSize: '12px', color: '#94a3b8' }}>
          Live system events and tracking notifications captured from the AI processing engine.
        </div>
        <div className="activity-list" style={{ flex: '1', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {activityLogs.length === 0 ? (
            <div style={{ padding: '24px', textAlign: 'center', color: '#64748b', fontSize: '13px' }}>
              No system activity logs yet.
            </div>
          ) : (
            activityLogs.map(log => (
              <div 
                key={log.id} 
                className="activity-item" 
                style={{ 
                  display: 'flex', 
                  gap: '10px', 
                  padding: '8px 10px', 
                  backgroundColor: 'rgba(30, 41, 59, 0.5)', 
                  borderRadius: '4px',
                  borderLeft: `3px solid ${getIndicatorColor(log.type)}`,
                  fontSize: '11px',
                  lineHeight: '1.4'
                }}
              >
                <span className="activity-time" style={{ color: '#64748b', fontWeight: '600', whiteSpace: 'nowrap' }}>
                  [{log.time}]
                </span>
                <span className="activity-text" style={{ color: '#e2e8f0' }}>
                  {log.text}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

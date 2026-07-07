import React from 'react';
import { ShieldCheck, Trash2 } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';

const ReIDAvatar = ({ id, color }) => (
  <svg viewBox="0 0 100 100" className="reid-avatar-svg" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', height: '100%', borderRadius: '6px' }}>
    <rect width="100" height="100" fill="#0f172a" />
    <circle cx="50" cy="40" r="18" fill={color} />
    <path d="M25 80 C25 60, 75 60, 75 80" fill={color} />
    <circle cx="43" cy="38" r="2.5" fill="#fff" />
    <circle cx="57" cy="38" r="2.5" fill="#fff" />
    <path d="M47 48 Q50 51 53 48" stroke="#fff" strokeWidth="2" fill="none" />
    <text x="50" y="90" fontFamily="'Plus Jakarta Sans', sans-serif" fontSize="10" fontWeight="700" fill="#94a3b8" textAnchor="middle">#{id}</text>
  </svg>
);

export const ReIDGallery = () => {
  const { reidDatabase, clearReidGallery } = useDashboard();

  return (
    <div className="panel reid-panel">
      <div className="panel-header">
        <div className="panel-header-title">
          <ShieldCheck size={18} className="header-icon green" />
          <h3>Identified Visitor Profiles</h3>
        </div>
        <div className="actions">
          <button className="btn-icon" title="Clear Gallery" onClick={clearReidGallery}>
            <Trash2 size={16} />
          </button>
        </div>
      </div>
      <div className="panel-body gallery-container" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div className="reid-desc" style={{ marginBottom: '12px', fontSize: '12px', color: '#94a3b8' }}>
          Shows historical visitor profiles matched by similarity comparison against the directory.
        </div>
        <div className="reid-list" id="reid-gallery-list" style={{ flex: '1', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {reidDatabase.length === 0 ? (
            <div style={{ padding: '24px', textAlign: 'center', color: '#64748b', fontSize: '13px' }}>
              No visitor profiles registered.
            </div>
          ) : (
            reidDatabase.map(person => (
              <div className="reid-card" key={person.id} style={{ display: 'flex', gap: '12px', padding: '10px', backgroundColor: '#1e293b', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div className="reid-avatar-wrapper" style={{ width: '45px', height: '45px', flexShrink: 0 }}>
                  <ReIDAvatar id={person.id} color={person.color} />
                </div>
                <div className="reid-info" style={{ flex: '1', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <div className="reid-meta-row" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px', fontSize: '11px', fontWeight: '600' }}>
                    <span className="reid-id-tag" style={{ color: '#10b981' }}>ID: #{person.id}</span>
                    <span className="reid-time" style={{ color: '#64748b' }}>{person.lastSeen}</span>
                  </div>
                  <div className="reid-meta-row" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '10px', color: '#94a3b8' }}>
                    <span className="reid-history">Zone: {person.area}</span>
                    <span className="reid-confidence">Match Confidence: {(person.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="reid-similarity-bar" style={{ height: '4px', backgroundColor: '#0f172a', borderRadius: '2px', overflow: 'hidden' }}>
                    <div 
                      className="reid-similarity-fill" 
                      style={{ 
                        width: `${person.confidence * 100}%`, 
                        height: '100%', 
                        backgroundColor: '#10b981', 
                        borderRadius: '2px',
                        transition: 'width 0.3s ease'
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

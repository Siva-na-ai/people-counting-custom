import React from 'react';
import { LayoutDashboard, Camera, Map, Download, Fingerprint, Sun, Moon } from 'lucide-react';
import { useDashboard, DashboardProvider } from './context/DashboardContext';
import { useCameraData } from './hooks/useCameraData';
import { KPISection } from './features/dashboard/KPISection';
import { LiveStream } from './features/dashboard/LiveStream';
import { ReIDGallery } from './features/dashboard/ReIDGallery';
import { AreaConfigurator } from './features/dashboard/AreaConfigurator';
import { TrafficChart } from './features/dashboard/TrafficChart';
import { ActivityFeed } from './features/dashboard/ActivityFeed';
import { CameraConfigModal } from './features/settings/CameraConfigModal';
import { ToastContainer } from './components/Toast';

const DashboardApp = () => {
  const { 
    cameraActive, 
    setShowConfigModal, 
    addToast, 
    addActivityLog,
    theme,
    setTheme
  } = useDashboard();

  // Initialize camera data polling
  useCameraData();

  // Apply theme class to document body
  React.useEffect(() => {
    document.body.className = `${theme}-theme`;
  }, [theme]);

  const handleExportData = () => {
    window.location.href = '/api/export';
    addToast('Generating MuseTrack Analytics Excel Report...', 'info');
    addActivityLog('User requested visitor analytics Excel report download.', 'info');
  };

  const scrollToSection = (id) => {
    if (id === 'camera-streams') {
      setShowConfigModal(true);
      return;
    }

    const element = document.getElementById(id);
    if (element) {
      const y = element.getBoundingClientRect().top + window.pageYOffset - 20;
      window.scrollTo({ top: y, behavior: 'smooth' });
    }
  };

  return (
    <div className={`app-container ${theme}-theme`}>
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">
            <Fingerprint className="logo-icon" size={24} />
          </div>
          <div className="brand-text">
            <h2>MuseTrack</h2>
            <span>Vision Analytics</span>
          </div>
        </div>
        
        <nav className="nav-menu">
          <a 
            href="#" 
            className="nav-item active" 
            onClick={(e) => { e.preventDefault(); scrollToSection('control-panel'); }}
          >
            <LayoutDashboard size={18} />
            <span>Control Panel</span>
          </a>
          <a 
            href="#" 
            className="nav-item" 
            onClick={(e) => { e.preventDefault(); scrollToSection('camera-streams'); }}
          >
            <Camera size={18} />
            <span>Camera Streams</span>
            {cameraActive && <span className="nav-item-badge-live">Live</span>}
          </a>
          <a 
            href="#" 
            className="nav-item" 
            onClick={(e) => { e.preventDefault(); scrollToSection('zone-configurator'); }}
          >
            <Map size={18} />
            <span>Zone Configurator</span>
          </a>
        </nav>

        <div className="sidebar-footer">
          <div className="user-profile">
            <div className="avatar">EX</div>
            <div className="user-info">
              <h4>Executive User</h4>
              <span>Administrator</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content" id="control-panel">
        {/* Header */}
        <header className="header">
          <div className="header-left">
            <h1>Museum Occupancy Dashboard</h1>
          </div>
          <div className="header-right" style={{ display: 'flex', gap: '12px' }}>
            <button 
              className="btn btn-outline" 
              onClick={() => setTheme(prev => prev === 'light' ? 'dark' : 'light')}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
            >
              {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
              <span>{theme === 'light' ? 'Dark Mode' : 'Light Mode'}</span>
            </button>
            <button className="btn btn-primary" onClick={handleExportData}>
              <Download size={16} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: '4px' }} /> 
              Export Data
            </button>
          </div>
        </header>

        {/* Metrics Overview */}
        <KPISection />

        {/* Live Video & ReID Grid */}
        <section className="grid-two-cols mt-6">
          <LiveStream />
          <ReIDGallery />
        </section>

        {/* Configurator & Chart Grid */}
        <section className="grid-two-cols mt-6" id="zone-configurator">
          <AreaConfigurator />
          <TrafficChart />
        </section>

        {/* Activity Feed Section */}
        <section className="mt-6" style={{ height: '300px', marginBottom: '24px' }}>
          <ActivityFeed />
        </section>
      </main>

      {/* Settings Modal */}
      <CameraConfigModal />

      {/* Toast Alert Notifications */}
      <ToastContainer />
    </div>
  );
};

export default function App() {
  return (
    <DashboardProvider>
      <DashboardApp />
    </DashboardProvider>
  );
}

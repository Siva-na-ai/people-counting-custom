import React from 'react';
import { Info, AlertTriangle, CheckCircle } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';

export const ToastItem = ({ message, type = 'success' }) => {
  const getIcon = () => {
    switch (type) {
      case 'warning':
        return <AlertTriangle size={14} className="sm-icon" />;
      case 'info':
        return <Info size={14} className="sm-icon" />;
      default:
        return <CheckCircle size={14} className="sm-icon" />;
    }
  };

  const getStyle = () => {
    if (type === 'warning') {
      return { borderLeftColor: '#ef4444' };
    }
    if (type === 'info') {
      return { borderLeftColor: '#3b82f6' };
    }
    return {};
  };

  return (
    <div className="toast" style={getStyle()}>
      {getIcon()}
      <span>{message}</span>
    </div>
  );
};

export const ToastContainer = () => {
  const { toasts } = useDashboard();

  return (
    <div className="toast-container" id="toast-container">
      {toasts.map(toast => (
        <ToastItem key={toast.id} message={toast.message} type={toast.type} />
      ))}
    </div>
  );
};

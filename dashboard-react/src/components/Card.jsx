import React from 'react';

export const Card = ({ children, className = '' }) => {
  return (
    <div className={`panel ${className}`}>
      {children}
    </div>
  );
};

export const KPICard = ({ title, value, icon, trend, footer, iconColorClass = 'green' }) => {
  return (
    <div className="kpi-card">
      <div className="kpi-header">
        <span class="kpi-title">{title}</span>
        <div className={`kpi-icon-wrapper ${iconColorClass}`}>
          {icon}
        </div>
      </div>
      <div className="kpi-value-container">
        <h3 className="kpi-value">{value}</h3>
        {trend && (
          <span className={`kpi-trend ${trend.type}`}>
            {trend.icon} {trend.text}
          </span>
        )}
      </div>
      {footer && <div className="kpi-footer">{footer}</div>}
    </div>
  );
};

import React from 'react';
import { UserCheck, Users, Clock, UserMinus, TrendingUp, Activity } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import { KPICard } from '../../components/Card';

export const KPISection = () => {
  const { cameraData } = useDashboard();

  return (
    <section className="kpi-grid">
      <KPICard
        title="Total Unique Visitors"
        value={cameraData.unique}
        icon={<UserCheck size={20} />}
        iconColorClass="green"
        trend={{
          type: 'up',
          icon: <TrendingUp size={12} />,
          text: '+12% vs last hour'
        }}
        footer="De-duplicated via Vision Profiles"
      />

      <KPICard
        title="Current Inside Count"
        value={cameraData.inside}
        icon={<Users size={20} />}
        iconColorClass="pulse-green"
        trend={{
          type: 'up',
          icon: <Activity size={12} />,
          text: 'Active inside zones'
        }}
        footer="Engaged customer traffic"
      />

      <KPICard
        title="Avg. Customer Dwell Time"
        value="45.2m"
        icon={<Clock size={20} />}
        iconColorClass="green"
        trend={{
          type: 'up',
          icon: <TrendingUp size={12} />,
          text: '+8.4% this week'
        }}
        footer="Average stay duration"
      />

      <KPICard
        title="Current Outside Count"
        value={cameraData.outside}
        icon={<UserMinus size={20} />}
        iconColorClass="green"
        trend={{
          type: 'neutral',
          icon: <Activity size={12} />,
          text: 'Outside active zones'
        }}
        footer="Passersby/window traffic"
      />
    </section>
  );
};

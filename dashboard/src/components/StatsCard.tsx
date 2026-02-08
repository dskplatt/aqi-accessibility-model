'use client';

interface StatsCardProps {
  title: string;
  value: string;
  description: string;
  gradient: string;
}

export default function StatsCard({ title, value, description, gradient }: StatsCardProps) {
  return (
    <div className={`relative overflow-hidden rounded-2xl p-8 text-white ${gradient}`}>
      <div className="relative z-10">
        <h3 className="text-lg font-semibold mb-2">{title}</h3>
        <p className="text-4xl font-bold mb-3">{value}</p>
        <p className="text-sm opacity-90">{description}</p>
      </div>
      <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16" />
    </div>
  );
}

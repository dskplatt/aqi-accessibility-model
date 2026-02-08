'use client';

interface FeatureCardProps {
  icon: string;
  title: string;
  description: string;
}

export default function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
    <div className="group relative bg-gray-800 border border-gray-700 rounded-xl p-6 hover:shadow-xl hover:shadow-accent-start/20 hover:border-accent-start transition-all duration-300">
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-xl font-semibold mb-2 text-white">{title}</h3>
      <p className="text-gray-400">{description}</p>
      <div className="absolute inset-0 bg-gradient-to-br from-accent-start/10 to-accent-end/10 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl" />
    </div>
  );
}

'use client';

import Header from '@/components/Header';
import ScrollSection from '@/components/ScrollSection';

const visuals = [
    {
        title: 'Geographic Heatmap',
        description: 'Detailed analysis of AQI predictions mapped over demographic census tracts across the United States. Darker regions indicate higher correlation between low income and poor air quality.',
        type: 'Map Implementation',
        id: 'heatmap'
    },
    {
        title: 'Demographic Correlations',
        description: 'A multi-variable scatter plot showing the relationship between median household income, racial diversity indices, and local AQI measurements.',
        type: 'Bivariate Analysis',
        id: 'correlations'
    },
    {
        title: 'Temporal Trends',
        description: 'Tracking the shift in air quality metrics over a 10-year period, correlated with urbanization patterns and industrial shifts detected in census data.',
        type: 'Time Series',
        id: 'trends'
    },
    {
        title: 'Feature Importance Matrix',
        description: 'A breakdown of which census variables (housing age, education level, transport methods) have the highest predictive power for our ML models.',
        type: 'Model Insights',
        id: 'importance'
    }
];

export default function VisualsPage() {
    return (
        <div className="min-h-screen bg-gray-950 text-white">
            <Header />

            <main className="pt-32 pb-20 px-6">
                <div className="max-w-7xl mx-auto">
                    <ScrollSection>
                        <div className="mb-16">
                            <h1 className="text-5xl md:text-7xl font-bold mb-6">
                                Data{' '}
                                <span className="bg-gradient-to-r from-accent-start to-accent-end bg-clip-text text-transparent">
                                    Visualizations
                                </span>
                            </h1>
                            <p className="text-xl text-gray-400 max-w-2xl">
                                Explore the deep connections between community demographics and environmental health through our interactive insights and machine learning outputs.
                            </p>
                        </div>
                    </ScrollSection>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {visuals.map((visual, index) => (
                            <ScrollSection key={visual.id} delay={index * 100}>
                                <div className="group bg-gray-900 border border-gray-800 rounded-3xl overflow-hidden hover:border-accent-start/50 transition-all duration-300">
                                    {/* Placeholder for actual visual/image */}
                                    <div className="aspect-video bg-gray-800 relative flex items-center justify-center overflow-hidden">
                                        <div className="absolute inset-0 bg-gradient-to-br from-accent-start/5 to-accent-end/10 group-hover:opacity-100 transition-opacity" />
                                        <div className="text-accent-start text-5xl opacity-20 group-hover:scale-110 transition-transform duration-500">
                                            {visual.id === 'heatmap' && '🗺️'}
                                            {visual.id === 'correlations' && '📊'}
                                            {visual.id === 'trends' && '📈'}
                                            {visual.id === 'importance' && '🔍'}
                                        </div>
                                    </div>

                                    <div className="p-8">
                                        <div className="flex items-center gap-2 mb-4">
                                            <span className="px-3 py-1 bg-accent-start/10 text-accent-start text-xs font-bold uppercase tracking-wider rounded-full border border-accent-start/20">
                                                {visual.type}
                                            </span>
                                        </div>
                                        <h3 className="text-2xl font-bold mb-4 group-hover:text-accent-start transition-colors">
                                            {visual.title}
                                        </h3>
                                        <p className="text-gray-400 leading-relaxed">
                                            {visual.description}
                                        </p>
                                    </div>
                                </div>
                            </ScrollSection>
                        ))}
                    </div>
                </div>
            </main>

            <footer className="py-12 px-6 border-t border-gray-800 text-center text-gray-500">
                <p>© 2026 AQI Predictor. Powered by Census Data & Environmental Insights.</p>
            </footer>
        </div>
    );
}

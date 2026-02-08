'use client';

import Header from '@/components/Header';
import ScrollSection from '@/components/ScrollSection';
import AQIChoropleth from '@/components/AQIChoropleth';
import Image from 'next/image';
import { useState, useEffect } from 'react';

const visuals = [
    {
        title: 'Counties by Median AQI',
        description: 'This map contains the median AQI for each county in the United States. The color of the county is determined by the median AQI of the county. The darker the color, the higher the median AQI of the county. Not all counties contained data.',
        type: 'Geographic Heatmap',
        id: 'heatmap',
        component: <AQIChoropleth />
    },
    {
        title: 'Predictions vs Actual',
        description: 'A comparison of our XGBoost model\'s predicted AQI values against the actual recorded data, demonstrating the model\'s accuracy and reliability.',
        type: 'Model Performance',
        id: 'correlations',
        component: (
            <div className="relative w-full h-full p-4 flex items-center justify-center bg-white">
                <img
                    src="/predictions_vs_actual_dark_v3.png"
                    alt="Predictions vs Actual"
                    className="max-w-full max-h-full object-contain"
                />
            </div>
        )
    },
    {
        title: 'AQI by Race',
        description: 'The average median AQI for each county with a majority of a specific race race was calculated. This graph shows the average AQI for each race that has a majority of the population in that county.',
        type: 'Demographic Analysis',
        id: 'race-aqi',
        component: (
            <div className="relative w-full h-full p-4 flex items-center justify-center bg-white">
                <img
                    src="/aqi_by_race_dark_v4.png"
                    alt="AQI by Race"
                    className="max-w-full max-h-full object-contain"
                />
            </div>
        )
    },
    {
        title: 'Top 5 Predictive Features For Median AQI',
        description: 'The five features shown had the highest predictive power in our XGBoost Model. Each of these factors had the highest influence on the predictive model, and the power was modeled by their importance score.',
        type: 'Model Insights',
        id: 'importance',
        component: (
            <div className="relative w-full h-full p-4 flex items-center justify-center bg-white">
                <img
                    src="/feature_importance_dark_v2.png"
                    alt="Top 5 Features"
                    className="max-w-full max-h-full object-contain"
                />
            </div>
        )
    }
];

export default function VisualsPage() {
    const [expandedId, setExpandedId] = useState<string | null>(null);

    // Prevent scrolling when a visual is expanded
    useEffect(() => {
        if (expandedId) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
        }
    }, [expandedId]);

    return (
        <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 text-slate-900">
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
                            <p className="text-xl text-gray-600 max-w-2xl">
                                Explore the deep connections between community demographics and environmental health through our interactive insights and machine learning outputs.
                            </p>
                        </div>
                    </ScrollSection>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
                        {visuals.map((visual, index) => (
                            <ScrollSection key={visual.id} delay={index * 100} className="h-full">
                                <div
                                    onClick={() => setExpandedId(visual.id)}
                                    className="group bg-white border border-gray-200 rounded-3xl overflow-hidden hover:border-accent-start/50 hover:shadow-lg transition-all duration-300 h-full flex flex-col cursor-pointer hover:scale-[1.02]"
                                >
                                    {/* Visual/Image Container */}
                                    <div className="aspect-video bg-gray-50 relative flex items-center justify-center overflow-hidden">
                                        <div className="absolute inset-0 bg-gradient-to-br from-accent-start/5 to-accent-end/10 group-hover:opacity-100 transition-opacity pointer-events-none" />
                                        
                                        {visual.component ? (
                                            <div className="w-full h-full pointer-events-none">
                                                {visual.component}
                                            </div>
                                        ) : (
                                            <div className="text-accent-start text-5xl opacity-20 group-hover:scale-110 transition-transform duration-500">
                                                {visual.id === 'heatmap' && '🗺️'}
                                                {visual.id === 'correlations' && '📊'}
                                                {visual.id === 'race-aqi' && '📈'}
                                                {visual.id === 'importance' && '🔍'}
                                            </div>
                                        )}
                                    </div>

                                    <div className="p-8 flex flex-col flex-grow">
                                        <div className="flex items-center gap-2 mb-4">
                                            <span className="px-3 py-1 bg-accent-start/10 text-accent-start text-xs font-bold uppercase tracking-wider rounded-full border border-accent-start/20">
                                                {visual.type}
                                            </span>
                                        </div>
                                        <h3 className="text-2xl font-bold mb-4 group-hover:text-accent-start transition-colors">
                                            {visual.title}
                                        </h3>
                                        <p className="text-gray-600 leading-relaxed">
                                            {visual.description}
                                        </p>
                                    </div>
                                </div>
                            </ScrollSection>
                        ))}
                    </div>
                </div>
            </main>

            {/* Expanded Modal */}
            {expandedId && (
                <div
                    className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-10 bg-black/70 backdrop-blur-sm animate-in fade-in duration-300"
                    onClick={() => setExpandedId(null)}
                >
                    <div
                        className="relative w-full max-w-6xl bg-white border border-gray-200 rounded-[2.5rem] overflow-hidden shadow-2xl flex flex-col h-[85vh]"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Close Button */}
                        <button
                            onClick={() => setExpandedId(null)}
                            className="absolute top-6 right-6 z-[110] w-12 h-12 flex items-center justify-center rounded-full bg-white/90 hover:bg-gray-100 text-slate-900 transition-colors border border-gray-300 shadow-md"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                        </button>

                        {/* Expanded Visual */}
                        <div className="flex-grow bg-gray-50 flex items-center justify-center overflow-hidden">
                            {visuals.find(v => v.id === expandedId)?.component}
                        </div>

                        {/* Info Section */}
                        <div className="p-8 md:p-10 bg-white border-t border-gray-200">
                            <div className="flex items-center gap-3 mb-4">
                                <span className="px-4 py-1.5 bg-accent-start/10 text-accent-start text-sm font-bold uppercase tracking-widest rounded-full border border-accent-start/20">
                                    {visuals.find(v => v.id === expandedId)?.type}
                                </span>
                            </div>
                            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-slate-900">
                                {visuals.find(v => v.id === expandedId)?.title}
                            </h2>
                            <p className="text-xl text-gray-600 leading-relaxed max-w-4xl">
                                {visuals.find(v => v.id === expandedId)?.description}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            <footer className="py-12 px-6 border-t border-gray-200 text-center text-gray-600">
                <p>© 2026 AQI Predictor. Powered by Census Data & Environmental Insights.</p>
            </footer>
        </div>
    );
}

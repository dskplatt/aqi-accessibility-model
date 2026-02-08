'use client';

import Header from '@/components/Header';
import ScrollSection from '@/components/ScrollSection';

export default function FindingsPage() {
    return (
        <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 text-slate-900 font-sans">
            <Header />

            <main className="pt-32 pb-20 px-6">
                <div className="max-w-4xl mx-auto">
                    <ScrollSection>
                        <div className="mb-16">
                            <h1 className="text-5xl md:text-7xl font-bold mb-6">
                                Key{' '}
                                <span className="bg-gradient-to-r from-accent-start to-accent-end bg-clip-text text-transparent">
                                    Findings
                                </span>
                            </h1>
                            <p className="text-xl text-gray-600 leading-relaxed">
                                Our analysis reveals significant correlations between socioeconomic factors and environmental quality. Below are the primary insights derived from our machine learning models and data exploration.
                            </p>
                        </div>
                    </ScrollSection>

                    <div className="space-y-20">
                        {/* Finding 1 */}
                        <ScrollSection delay={100}>
                            <div className="group">
                                <div className="flex items-center gap-4 mb-4">
                                    <span className="text-accent-start font-mono text-lg">01.</span>
                                    <div className="h-px flex-grow bg-gray-200 group-hover:bg-accent-start/30 transition-colors" />
                                </div>
                                <h2 className="text-3xl font-bold mb-6 group-hover:text-accent-start transition-colors">Socioeconomic Disparities</h2>
                                <p className="text-lg text-gray-600 leading-relaxed">
                                    Counties with lower median household incomes consistently show higher average AQI levels. Our model identified income per capita as one of the top predictors of air quality, suggesting a strong link between economic status and environmental exposure.
                                </p>
                            </div>
                        </ScrollSection>

                        {/* Finding 2 */}
                        <ScrollSection delay={200}>
                            <div className="group">
                                <div className="flex items-center gap-4 mb-4">
                                    <span className="text-accent-start font-mono text-lg">02.</span>
                                    <div className="h-px flex-grow bg-gray-200 group-hover:bg-accent-start/30 transition-colors" />
                                </div>
                                <h2 className="text-3xl font-bold mb-6 group-hover:text-accent-start transition-colors">Demographic Impact</h2>
                                <p className="text-lg text-gray-600 leading-relaxed">
                                    Our demographic analysis shows that counties where minority groups are predominant often face higher median AQI values. Specifically, population density within these communities was a highly influential feature in our XGBoost model.
                                </p>
                            </div>
                        </ScrollSection>

                        {/* Finding 3 */}
                        <ScrollSection delay={300}>
                            <div className="group">
                                <div className="flex items-center gap-4 mb-4">
                                    <span className="text-accent-start font-mono text-lg">03.</span>
                                    <div className="h-px flex-grow bg-gray-200 group-hover:bg-accent-start/30 transition-colors" />
                                </div>
                                <h2 className="text-3xl font-bold mb-6 group-hover:text-accent-start transition-colors">Urban Density Correlation</h2>
                                <p className="text-lg text-gray-600 leading-relaxed">
                                    The interaction between population density and industrial presence remains the primary driver of poor air quality. However, the data suggests that the burden of this density is not distributed equally across income brackets.
                                </p>
                            </div>
                        </ScrollSection>
                    </div>
                </div>
            </main>

            <footer className="py-12 px-6 border-t border-gray-200 text-center text-gray-600 mt-20">
                <p>© 2026 AQI Predictor. Powered by Census Data & Environmental Insights.</p>
            </footer>
        </div>
    );
}

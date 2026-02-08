import Header from '@/components/Header';
import ScrollSection from '@/components/ScrollSection';
import StatsCard from '@/components/StatsCard';
import FeatureCard from '@/components/FeatureCard';
import AnimatedBackground from '@/components/AnimatedBackground';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-950 to-gray-900">
      <Header />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6 overflow-hidden">
        <AnimatedBackground />
        <div className="relative z-10 max-w-7xl mx-auto text-center">
          <ScrollSection>
            <h1 className="text-6xl md:text-8xl font-bold text-white mb-6 leading-tight">
              Predicting AQI from
              <br />
              <span className="bg-gradient-to-r from-accent-start to-accent-end bg-clip-text text-transparent">
                Census Data
              </span>
            </h1>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto mb-8">
              Applying machine learning and demographic insights to forecast air quality
              across communities
            </p>
            <div className="flex gap-4 justify-center">
              <Link href="/visuals" className="px-8 py-3 bg-accent-end text-white rounded-full font-medium hover:bg-accent-start transition-colors">
                Explore Visuals
              </Link>
              <Link
                href="/data"
                className="px-8 py-3 bg-gray-800 text-white rounded-full font-medium border border-gray-700 hover:border-gray-600 hover:bg-gray-700 transition-colors"
              >
                View Data
              </Link>
            </div>
          </ScrollSection>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-6" id="data">
        <div className="max-w-7xl mx-auto">
          <ScrollSection delay={100}>
            <h2 className="text-4xl font-bold text-white mb-12 text-center">
              By The Numbers
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <StatsCard
                title="Error Margin"
                value="<5.5 AQI"
                description="Median absolute error across census features"
                gradient="bg-gradient-to-br from-accent-start to-accent-end"
              />
              <div className="relative overflow-hidden rounded-2xl p-8 text-white bg-gradient-to-br from-accent-end to-accent-start">
                <div className="relative z-10">
                  <h3 className="text-lg font-semibold mb-3">Model Accuracy</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1 text-sm">
                        <span className="opacity-90">Within 10 AQI</span>
                        <span className="font-bold">85.71%</span>
                      </div>
                      <div className="w-full bg-white/20 rounded-full h-1.5 overflow-hidden">
                        <div className="bg-white rounded-full h-1.5" style={{ width: '85.71%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1 text-sm">
                        <span className="opacity-90">Within 12 AQI</span>
                        <span className="font-bold">91.01%</span>
                      </div>
                      <div className="w-full bg-white/20 rounded-full h-1.5 overflow-hidden">
                        <div className="bg-white rounded-full h-1.5" style={{ width: '91.01%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16" />
              </div>
              <StatsCard
                title="Correlation (R²)"
                value="0.534"
                description="Strong correlation for complex census data"
                gradient="bg-gradient-to-br from-accent-start/80 to-accent-end/80"
              />
            </div>
          </ScrollSection>
        </div>
      </section>


      {/* Footer */}
      <footer className="py-12 px-6 border-t border-gray-800">
        <div className="max-w-7xl mx-auto text-center text-gray-400">
          <p>Built with Next.js, React, and Tailwind CSS</p>
        </div>
      </footer>
    </div>
  );
}

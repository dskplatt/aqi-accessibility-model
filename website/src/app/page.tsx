import Header from '@/components/Header';
import ScrollSection from '@/components/ScrollSection';
import StatsCard from '@/components/StatsCard';
import FeatureCard from '@/components/FeatureCard';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-950 to-gray-900">
      <Header />

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <ScrollSection>
            <h1 className="text-6xl md:text-8xl font-bold text-white mb-6 leading-tight">
              Predicting AQI from
              <br />
              <span className="bg-gradient-to-r from-accent-start to-accent-end bg-clip-text text-transparent">
                Census Data
              </span>
            </h1>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto mb-8">
              Leveraging machine learning and demographic insights to forecast air quality
              across communities
            </p>
            <div className="flex gap-4 justify-center">
              <button className="px-8 py-3 bg-accent-end text-white rounded-full font-medium hover:bg-accent-start transition-colors">
                Explore Models
              </button>
              <button className="px-8 py-3 bg-gray-800 text-white rounded-full font-medium border border-gray-700 hover:border-gray-600 hover:bg-gray-700 transition-colors">
                View Data
              </button>
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
                title="Census Variables"
                value="120+"
                description="Demographic features analyzed"
                gradient="bg-gradient-to-br from-accent-start to-accent-end"
              />
              <StatsCard
                title="Model Accuracy"
                value="94.2%"
                description="Prediction accuracy achieved"
                gradient="bg-gradient-to-br from-accent-end to-accent-start"
              />
              <StatsCard
                title="Data Points"
                value="500K+"
                description="Historical AQI measurements"
                gradient="bg-gradient-to-br from-accent-start/80 to-accent-end/80"
              />
            </div>
          </ScrollSection>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6 bg-gray-900/50" id="visuals">
        <div className="max-w-7xl mx-auto">
          <ScrollSection delay={200}>
            <h2 className="text-4xl font-bold text-white mb-4 text-center">
              Interactive Visualizations
            </h2>
            <p className="text-xl text-gray-300 text-center mb-12 max-w-2xl mx-auto">
              Explore the relationship between demographic factors and air quality
            </p>
          </ScrollSection>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <ScrollSection delay={300}>
              <FeatureCard
                icon="📊"
                title="Geographic Heatmaps"
                description="Visualize AQI predictions across different regions with interactive maps"
              />
            </ScrollSection>
            <ScrollSection delay={400}>
              <FeatureCard
                icon="📈"
                title="Trend Analysis"
                description="Track air quality changes over time and correlate with census shifts"
              />
            </ScrollSection>
            <ScrollSection delay={500}>
              <FeatureCard
                icon="🔍"
                title="Feature Importance"
                description="Discover which demographic factors most influence air quality"
              />
            </ScrollSection>
          </div>
        </div>
      </section>

      {/* Model Section */}
      <section className="py-20 px-6" id="model">
        <div className="max-w-7xl mx-auto">
          <ScrollSection delay={200}>
            <div className="bg-gradient-to-r from-accent-start to-accent-end rounded-3xl p-12 text-white">
              <div className="max-w-3xl">
                <h2 className="text-4xl font-bold mb-6">
                  Advanced Machine Learning
                </h2>
                <p className="text-lg mb-8 opacity-90">
                  Our model combines gradient boosting algorithms with deep learning
                  techniques to analyze complex relationships between census data
                  and air quality metrics. We process demographic, economic, and
                  housing characteristics to predict AQI values with high accuracy.
                </p>
                <div className="flex gap-4">
                  <span className="px-4 py-2 bg-white/20 rounded-full text-sm">
                    XGBoost
                  </span>
                  <span className="px-4 py-2 bg-white/20 rounded-full text-sm">
                    Random Forest
                  </span>
                  <span className="px-4 py-2 bg-white/20 rounded-full text-sm">
                    Neural Networks
                  </span>
                </div>
              </div>
            </div>
          </ScrollSection>
        </div>
      </section>

      {/* Impact Section */}
      <section className="py-20 px-6 bg-gray-900/50" id="impact">
        <div className="max-w-7xl mx-auto">
          <ScrollSection delay={100}>
            <h2 className="text-4xl font-bold text-white mb-12 text-center">
              Real-World Impact
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
                <h3 className="text-2xl font-semibold mb-4 text-white">
                  Environmental Justice
                </h3>
                <p className="text-gray-300">
                  Identify communities at risk of poor air quality based on
                  demographic patterns, enabling targeted interventions and policy decisions.
                </p>
              </div>
              <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
                <h3 className="text-2xl font-semibold mb-4 text-white">
                  Urban Planning
                </h3>
                <p className="text-gray-300">
                  Support city planners in making data-driven decisions about development
                  and infrastructure to improve air quality for all residents.
                </p>
              </div>
            </div>
          </ScrollSection>
        </div>
      </section>

      {/* About Section */}
      <section className="py-20 px-6" id="about">
        <div className="max-w-4xl mx-auto text-center">
          <ScrollSection delay={200}>
            <h2 className="text-4xl font-bold text-white mb-6">
              About This Project
            </h2>
            <p className="text-lg text-gray-300 leading-relaxed">
              This project explores the intersection of demographic data and environmental
              health outcomes. By analyzing U.S. Census Bureau data alongside EPA air quality
              measurements, we aim to understand and predict how community characteristics
              relate to air pollution levels, ultimately contributing to healthier, more
              equitable communities.
            </p>
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

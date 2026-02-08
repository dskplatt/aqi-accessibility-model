'use client';

export default function AnimatedBackground() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Animated gradient overlay */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute inset-0 bg-gradient-to-br from-accent-start/40 via-accent-end/30 to-transparent animate-gradient-shift" />
      </div>

      {/* Floating orbs */}
      <div className="absolute -top-10 -left-10 w-96 h-96 bg-accent-start/20 rounded-full blur-3xl animate-float-slow" />
      <div className="absolute top-0 -right-10 w-80 h-80 bg-accent-end/20 rounded-full blur-3xl animate-float-delayed" />
      <div className="absolute top-1/4 -left-20 w-72 h-72 bg-accent-start/15 rounded-full blur-3xl animate-float-medium" />

      {/* Smaller accent orbs */}
      <div className="absolute top-[15%] right-[5%] w-48 h-48 bg-accent-end/25 rounded-full blur-2xl animate-pulse-slow" />
      <div className="absolute top-[40%] -right-10 w-56 h-56 bg-accent-start/20 rounded-full blur-2xl animate-float-reverse" />

      {/* Grid overlay with animation */}
      <div className="absolute inset-0 opacity-10">
        <div
          className="w-full h-full animate-grid-flow"
          style={{
            backgroundImage: `
              linear-gradient(to right, rgba(5, 150, 105, 0.1) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(2, 132, 199, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '80px 80px'
          }}
        />
      </div>

      {/* Particle dots */}
      <div className="absolute top-20 left-20 w-2 h-2 bg-accent-start rounded-full animate-twinkle" />
      <div className="absolute top-40 right-32 w-2 h-2 bg-accent-end rounded-full animate-twinkle-delayed" />
      <div className="absolute bottom-32 left-40 w-2 h-2 bg-accent-start rounded-full animate-twinkle-slow" />
      <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-accent-end rounded-full animate-twinkle" />
      <div className="absolute bottom-20 right-20 w-2 h-2 bg-accent-start rounded-full animate-twinkle-delayed" />
    </div>
  );
}

'use client';

import { motion, useReducedMotion } from 'framer-motion';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.05,
    },
  },
};

const planetVariants = {
  hidden: {
    y: -150,
    opacity: 0,
    scale: 0.8,
  },
  visible: {
    y: 0,
    opacity: 0.7,
    scale: 1,
    transition: {
      type: 'spring' as const,
      stiffness: 120,
      damping: 14,
    },
  },
};

const planetVariantsReduced = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 0.7,
    transition: { duration: 0.4 },
  },
};

const PLANETS = [
  { size: 'w-12 h-12', pos: 'top-[18%] left-[10%]', gradient: 'from-accent-start/60 to-accent-end/40' },
  { size: 'w-16 h-16', pos: 'top-[25%] right-[15%]', gradient: 'from-accent-end/50 to-accent-start/50' },
  { size: 'w-14 h-14', pos: 'top-[40%] left-[18%]', gradient: 'from-accent-start/70 to-accent-end/30' },
  { size: 'w-20 h-20', pos: 'top-[50%] right-[8%]', gradient: 'from-accent-end/60 to-accent-start/40' },
  { size: 'w-12 h-12', pos: 'top-[30%] left-[72%]', gradient: 'from-accent-start/50 to-accent-end/50' },
  { size: 'w-16 h-16', pos: 'top-[55%] left-[78%]', gradient: 'from-accent-end/70 to-accent-start/30' },
];

export default function PlanetDropAnimation() {
  const prefersReducedMotion = useReducedMotion();
  const variants = prefersReducedMotion ? planetVariantsReduced : planetVariants;

  return (
    <motion.div
      className="absolute inset-0 pointer-events-none z-[1]"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {PLANETS.map((planet, i) => (
        <motion.div
          key={i}
          className={`absolute ${planet.pos} ${planet.size} rounded-full bg-gradient-to-br ${planet.gradient} shadow-lg shadow-accent-start/10`}
          variants={variants}
        />
      ))}
    </motion.div>
  );
}

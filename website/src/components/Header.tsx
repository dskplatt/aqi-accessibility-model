'use client';

import Link from 'next/link';
import { useState } from 'react';

export default function Header() {
  const [activeNav, setActiveNav] = useState('');

  const navItems = [
    { name: 'Data', href: '/data' },
    { name: 'Process', href: '/process' },
    { name: 'Visuals', href: '/visuals' },
    { name: 'Findings', href: '/findings' },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
      <nav className="max-w-[1440px] mx-auto px-8 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="text-2xl font-montserrat tracking-wider bg-gradient-to-r from-accent-start to-accent-end bg-clip-text text-transparent font-bold hover:opacity-90 transition-opacity">
            Platypus
          </Link>

          <div className="flex gap-8">
            {navItems.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                onClick={() => setActiveNav(item.name)}
                className={`text-sm font-medium transition-colors hover:text-accent-start ${activeNav === item.name ? 'text-accent-start' : 'text-slate-600'
                  }`}
              >
                {item.name}
              </Link>
            ))}
          </div>
        </div>
      </nav>
    </header>
  );
}

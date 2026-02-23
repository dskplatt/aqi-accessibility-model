import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    // Use this app directory as root so Turbopack finds node_modules/next here
    // (avoids "Next.js package not found" when a lockfile exists at repo root)
    root: __dirname,
  },
};

export default nextConfig;

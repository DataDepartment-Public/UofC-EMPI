import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output for the production Docker image (Dockerfile) — traces
  // and bundles only the node_modules the server actually needs instead of
  // shipping the whole tree.
  output: "standalone",
};

export default nextConfig;

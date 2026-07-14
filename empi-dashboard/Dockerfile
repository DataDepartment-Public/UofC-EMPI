# syntax=docker/dockerfile:1
#
# Production image for the eMPI dashboard (Next.js App Router + BFF). Three
# stages: deps (npm ci), builder (next build --standalone), runner (just the
# traced server.js + static assets on a minimal base, non-root).
#
# The BFF proxies to empi-service over EMPI_API_URL (src/lib/server-api.ts) —
# set it at `docker run`/compose time to the backend's reachable address
# (e.g. http://empi-service:8000 on a compose network); it is never baked
# into the image.

# ---- deps ----
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- builder ----
FROM node:22-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ---- runner ----
FROM node:22-slim AS runner
WORKDIR /app

RUN groupadd --system empi \
    && useradd --system --gid empi --create-home --home-dir /home/empi empi

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# `output: "standalone"` (next.config.ts) traces only the node_modules the
# server actually needs — .next/static isn't included by standalone and must
# be copied in separately. No public/ directory exists in this project today
# (nothing to copy); add `COPY --from=builder /app/public ./public` if one is
# added later.
COPY --from=builder --chown=empi:empi /app/.next/standalone ./
COPY --from=builder --chown=empi:empi /app/.next/static ./.next/static

USER empi

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "fetch('http://localhost:3000/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]

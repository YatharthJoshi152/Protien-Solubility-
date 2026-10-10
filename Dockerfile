FROM node:22-bookworm-slim AS frontend
WORKDIR /app
COPY package.json pnpm-workspace.yaml tsconfig.json vite.config.ts index.html ./
COPY src ./src
COPY public ./public
RUN corepack enable && pnpm install --no-frozen-lockfile && pnpm build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend ./backend
COPY artifacts ./artifacts
COPY data ./data
COPY --from=frontend /app/dist ./dist
EXPOSE 3000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-3000}"]

FROM oven/bun:1.1-alpine

WORKDIR /app

COPY package.json .
RUN bun install --production

COPY bot.js .

EXPOSE 3000

CMD ["bun", "run", "bot.js"]

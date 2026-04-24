# 🦕 Dino — Chase FAQ Telegram Bot

A Claude-powered Telegram chatbot scoped strictly to Chase Bank FAQ topics.

---

## What Dino can answer

| Topic | Examples |
|---|---|
| Account types | Checking, savings, CDs, money market, student, business |
| Mortgage products | Purchase loans, refinancing, HELOC, jumbo, FHA/VA |
| Branch & ATM hours | General hours, holiday closures, ATM availability |
| Fee structures | Monthly fees, overdraft, ATM, wire transfer fees |
| Support numbers | Toll-free Chase customer service contacts |

Dino will firmly but warmly decline everything else — transactional requests, bank comparisons, news, entertainment, weather, legal advice, and more.

---

## Prerequisites

- [Bun](https://bun.sh) v1.0+ (runtime)
- A Telegram Bot Token from [@BotFather](https://t.me/BotFather)
- An Anthropic API key from [console.anthropic.com](https://console.anthropic.com)
- A public HTTPS URL (for the webhook)

---

## Local setup

```bash
# 1. Clone / copy the project
cd dino-bot

# 2. Install dependencies
bun install

# 3. Configure environment
cp .env.example .env
# Edit .env with your tokens

# 4. Start the bot
bun run start
```

---

## Getting a Telegram Bot Token

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`
3. Follow the prompts — choose a name (e.g. `Dino Chase Helper`) and a username (e.g. `DinoChaseBot`)
4. Copy the token BotFather gives you into `TELEGRAM_BOT_TOKEN` in your `.env`

---

## Webhook setup

Telegram requires a **public HTTPS URL** to deliver messages. Options:

### Option A — ngrok (local dev / testing)
```bash
# Install ngrok: https://ngrok.com
ngrok http 3000

# Copy the https://xxxx.ngrok-free.app URL into WEBHOOK_URL in .env
# Restart the bot — it will auto-register the webhook
```

### Option B — Railway (recommended for production)
1. Push this folder to a GitHub repo
2. Create a new project at [railway.app](https://railway.app)
3. Connect your repo
4. Add environment variables in Railway's dashboard
5. Railway auto-assigns a public HTTPS URL — paste it into `WEBHOOK_URL`

### Option C — Render
1. Push to GitHub
2. New Web Service at [render.com](https://render.com)
3. Set build command: `bun install`
4. Set start command: `bun run bot.js`
5. Add env vars, deploy, copy the `.onrender.com` URL into `WEBHOOK_URL`

### Option D — Docker (any VPS)
```bash
docker build -t dino-bot .
docker run -d \
  -p 3000:3000 \
  -e TELEGRAM_BOT_TOKEN=... \
  -e ANTHROPIC_API_KEY=... \
  -e WEBHOOK_URL=https://yourdomain.com \
  --name dino-bot \
  dino-bot
```
Put Nginx or Caddy in front for HTTPS termination.

---

## Bot commands

| Command | Action |
|---|---|
| `/start` | Welcome message + topic overview |
| `/help` | Show what Dino can and can't help with |
| `/reset` | Clear conversation history for this chat |

---

## Architecture

```
Telegram user
     │  (HTTPS POST)
     ▼
 bot.js (Bun HTTP server)
     │  /webhook endpoint
     ▼
 handleUpdate()
     │
     ├─ /start, /help, /reset  →  static responses
     │
     └─ any text  →  askClaude()
                          │
                          ▼
                   Anthropic API
                   (claude-sonnet-4)
                   with system prompt
                   + per-user history
                          │
                          ▼
                   sendMessage()
                   back to Telegram
```

- Conversation history is stored in-memory per `chatId` (up to 20 turns)
- For production multi-instance deployments, replace the `conversations` Map with Redis

---

## Files

```
dino-bot/
├── bot.js          # Main server + bot logic
├── package.json
├── Dockerfile
├── .env.example    # Copy to .env and fill in
└── README.md
```

---

## Customization tips

- **Tone**: Edit the `SYSTEM_PROMPT` in `bot.js` to adjust Dino's personality
- **History length**: Change `MAX_HISTORY` (default 20) to control memory
- **Persistent storage**: Replace the `conversations` Map with a Redis or SQLite adapter
- **Rate limiting**: Add per-user request throttling before the `askClaude()` call for production

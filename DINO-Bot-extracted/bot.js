import Anthropic from "@anthropic-ai/sdk";
import { appendFileSync, existsSync, writeFileSync } from "fs";
import "dotenv/config";

const TELEGRAM_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const PORT = process.env.PORT || 3000;
const WEBHOOK_URL = process.env.WEBHOOK_URL; // e.g. https://yourdomain.com
const CALIBRAI_URL = process.env.CALIBRAI_URL || "http://localhost:8000";
const SAFETY_LEVEL = parseInt(process.env.SAFETY_LEVEL || "3", 10);
const INDUSTRY = process.env.INDUSTRY || "Banking";

const FALLBACK_MSG =
  "🦕 Rawr! I can't help with that one. Try asking about Chase accounts, mortgages, branch hours, fees, or support numbers!";

const anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

// In-memory conversation store: chatId -> messages[]
// For production, swap with Redis or a DB
const conversations = new Map();
const MAX_HISTORY = 20; // keep last 20 turns per user

const SYSTEM_PROMPT = `You are Dino — a cheerful, friendly baby boy dinosaur chatbot on the Chase Bank website. You speak in a warm, helpful, slightly playful tone (you're a baby dino after all!), but always stay professional and accurate.

You are ONLY allowed to answer questions in these exact topic areas:
1. Chase account types (checking, savings, CDs, money market, student accounts, business accounts)
2. Chase mortgage products (home purchase loans, refinancing, HELOCs, jumbo loans, FHA/VA loans)
3. Chase branch and ATM hours and general location info
4. Chase fee structures (monthly maintenance fees, overdraft fees, ATM fees, wire transfer fees, etc.)
5. Chase toll-free customer support contact numbers

HARD STOPS — you must NEVER help with the following, no exceptions:
- Any transactional action: checking balances, making transfers, disputing charges, paying bills, moving money — these require authentication and real backend access
- Comparing Chase to other banks or financial institutions
- Media, entertainment, movies, music, TV, sports
- Therapy, emotional support, empathy conversations, mental health
- Weather
- Legal advice
- Job interview coaching or career advice
- Holiday or travel planning
- Romance, friendship, or relationship advice
- AI, chatbots, or how you work
- News — about Chase, banking industry, or anything else
- Chase leadership, executives, or organizational structure
- Hotels, restaurants, reviews, recommendations for anything non-Chase
- Anything not directly related to the 5 allowed topics above

When someone asks about something outside your scope, respond warmly but firmly as Dino: explain you can only help with Chase account types, mortgage products, branch/ATM hours, fees, and the support phone number. Do NOT try to help even a little with off-topic questions. Do NOT apologize excessively — one brief friendly note is enough, then redirect.

Keep responses concise (2–5 sentences max unless listing items). Use a friendly, slightly warm tone. You can use occasional light dinosaur personality (e.g., "Rawr! Great question!") but don't overdo it. Always be accurate — only state facts you're confident about regarding Chase products and services.

You are operating in Telegram, so format your replies using Telegram-compatible Markdown:
- Use *bold* for emphasis (not **)
- Use plain text for most content
- Keep bullet lists clean with a dash or emoji prefix
- No HTML tags`;

// ─── Telegram API helpers ────────────────────────────────────────────────────

async function telegramRequest(method, body) {
  const res = await fetch(
    `https://api.telegram.org/bot${TELEGRAM_TOKEN}/${method}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
  return res.json();
}

async function sendMessage(chatId, text) {
  return telegramRequest("sendMessage", {
    chat_id: chatId,
    text,
    parse_mode: "Markdown",
  });
}

async function sendTyping(chatId) {
  return telegramRequest("sendChatAction", {
    chat_id: chatId,
    action: "typing",
  });
}

// ─── CalibrAI evaluation ─────────────────────────────────────────────────────

async function evaluateWithCalibrAI(query, response, safetyLevel, industry) {
  const res = await fetch(`${CALIBRAI_URL}/api/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      response,
      safety_level: safetyLevel,
      industry,
    }),
  });
  if (!res.ok) throw new Error(`CalibrAI HTTP ${res.status}`);
  return res.json(); // { blocked: bool, reason: string }
}

// ─── Interaction logging ─────────────────────────────────────────────────────

const CSV_PATH = "dino_interactions.csv";
const CSV_HEADER = "timestamp,threshold_level,query,response,blocked,latency_ms\n";

function csvEscape(s) {
  return `"${String(s).replace(/"/g, '""')}"`;
}

function logInteraction(timestamp, safetyLevel, query, response, blocked, latencyMs) {
  if (!existsSync(CSV_PATH)) writeFileSync(CSV_PATH, CSV_HEADER);
  const row = [
    csvEscape(timestamp),
    safetyLevel,
    csvEscape(query),
    csvEscape(response),
    blocked,
    latencyMs,
  ].join(",");
  appendFileSync(CSV_PATH, row + "\n");
}

// ─── Claude handler ──────────────────────────────────────────────────────────

async function askClaude(chatId, userText) {
  // Get or create history for this chat
  if (!conversations.has(chatId)) {
    conversations.set(chatId, []);
  }
  const history = conversations.get(chatId);

  history.push({ role: "user", content: userText });

  // Trim to last MAX_HISTORY messages to stay within context limits
  if (history.length > MAX_HISTORY) {
    history.splice(0, history.length - MAX_HISTORY);
  }

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    system: SYSTEM_PROMPT,
    messages: history,
  });

  const reply = response.content.find((b) => b.type === "text")?.text || "Rawr... something went wrong! Please try again in a moment.";

  history.push({ role: "assistant", content: reply });

  return reply;
}

// ─── Update router ───────────────────────────────────────────────────────────

async function handleUpdate(update) {
  const msg = update.message || update.edited_message;
  if (!msg || !msg.text) return;

  const chatId = msg.chat.id;
  const text = msg.text.trim();

  // /start command — greeting
  if (text === "/start") {
    await sendMessage(
      chatId,
      `🦕 *Rawr! Hi there, I'm Dino!*\n\nI'm your friendly Chase helper. Ask me anything about:\n\n🏦 Account types\n🏠 Mortgage products\n🕐 Branch & ATM hours\n💰 Fee structures\n📞 Support contact numbers\n\nWhat can I help you with today?`
    );
    return;
  }

  // /help command
  if (text === "/help") {
    await sendMessage(
      chatId,
      `🦕 *Dino's Help Menu*\n\nI can only answer questions about:\n\n- Chase account types (checking, savings, CDs, etc.)\n- Chase mortgage products\n- Branch & ATM hours\n- Fee structures\n- Toll-free support numbers\n\nFor anything transactional (balances, transfers, disputes), please log in to chase.com or call *1-800-935-9935*.`
    );
    return;
  }

  // /reset — clear conversation history
  if (text === "/reset") {
    conversations.delete(chatId);
    await sendMessage(chatId, "🦕 Rawr! Fresh start! What would you like to know about Chase?");
    return;
  }

  // /status — show current safety configuration
  if (text === "/status") {
    await sendMessage(
      chatId,
      `🦕 *Dino Status*\n\n🛡 Safety Level: *${SAFETY_LEVEL}* / 5\n🏢 Industry: *${INDUSTRY}*\n🔗 CalibrAI: ${CALIBRAI_URL}`
    );
    return;
  }

  // Show typing while we wait for Claude
  await sendTyping(chatId);
  const t0 = Date.now();

  try {
    const reply = await askClaude(chatId, text);

    // Evaluate Dino's response with CalibrAI before delivering it
    let blocked = false;
    try {
      const evaluation = await evaluateWithCalibrAI(text, reply, SAFETY_LEVEL, INDUSTRY);
      blocked = evaluation.blocked;
    } catch (err) {
      // Fail open — if CalibrAI is unreachable, don't block the user
      console.error("CalibrAI evaluation error (fail-open):", err);
    }

    const latencyMs = Date.now() - t0;
    // Log the original reply so blocked responses are visible in the CSV
    logInteraction(new Date().toISOString(), SAFETY_LEVEL, text, reply, blocked, latencyMs);

    await sendMessage(chatId, blocked ? FALLBACK_MSG : reply);
  } catch (err) {
    console.error("Claude error:", err);
    await sendMessage(chatId, "🦕 Rawr! I hit a little snag. Please try again in a moment!");
  }
}

// ─── HTTP server (webhook receiver) ─────────────────────────────────────────

async function startServer() {
  const server = Bun.serve({
    port: PORT,
    async fetch(req) {
      const url = new URL(req.url);

      // Health check
      if (url.pathname === "/health") {
        return new Response(JSON.stringify({ status: "ok", bot: "Dino" }), {
          headers: { "Content-Type": "application/json" },
        });
      }

      // Telegram webhook endpoint
      if (url.pathname === "/webhook" && req.method === "POST") {
        try {
          const update = await req.json();
          // Handle async without blocking the response
          handleUpdate(update).catch((e) => console.error("Update error:", e));
          return new Response("OK");
        } catch (e) {
          console.error("Webhook parse error:", e);
          return new Response("Bad Request", { status: 400 });
        }
      }

      return new Response("Not Found", { status: 404 });
    },
  });

  console.log(`🦕 Dino is running on port ${PORT}`);
  return server;
}

// ─── Webhook registration ────────────────────────────────────────────────────

async function registerWebhook() {
  const webhookEndpoint = `${WEBHOOK_URL}/webhook`;
  const result = await telegramRequest("setWebhook", {
    url: webhookEndpoint,
    allowed_updates: ["message", "edited_message"],
    drop_pending_updates: true,
  });

  if (result.ok) {
    console.log(`✅ Webhook registered: ${webhookEndpoint}`);
  } else {
    console.error("❌ Webhook registration failed:", result);
  }
}

// ─── Long polling (local dev — no webhook needed) ────────────────────────────

async function startPolling() {
  // Clear any existing webhook so Telegram delivers updates to getUpdates
  await telegramRequest("deleteWebhook", { drop_pending_updates: true });
  console.log(`🦕 Dino L${SAFETY_LEVEL} polling for updates (no webhook)…`);

  let offset = 0;
  while (true) {
    try {
      const data = await telegramRequest("getUpdates", {
        offset,
        timeout: 30,
        allowed_updates: ["message", "edited_message"],
      });

      if (data.ok && data.result.length > 0) {
        for (const update of data.result) {
          offset = update.update_id + 1;
          handleUpdate(update).catch((e) => console.error("Update error:", e));
        }
      }
    } catch (err) {
      console.error("Polling error:", err);
      // Brief back-off to avoid hammering Telegram on transient errors
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
}

// ─── Boot ────────────────────────────────────────────────────────────────────

if (WEBHOOK_URL) {
  await startServer();
  await registerWebhook();
} else {
  await startPolling();
}

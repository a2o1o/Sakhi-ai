import "dotenv/config";
import cors from "cors";
import express from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "csv-parse/sync";
import { GoogleGenAI } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const backendDir = path.dirname(__filename);
const repoRoot = path.resolve(backendDir, "../..");
const publicDir = path.join(backendDir, "public");

const app = express();
const port = Number(process.env.PORT || 3000);
const allowedOrigin = process.env.ALLOWED_ORIGIN || "*";
const sharedAccessToken = process.env.APP_ACCESS_TOKEN || "";
const model = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const maxOutputTokens = Number(process.env.MAX_OUTPUT_TOKENS || 1100);
const gemini = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const extraInstructions = process.env.CUSTOM_GPT_INSTRUCTIONS || "";

const baseSystemPrompt = [
  "You are Sakhi, a warm reflection companion for girls and young women in the Maitri community.",
  "Sound like a thoughtful peer, not a lecturer, therapist, or authority figure.",
  "Be calm, human, non-judgmental, and clear.",
  "Match the user's language naturally. If the user writes in English, reply in English. If the user writes in Hinglish, reply in Hinglish. If the user writes in Hindi, reply in Hindi.",
  "Reflect first, then offer useful next steps only if they help.",
  "For greetings or very short messages, reply lightly in 1 or 2 sentences.",
  "For meaningful concerns, aim for 4 to 8 sentences unless the user clearly wants more.",
  "If the user is overwhelmed, keep it simple and offer one small next step.",
  "If the user is choosing between paths, give 2 or 3 ways to think about it without deciding for them.",
  "Use anonymized peer grounding only when genuinely relevant and keep it subtle.",
  "Never use names or identifying details.",
  "Never mention datasets, internal sources, files, or hidden context.",
  "Never diagnose, prescribe, or overpromise.",
  "Do not just restate the problem. Move the conversation forward."
].join(" ");

const systemPrompt = extraInstructions
  ? `${baseSystemPrompt}\n\nAdditional instructions:\n${extraInstructions}`
  : baseSystemPrompt;

const ignoredColumns = [
  "timestamp",
  "email address",
  "name",
  "phone number",
  "column 17",
  "column 31",
  "column 32",
  "column 33",
  "do you consent to participating in this?",
  "do you consent to participating in this form under the above conditions?"
];

const stageFileMap = {
  school:
    process.env.SCHOOL_RESPONSES_CSV ||
    path.join(repoRoot, "data", "Maitri-school-responses.csv"),
  college:
    process.env.COLLEGE_RESPONSES_CSV ||
    path.join(repoRoot, "data", "Maitri-college-responses.csv"),
  "early work":
    process.env.WORKING_WOMEN_RESPONSES_CSV ||
    path.join(repoRoot, "data", "Maitri-working-women-responses.csv"),
  work:
    process.env.WORKING_WOMEN_RESPONSES_CSV ||
    path.join(repoRoot, "data", "Maitri-working-women-responses.csv")
};

function normalizeStage(stage) {
  return String(stage || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
}

function normalizeHeader(value) {
  return String(value || "")
    .replace(/\uFFFD/g, "")
    .replace(/[^\w\s]/g, " ")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\u0900-\u097F\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2);
}

function classifyMessage(message) {
  const trimmed = String(message || "").trim();
  const lowered = trimmed.toLowerCase();
  const greetingOnly = /^(hi+|hey+|hello|yo+|sup)\b[.!? ]*$/i.test(trimmed);
  const casualPatterns = [
    "how are you",
    "what's up",
    "whats up",
    "good morning",
    "good evening",
    "good afternoon"
  ];

  if (greetingOnly || casualPatterns.some((pattern) => lowered.includes(pattern))) {
    return "casual";
  }

  return "reflective";
}

function detectLanguage(message) {
  const text = String(message || "").trim();
  const lowered = text.toLowerCase();
  const hasDevanagari = /[\u0900-\u097F]/.test(text);
  const hinglishMarkers = [
    "mujhe",
    "mera",
    "meri",
    "kyu",
    "kyun",
    "nahi",
    "nahin",
    "hai",
    "hoon",
    "karna",
    "karni",
    "karu",
    "karun",
    "kya",
    "kaise",
    "acha",
    "accha",
    "samajh",
    "bata",
    "batao",
    "lag",
    "raha",
    "rahi",
    "chahiye"
  ];
  const latinWords = lowered.match(/[a-z]+/g) || [];
  const hinglishHits = hinglishMarkers.filter((word) => lowered.includes(word)).length;

  if (hasDevanagari && latinWords.length > 0) return "hinglish";
  if (hasDevanagari) return "hindi";
  if (hinglishHits >= 2) return "hinglish";
  return "english";
}

function getLanguageInstruction(language) {
  if (language === "hindi") return "Reply fully in Hindi.";
  if (language === "hinglish") {
    return "Reply naturally in Hinglish, using the same kind of Hindi-English mix as the user.";
  }
  return "Reply fully in English.";
}

function isHighRiskMessage(message) {
  const text = String(message || "").toLowerCase();
  const patterns = [
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "don't want to live",
    "self harm",
    "self-harm",
    "hurt myself",
    "harm myself",
    "cut myself",
    "marna chahti",
    "marna chahta",
    "jeena nahi",
    "jeena nahin",
    "khud ko nuksan",
    "khud ko maar",
    "i want to disappear"
  ];

  return patterns.some((pattern) => text.includes(pattern));
}

function getSafetyReply(language) {
  if (language === "hindi") {
    return "Mujhe lag raha hai ki aap abhi bahut zyada takleef mein ho. Agar aapko lag raha hai ki aap khud ko nuksan pahucha sakte ho, abhi kisi trusted insaan ko call kijiye ya unke paas jaiye. Agar immediate danger ho, apne local emergency number par abhi call kijiye. Aapko is waqt akela nahi rehna chahiye.";
  }

  if (language === "hinglish") {
    return "Mujhe lag raha hai ki tum abhi bahut zyada distress mein ho. Agar tumhe lag raha hai ki tum khud ko nuksan pahucha sakti ho, please abhi kisi trusted person ko call karo ya unke paas chale jao. Agar immediate danger hai, apne local emergency number par abhi call karo. Is waqt akeli mat raho.";
  }

  return "It sounds like you may be in immediate distress. If you might hurt yourself or are in immediate danger, call your local emergency number right now or go to a trusted person nearby immediately. Please do not stay alone with this right now.";
}

function getTemporaryFailureReply(language) {
  if (language === "hindi") {
    return "Sakhi abhi kaam nahi kar pa rahi hai. Kripya thodi der baad phir se try kijiye.";
  }

  if (language === "hinglish") {
    return "Sakhi abhi kaam nahi kar pa rahi hai. Please thodi der baad phir se try karo.";
  }

  return "Sakhi is not working right now. Please come back a little later and try again.";
}

function isRetryableModelError(error) {
  const message = String(error?.message || error || "").toLowerCase();
  return /429|503|rate|quota|too many|service unavailable|overloaded|unavailable/.test(message);
}

function looksIncompleteResponse(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return true;
  if (trimmed.length < 80) return false;
  if (/[.!?)"'\u0964]$/.test(trimmed)) return false;
  return /[a-z0-9\u0900-\u097F]$/i.test(trimmed);
}

function getCasualReply(message) {
  const lowered = String(message || "").trim().toLowerCase();

  if (/^(hi+|hey+|hello)\b/.test(lowered)) {
    return "Hey :) what's on your mind?";
  }

  if (lowered === "sup" || lowered.startsWith("yo") || lowered.startsWith("yoo")) {
    return "yo :) what's up?";
  }

  if (lowered.includes("how are you")) {
    return "I'm here :) what's been going on?";
  }

  return "You can take your time. What's been sitting with you lately?";
}

function resolveDataFile(filePath) {
  if (!filePath) {
    return "";
  }
  return path.isAbsolute(filePath) ? filePath : path.resolve(repoRoot, filePath);
}

function loadStageResponses(filePath, stageLabel) {
  const resolved = resolveDataFile(filePath);
  if (!resolved || !fs.existsSync(resolved)) {
    return [];
  }

  const csvText = fs.readFileSync(resolved, "utf8");
  const rows = parse(csvText, {
    columns: true,
    skip_empty_lines: true,
    relax_column_count: true,
    bom: true
  });

  const snippets = [];

  for (const row of rows) {
    for (const [header, rawValue] of Object.entries(row)) {
      const headerKey = normalizeHeader(header);
      if (!headerKey || ignoredColumns.includes(headerKey)) {
        continue;
      }

      const text = String(rawValue || "").replace(/\s+/g, " ").trim();
      if (text.length < 35) {
        continue;
      }

      if (/^(yes|no|maybe|n\/a|none|not applicable|na)$/i.test(text)) {
        continue;
      }

      snippets.push({
        stage: stageLabel,
        header: headerKey,
        text
      });
    }
  }

  return snippets;
}

const stageResponses = {
  school: loadStageResponses(stageFileMap.school, "School"),
  college: loadStageResponses(stageFileMap.college, "College"),
  "early work": loadStageResponses(stageFileMap["early work"], "Early Work")
};

const conversationStore = new Map();
const MAX_TURNS_PER_SESSION = 2;
const practicalTopics = new Set([
  "scholarships",
  "after 12th options",
  "internships",
  "courses & upskilling",
  "courses and upskilling"
]);
const analytics = {
  totalRequests: 0,
  successes: 0,
  failures: 0,
  rateLimitErrors: 0,
  totalResponseChars: 0
};

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function scoreSnippet(snippet, queryTokens) {
  const haystackTokens = tokenize(`${snippet.header} ${snippet.text}`);
  const tokenSet = new Set(haystackTokens);
  let score = 0;

  for (const token of queryTokens) {
    if (tokenSet.has(token)) {
      score += 3;
    }
  }

  const compact = snippet.text.toLowerCase();
  if (compact.includes("confused") || compact.includes("self doubt")) score += 1;
  if (compact.includes("career") || compact.includes("college")) score += 1;
  if (compact.includes("friends") || compact.includes("family")) score += 1;

  return score;
}

function shouldUsePeerContext(message, peerSnippets) {
  const lowered = String(message || "").trim().toLowerCase();
  const queryTokens = tokenize(message);

  if (!peerSnippets.length) {
    return false;
  }

  const simpleEmotionOnly =
    queryTokens.length <= 4 &&
    /(sad|lonely|empty|low|tired|down|upset|bad)/.test(lowered) &&
    !/(because|about|career|future|friends?|family|college|school|work|job|money|compare|pressure)/.test(
      lowered
    );

  if (simpleEmotionOnly) {
    return false;
  }

  return peerSnippets[0].score >= 6;
}

function getPeerContext(stage, message) {
  const normalizedStage = normalizeStage(stage);
  const pool =
    stageResponses[normalizedStage] ||
    stageResponses[
      normalizedStage.includes("work")
        ? "early work"
        : normalizedStage.includes("college")
          ? "college"
          : "school"
    ] ||
    [];

  const queryTokens = tokenize(message);
  const ranked = pool
    .map((snippet) => ({ ...snippet, score: scoreSnippet(snippet, queryTokens) }))
    .filter((snippet) => snippet.score > 0)
    .sort((a, b) => b.score - a.score || b.text.length - a.text.length);

  const selected = [];
  const seenTexts = new Set();

  for (const snippet of ranked) {
    if (selected.length === 2) break;
    if (seenTexts.has(snippet.text)) continue;
    seenTexts.add(snippet.text);
    selected.push(snippet);
  }

  return shouldUsePeerContext(message, selected) ? selected : [];
}

function getSessionKey({ req, source, sessionId, userId, stage, topic }) {
  if (typeof sessionId === "string" && sessionId.trim()) {
    return `session:${sessionId.trim().slice(0, 120)}`;
  }

  if (typeof userId === "string" && userId.trim()) {
    return `user:${userId.trim().slice(0, 120)}`;
  }

  const ip = String(req.ip || req.headers["x-forwarded-for"] || "anon")
    .split(",")[0]
    .trim();
  const normalizedStage = normalizeStage(stage || "unknown");
  const normalizedTopic = normalizeStage(topic || "general");
  return `fallback:${source || "app"}:${ip}:${normalizedStage}:${normalizedTopic}`;
}

function getConversationHistory(sessionKey) {
  return conversationStore.get(sessionKey) || [];
}

function storeConversationTurn(sessionKey, role, text) {
  const history = conversationStore.get(sessionKey) || [];
  history.push({ role, text: String(text || "").trim() });
  conversationStore.set(sessionKey, history.slice(-MAX_TURNS_PER_SESSION));
}

function isPracticalTopic(topic) {
  return practicalTopics.has(normalizeStage(topic));
}

function getEffectivePeerSnippets(topic, peerSnippets) {
  if (!peerSnippets.length) {
    return [];
  }

  return isPracticalTopic(topic) ? peerSnippets.slice(0, 1) : peerSnippets.slice(0, 2);
}

function buildReflectivePrompt({ message, stage, topic, peerSnippets, history, language }) {
  const parts = [
    "mode: reflective",
    `stage: ${stage || "unspecified"}`,
    `topic: ${topic || "general"}`,
    `language: ${language}`,
    "",
    "User message",
    message.trim()
  ];

  if (history.length) {
    parts.push("", "Recent context");
    for (const item of history) {
      parts.push(`${item.role}: ${item.text}`);
    }
  }

  if (peerSnippets.length) {
    parts.push("", "Relevant anonymized Maitri reflections");
    for (const [index, snippet] of peerSnippets.entries()) {
      parts.push(`${index + 1}. (${snippet.stage}) ${snippet.text}`);
    }
  }

  parts.push(
    "",
    "Reply requirements",
    `- ${getLanguageInstruction(language)}`,
    "- Acknowledge briefly, reflect the feeling, then offer something useful.",
    "- Keep it warm, human, and concise.",
    "- If helpful, add one subtle peer-grounded line.",
    "- Offer one small next step or one focused question, not both unless very short.",
    "- Finish cleanly. Do not end mid-sentence."
  );

  return parts.join("\n");
}

function buildPracticalPrompt({ message, stage, topic, peerSnippets, history, language }) {
  const parts = [
    "mode: practical",
    `stage: ${stage || "unspecified"}`,
    `topic: ${topic || "general"}`,
    `language: ${language}`,
    "",
    "User question",
    message.trim()
  ];

  if (history.length) {
    parts.push("", "Minimal context");
    for (const item of history) {
      parts.push(`${item.role}: ${item.text}`);
    }
  }

  if (peerSnippets.length) {
    parts.push("", "One relevant Maitri reflection");
    parts.push(`1. (${peerSnippets[0].stage}) ${peerSnippets[0].text}`);
  }

  parts.push(
    "",
    "Reply requirements",
    `- ${getLanguageInstruction(language)}`,
    "- Stay anchored to the selected topic.",
    "- Be practical, direct, and mobile-friendly.",
    "- Use a short intro, then 2 or 3 compact bullets or options if useful.",
    "- If the user sounds emotional, validate briefly before the practical guidance.",
    "- Prefer criteria, next steps, and shortlists over long explanations.",
    "- Finish cleanly. Do not end mid-sentence."
  );

  return parts.join("\n");
}

function buildGeminiPrompt({ message, stage, topic, peerSnippets, history, language }) {
  return isPracticalTopic(topic)
    ? buildPracticalPrompt({ message, stage, topic, peerSnippets, history, language })
    : buildReflectivePrompt({ message, stage, topic, peerSnippets, history, language });
}

function getResponseTokenLimit(topic) {
  return isPracticalTopic(topic) ? Math.min(maxOutputTokens, 850) : maxOutputTokens;
}

async function generateGeminiText({ prompt, tokenLimit }) {
  const response = await gemini.models.generateContent({
    model,
    contents: prompt,
    config: {
      systemInstruction: systemPrompt,
      maxOutputTokens: tokenLimit
    }
  });

  return response.text?.trim() || "No response text returned.";
}

async function repairIncompleteResponse({ text, language, topic }) {
  const repairPrompt = [
    `language: ${language}`,
    `topic: ${topic || "general"}`,
    "",
    "The following draft reply got cut off.",
    "Rewrite it as one complete reply in the same language and same tone.",
    "Keep the meaning, do not add major new ideas, and end cleanly.",
    "Keep it concise and mobile-friendly.",
    "",
    "Draft reply",
    text
  ].join("\n");

  return generateGeminiText({
    prompt: repairPrompt,
    tokenLimit: Math.min(getResponseTokenLimit(topic), 500)
  });
}

async function generateSakhiReply({ message, stage, topic, language, peerSnippets, history }) {
  const prompt = buildGeminiPrompt({
    message,
    stage,
    topic,
    language,
    peerSnippets,
    history
  });
  const tokenLimit = getResponseTokenLimit(topic);

  let lastError;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      let text = await generateGeminiText({ prompt, tokenLimit });

      if (looksIncompleteResponse(text)) {
        try {
          const repaired = await repairIncompleteResponse({ text, language, topic });
          if (repaired && !looksIncompleteResponse(repaired)) {
            text = repaired;
          }
        } catch (repairError) {
          console.log(
            JSON.stringify({
              type: "sakhi_repair",
              outcome: "failed",
              topic: topic || "general",
              error: String(repairError?.message || repairError).slice(0, 200)
            })
          );
        }
      }

      return text;
    } catch (error) {
      lastError = error;
      if (!isRetryableModelError(error) || attempt === 1) {
        throw error;
      }
      await wait(600 * (attempt + 1));
    }
  }

  throw lastError;
}

function recordSuccess(topic, responseText) {
  analytics.totalRequests += 1;
  analytics.successes += 1;
  analytics.totalResponseChars += String(responseText || "").length;
  console.log(
    JSON.stringify({
      type: "sakhi_analytics",
      outcome: "success",
      topic: topic || "general",
      averageResponseChars: analytics.successes
        ? Math.round(analytics.totalResponseChars / analytics.successes)
        : 0
    })
  );
}

function recordFailure(topic, error) {
  analytics.totalRequests += 1;
  analytics.failures += 1;
  const message = String(error?.message || error || "");
  if (message.includes("429") || /rate|quota|too many/i.test(message)) {
    analytics.rateLimitErrors += 1;
  }
  console.log(
    JSON.stringify({
      type: "sakhi_analytics",
      outcome: "failure",
      topic: topic || "general",
      rateLimitErrors: analytics.rateLimitErrors,
      error: message.slice(0, 300)
    })
  );
}

app.use(
  cors({
    origin: allowedOrigin === "*" ? true : allowedOrigin
  })
);
app.use(express.json({ limit: "1mb" }));
app.use(express.text({ type: "*/*", limit: "64kb" }));
app.use(express.static(publicDir));

function requireConfiguredApiKey(_req, res, next) {
  if (!process.env.GEMINI_API_KEY) {
    res.status(500).json({ error: "GEMINI_API_KEY is not configured." });
    return;
  }

  next();
}

function requireAppToken(req, res, next) {
  if (!sharedAccessToken) {
    res.status(500).json({ error: "APP_ACCESS_TOKEN is not configured." });
    return;
  }

  const token = req.header("x-app-token");
  if (token !== sharedAccessToken) {
    res.status(401).json({ error: "Unauthorized." });
    return;
  }

  next();
}

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    model,
    protected: Boolean(sharedAccessToken),
    analytics: {
      totalRequests: analytics.totalRequests,
      successes: analytics.successes,
      failures: analytics.failures,
      rateLimitErrors: analytics.rateLimitErrors,
      averageResponseChars: analytics.successes
        ? Math.round(analytics.totalResponseChars / analytics.successes)
        : 0
    }
  });
});

app.get(
  "/api/chat-text",
  requireConfiguredApiKey,
  (req, res, next) => {
    const token = req.query?.token;
    if (!sharedAccessToken || token !== sharedAccessToken) {
      res.status(401).type("text/plain").send("Unauthorized.");
      return;
    }

    next();
  },
  async (req, res) => {
    const message = typeof req.query?.message === "string" ? req.query.message : "";
    const stage = typeof req.query?.stage === "string" ? req.query.stage : "";
    const topic = typeof req.query?.topic === "string" ? req.query.topic : "";
    const source = typeof req.query?.source === "string" ? req.query.source : "appinventor";
    const sessionId = typeof req.query?.sessionId === "string" ? req.query.sessionId : "";
    const userId = typeof req.query?.userId === "string" ? req.query.userId : "";

    if (!message.trim()) {
      res.status(400).type("text/plain").send("message must be a non-empty string.");
      return;
    }

    try {
      const mode = classifyMessage(message);
      const language = detectLanguage(message);
      const sessionKey = getSessionKey({ req, source, sessionId, userId, stage, topic });

      if (isHighRiskMessage(message)) {
        const reply = getSafetyReply(language);
        storeConversationTurn(sessionKey, "user", message);
        storeConversationTurn(sessionKey, "sakhi", reply);
        recordSuccess(topic, reply);
        res.type("text/plain").send(reply);
        return;
      }

      if (mode === "casual") {
        const reply = getCasualReply(message);
        storeConversationTurn(sessionKey, "user", message);
        storeConversationTurn(sessionKey, "sakhi", reply);
        recordSuccess(topic, reply);
        res.type("text/plain").send(reply);
        return;
      }

      const history = getConversationHistory(sessionKey);
      const peerSnippets = getEffectivePeerSnippets(topic, getPeerContext(stage, message));
      const text = await generateSakhiReply({
        message,
        stage,
        topic,
        language,
        peerSnippets,
        history
      });
      storeConversationTurn(sessionKey, "user", message);
      storeConversationTurn(sessionKey, "sakhi", text);
      recordSuccess(topic, text);
      res.type("text/plain").send(text);
    } catch (error) {
      recordFailure(topic, error);
      res.status(503).type("text/plain").send(getTemporaryFailureReply(detectLanguage(message)));
    }
  }
);

app.post(
  "/api/chat-text",
  requireConfiguredApiKey,
  (req, res, next) => {
    const token = req.query?.token || req.header("x-app-token");
    if (!sharedAccessToken || token !== sharedAccessToken) {
      res.status(401).type("text/plain").send("Unauthorized.");
      return;
    }

    next();
  },
  async (req, res) => {
    const message = typeof req.body === "string" ? req.body : "";
    const stage = typeof req.query?.stage === "string" ? req.query.stage : "";
    const topic = typeof req.query?.topic === "string" ? req.query.topic : "";
    const source = typeof req.query?.source === "string" ? req.query.source : "appinventor";
    const sessionId = typeof req.query?.sessionId === "string" ? req.query.sessionId : "";
    const userId = typeof req.query?.userId === "string" ? req.query.userId : "";

    if (!message.trim()) {
      res.status(400).type("text/plain").send("message must be a non-empty string.");
      return;
    }

    try {
      const mode = classifyMessage(message);
      const language = detectLanguage(message);
      const sessionKey = getSessionKey({ req, source, sessionId, userId, stage, topic });

      if (isHighRiskMessage(message)) {
        const reply = getSafetyReply(language);
        storeConversationTurn(sessionKey, "user", message);
        storeConversationTurn(sessionKey, "sakhi", reply);
        recordSuccess(topic, reply);
        res.type("text/plain").send(reply);
        return;
      }

      if (mode === "casual") {
        const reply = getCasualReply(message);
        storeConversationTurn(sessionKey, "user", message);
        storeConversationTurn(sessionKey, "sakhi", reply);
        recordSuccess(topic, reply);
        res.type("text/plain").send(reply);
        return;
      }

      const history = getConversationHistory(sessionKey);
      const peerSnippets = getEffectivePeerSnippets(topic, getPeerContext(stage, message));
      const text = await generateSakhiReply({
        message,
        stage,
        topic,
        language,
        peerSnippets,
        history
      });
      storeConversationTurn(sessionKey, "user", message);
      storeConversationTurn(sessionKey, "sakhi", text);
      recordSuccess(topic, text);
      res.type("text/plain").send(text);
    } catch (error) {
      recordFailure(topic, error);
      res.status(503).type("text/plain").send(getTemporaryFailureReply(detectLanguage(message)));
    }
  }
);

app.post("/api/chat", requireConfiguredApiKey, requireAppToken, async (req, res) => {
  const { message, userId, sessionId, stage, topic } = req.body ?? {};

  if (typeof message !== "string" || !message.trim()) {
    res.status(400).json({ error: "message must be a non-empty string." });
    return;
  }

  try {
    const mode = classifyMessage(message);
    const language = detectLanguage(message);
    const sessionKey = getSessionKey({
      req,
      source: "mit-app-inventor-ai2a",
      sessionId,
      userId,
      stage,
      topic
    });

    if (isHighRiskMessage(message)) {
      const reply = getSafetyReply(language);
      storeConversationTurn(sessionKey, "user", message);
      storeConversationTurn(sessionKey, "sakhi", reply);
      recordSuccess(topic, reply);
      res.json({
        reply,
        model,
        mode: "safety",
        peerContextCount: 0,
        topic
      });
      return;
    }

    if (mode === "casual") {
      const reply = getCasualReply(message);
      storeConversationTurn(sessionKey, "user", message);
      storeConversationTurn(sessionKey, "sakhi", reply);
      recordSuccess(topic, reply);
      res.json({
        reply,
        model,
        mode,
        peerContextCount: 0
      });
      return;
    }

    const history = getConversationHistory(sessionKey);
    const peerSnippets = getEffectivePeerSnippets(topic, getPeerContext(stage, message));
    const text = await generateSakhiReply({
      message,
      stage,
      topic,
      language,
      peerSnippets,
      history
    });
    storeConversationTurn(sessionKey, "user", message);
    storeConversationTurn(sessionKey, "sakhi", text);
    recordSuccess(topic, text);

    res.json({
      reply: text,
      model,
      mode,
      peerContextCount: peerSnippets.length,
      topic
    });
  } catch (error) {
    recordFailure(topic, error);
    res.status(503).json({
      error: getTemporaryFailureReply(detectLanguage(message))
    });
  }
});

app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});

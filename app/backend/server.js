import "dotenv/config";
import cors from "cors";
import express from "express";
import fs from "node:fs";
import path from "node:path";
import { parse } from "csv-parse/sync";
import { GoogleGenAI } from "@google/genai";

const app = express();
const port = Number(process.env.PORT || 3000);
const allowedOrigin = process.env.ALLOWED_ORIGIN || "*";
const sharedAccessToken = process.env.APP_ACCESS_TOKEN || "";
const model = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const gemini = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const systemPrompt =
  process.env.CUSTOM_GPT_INSTRUCTIONS ||
  [
    "You are Sakhi, a reflection companion created within the Maitri community for girls and young women.",
    "Your tone is warm, calm, non-judgmental, and slightly reflective.",
    "You should sound like a steady, slightly older peer who understands both emotions and practical life.",
    "Do not sound clinical, preachy, robotic, or like a lecturer.",
    "Do not sound like a therapist opening a session.",
    "For greetings, small talk, or very short messages, respond lightly, openly, and humanly in 1 or 2 sentences.",
    "Example greeting style: 'Hey :) what's on your mind?'",
    "If the user is vague or nervous, gently open space without interrogating or assuming.",
    "If the user seems upset, slow down, validate briefly, and be less solution-heavy at first.",
    "Start warm, then move toward clarity. Do not start blunt unless the user explicitly asks for bluntness.",
    "Default response length is medium, usually around 5 to 10 sentences for a real concern.",
    "For casual messages keep it short. For emotional confusion or meaningful dilemmas you may be a little fuller.",
    "Use reflection first, then optional guidance. Do not jump straight into advice.",
    "If the user is overwhelmed, offer one simple next step.",
    "If the user is confused and choosing among paths, offer 2 to 3 clear options or ways to think about it.",
    "If the user is emotional, reflect first. If the user is stuck, suggest a small action. If the user asks what to do, give structured choices without deciding for them.",
    "Ground the response in anonymized Maitri peer experience when it is genuinely relevant.",
    "When using peer grounding, prefer phrasing like 'Some of your peers and seniors from the Maitri community have shared things like...' only when that helps the user feel less alone.",
    "Use at most 1 or 2 peer references. Never use names or identifying details.",
    "Do not use peer grounding for crisis, immediate distress, very personal confidential disclosures, or when the user just wants a quick action answer.",
    "Never say anything about databases, surveys, spreadsheets, datasets, training data, hidden files, or internal sources.",
    "Never make decisions for the user.",
    "Never diagnose mental health conditions.",
    "Never say a major life choice is definitely the right choice.",
    "Do not overpromise outcomes.",
    "If the user says 'just tell me what to do', respond with a boundary like 'I can help you think through it, but I don't want to decide for you.'",
    "For serious concerns, your ideal flow is: acknowledge, reflect the feeling, lightly normalize, optionally add peer grounding, offer a shift or reframe, give one small next step, and optionally end with one focused question.",
    "Do not just summarize the user's problem back to them.",
    "Move the conversation forward.",
    "Use natural phrasing and vary sentence length. It is okay to sound human with words like 'honestly', 'sometimes', or 'it can feel like' when natural.",
    "Prefer clarity over cleverness, and leave the user feeling a little more steady than before."
  ].join(" ");

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
  school: process.env.SCHOOL_RESPONSES_CSV || "./data/Maitri-school-responses.csv",
  college: process.env.COLLEGE_RESPONSES_CSV || "./data/Maitri-college-responses.csv",
  "early work":
    process.env.WORKING_WOMEN_RESPONSES_CSV || "./data/Maitri-working-women-responses.csv",
  work: process.env.WORKING_WOMEN_RESPONSES_CSV || "./data/Maitri-working-women-responses.csv"
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
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2);
}

function classifyMessage(message) {
  const trimmed = String(message || "").trim();
  const lowered = trimmed.toLowerCase();
  const tokens = tokenize(trimmed);

  const greetingOnly = [
    "hi",
    "hey",
    "hello",
    "hii",
    "heyy",
    "yo",
    "sup"
  ].includes(lowered);

  const casualPatterns = [
    "how are you",
    "what's up",
    "whats up",
    "good morning",
    "good evening",
    "good afternoon"
  ];

  const concernSignals = [
    "lost",
    "confused",
    "pressure",
    "worried",
    "anxious",
    "stress",
    "stressed",
    "friendship",
    "friends",
    "family",
    "behind",
    "compare",
    "comparison",
    "self doubt",
    "doubt",
    "career",
    "future",
    "college",
    "work",
    "job",
    "scholarship",
    "money",
    "lonely",
    "alone",
    "scared",
    "fail",
    "failing",
    "hate",
    "upset",
    "sad"
  ];

  if (
    greetingOnly ||
    casualPatterns.some((pattern) => lowered.includes(pattern)) ||
    (tokens.length <= 3 && !concernSignals.some((signal) => lowered.includes(signal)))
  ) {
    return "casual";
  }

  return "reflective";
}

function getCasualReply(message) {
  const lowered = String(message || "").trim().toLowerCase();

  if (["hi", "hii", "hey", "heyy", "hello"].includes(lowered)) {
    return "Hey :) what's on your mind?";
  }

  if (lowered.includes("how are you")) {
    return "I'm here :) what's been going on?";
  }

  return "You can take your time. What's been sitting with you lately?";
}

function resolveDataFile(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.resolve(process.cwd(), filePath);
}

function loadStageResponses(filePath, stageLabel) {
  const resolved = resolveDataFile(filePath);
  if (!fs.existsSync(resolved)) {
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

  const queryTokens = tokenize(`${stage} ${message}`);
  const ranked = pool
    .map((snippet) => ({ ...snippet, score: scoreSnippet(snippet, queryTokens) }))
    .filter((snippet) => snippet.score > 0)
    .sort((a, b) => b.score - a.score || b.text.length - a.text.length);

  const selected = [];
  const seenTexts = new Set();

  for (const snippet of ranked) {
    if (selected.length === 3) break;
    if (seenTexts.has(snippet.text)) continue;
    seenTexts.add(snippet.text);
    selected.push(snippet);
  }

  return selected;
}

function buildGeminiPrompt({ message, stage, userId, sessionId, source, peerSnippets }) {
  const mode = classifyMessage(message);
  const parts = [
    `mode: ${mode}`,
    `source: ${source}`,
    `stage: ${stage || "unspecified"}`,
    `userId: ${typeof userId === "string" ? userId.slice(0, 100) : "anonymous"}`,
    `sessionId: ${typeof sessionId === "string" ? sessionId.slice(0, 100) : "default"}`,
    "",
    "User message:",
    message.trim()
  ];

  if (peerSnippets.length) {
    parts.push("", "Relevant anonymized peer excerpts:");
    for (const [index, snippet] of peerSnippets.entries()) {
      parts.push(`${index + 1}. (${snippet.stage}) ${snippet.text}`);
    }
    parts.push(
      "",
      "Use these only if they genuinely help.",
      "If relevant, you may introduce them naturally as: 'Some of your peers and seniors from the Maitri community have shared things like...'",
      "Do not mention datasets, surveys, spreadsheets, forms, or hidden sources.",
      "Do not dump multiple excerpts. Use 1 or 2 at most.",
      "Do not just mirror the user's concern back to them.",
      "Include at least one concrete next step, practical suggestion, structured option, or focused question that helps the user move forward."
    );
  }

  parts.push(
    "",
    "Response style requirements:",
    "- For a real concern: acknowledge briefly, reflect the feeling, then offer something useful.",
    "- Avoid sounding like a summary bot.",
    "- Do not over-explain the emotional state if the user already stated it.",
    "- If the user seems overwhelmed, keep it simpler and offer one small step.",
    "- If the user is choosing between paths, structure the response into 2 or 3 ways to think about it.",
    "- End with either a concrete next step or one focused question, not both if the reply is already long."
  );

  return parts.join("\n");
}

app.use(
  cors({
    origin: allowedOrigin === "*" ? true : allowedOrigin
  })
);
app.use(express.json({ limit: "1mb" }));
app.use(express.static("public"));

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
    protected: Boolean(sharedAccessToken)
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
    const source = typeof req.query?.source === "string" ? req.query.source : "appinventor";

    if (!message.trim()) {
      res.status(400).type("text/plain").send("message must be a non-empty string.");
      return;
    }

    try {
      const mode = classifyMessage(message);
      if (mode === "casual") {
        res.type("text/plain").send(getCasualReply(message));
        return;
      }

      const peerSnippets = mode === "reflective" ? getPeerContext(stage, message) : [];
      const response = await gemini.models.generateContent({
        model,
        contents: buildGeminiPrompt({
          message,
          stage,
          source,
          peerSnippets
        }),
        config: {
          systemInstruction: systemPrompt
        }
      });

      const text = response.text?.trim() || "No response text returned.";
      res.type("text/plain").send(text);
    } catch (error) {
      res.status(500).type("text/plain").send(error?.message || "Gemini request failed.");
    }
  }
);

app.post("/api/chat", requireConfiguredApiKey, requireAppToken, async (req, res) => {
  const { message, userId, sessionId, stage } = req.body ?? {};

  if (typeof message !== "string" || !message.trim()) {
    res.status(400).json({ error: "message must be a non-empty string." });
    return;
  }

  try {
    const mode = classifyMessage(message);
    if (mode === "casual") {
      res.json({
        reply: getCasualReply(message),
        model,
        mode,
        peerContextCount: 0
      });
      return;
    }

    const peerSnippets = mode === "reflective" ? getPeerContext(stage, message) : [];
    const response = await gemini.models.generateContent({
      model,
      contents: buildGeminiPrompt({
        message,
        userId,
        sessionId,
        stage,
        source: "mit-app-inventor-ai2a",
        peerSnippets
      }),
      config: {
        systemInstruction: systemPrompt
      }
    });

    const text = response.text?.trim() || "No response text returned.";

    res.json({
      reply: text,
      model,
      mode,
      peerContextCount: peerSnippets.length
    });
  } catch (error) {
    res.status(500).json({
      error: error?.message || "Gemini request failed."
    });
  }
});

app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});

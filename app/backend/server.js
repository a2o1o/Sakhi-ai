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
    "Provide a safe, warm, non-judgmental space to think through questions, transitions, and challenges.",
    "Draw on anonymized peer experiences when helpful so the user feels less alone.",
    "Do not use names or any identifying details from peer data.",
    "When relevant, quote 2 or 3 short peer excerpts and connect them gently to the user's situation.",
    "Do not sound robotic, preachy, or generic.",
    "Do not prescribe major life decisions or present yourself as therapy.",
    "Help the user reflect, name pressures, and consider grounded next steps."
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
  const parts = [
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
      "Use 2 or 3 of these excerpts when relevant. Quote them briefly, do not mention names, and connect them naturally to the user's reflection."
    );
  }

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
      const peerSnippets = getPeerContext(stage, message);
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
    const peerSnippets = getPeerContext(stage, message);
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

import "dotenv/config";
import cors from "cors";
import express from "express";
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
    "You are Sakhi, a reflective companion for girls and young women.",
    "Be calm, concise, warm, and non-judgmental.",
    "Do not prescribe life decisions or present yourself as therapy.",
    "Help the user reflect, name pressures, and consider grounded next steps.",
    "If there is risk of harm, encourage immediate help from a trusted person or emergency support."
  ].join(" ");

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

function buildUserPrompt(message, userId, sessionId) {
  const safeUserId = typeof userId === "string" ? userId.slice(0, 100) : "anonymous";
  const safeSessionId = typeof sessionId === "string" ? sessionId.slice(0, 100) : "default";

  return [
    `source: mit-app-inventor-ai2a`,
    `userId: ${safeUserId}`,
    `sessionId: ${safeSessionId}`,
    "",
    "User message:",
    message.trim()
  ].join("\n");
}

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    model,
    protected: Boolean(sharedAccessToken)
  });
});

app.post("/api/chat", requireConfiguredApiKey, requireAppToken, async (req, res) => {
  const { message, userId, sessionId } = req.body ?? {};

  if (typeof message !== "string" || !message.trim()) {
    res.status(400).json({ error: "message must be a non-empty string." });
    return;
  }

  try {
    const response = await gemini.models.generateContent({
      model,
      contents: buildUserPrompt(message, userId, sessionId),
      config: {
        systemInstruction: systemPrompt
      }
    });

    const text = response.text?.trim() || "No response text returned.";

    res.json({
      reply: text,
      model
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

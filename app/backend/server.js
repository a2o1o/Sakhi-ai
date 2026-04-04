import "dotenv/config";
import cors from "cors";
import express from "express";
import OpenAI from "openai";

const app = express();
const port = Number(process.env.PORT || 3000);
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const allowedOrigin = process.env.ALLOWED_ORIGIN || "*";
const sharedAccessToken = process.env.APP_ACCESS_TOKEN || "";
const model = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const systemPrompt =
  process.env.CUSTOM_GPT_INSTRUCTIONS ||
  [
    "You are an assistant used from an MIT App Inventor app.",
    "Keep answers concise, structured, and safe for general users.",
    "If a request needs sensitive actions, refuse and explain briefly.",
    "When steps are helpful, return short numbered lists."
  ].join(" ");

app.use(
  cors({
    origin: allowedOrigin === "*" ? true : allowedOrigin
  })
);
app.use(express.json({ limit: "1mb" }));
app.use(express.static("public"));

function requireConfiguredApiKey(req, res, next) {
  if (!process.env.OPENAI_API_KEY) {
    res.status(500).json({ error: "OPENAI_API_KEY is not configured." });
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

app.post("/api/chat", requireConfiguredApiKey, requireAppToken, async (req, res) => {
  const { message, userId, sessionId } = req.body ?? {};

  if (typeof message !== "string" || !message.trim()) {
    res.status(400).json({ error: "message must be a non-empty string." });
    return;
  }

  try {
    const response = await openai.responses.create({
      model,
      instructions: systemPrompt,
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: message.trim()
            }
          ]
        }
      ],
      metadata: {
        source: "mit-app-inventor-ai2a",
        userId: typeof userId === "string" ? userId.slice(0, 100) : "anonymous",
        sessionId: typeof sessionId === "string" ? sessionId.slice(0, 100) : "default"
      }
    });

    const text = response.output_text?.trim() || "No response text returned.";

    res.json({
      reply: text,
      model: response.model
    });
  } catch (error) {
    const status = error?.status || 500;
    res.status(status).json({
      error: error?.message || "OpenAI request failed."
    });
  }
});

app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});

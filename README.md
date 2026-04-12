# Sakhi AI

Sakhi AI is an MIT App Inventor project with a protected Gemini-backed reflection service.

## Structure

- `app/Sakhi.aia`: importable MIT App Inventor project
- `app/source/`: extracted App Inventor source files for version control
- `app/backend/`: Node/Express backend and local browser test client
- `app/.env.example`: backend environment variable template
- `app/appinventor/README.md`: App Inventor integration notes
- `data/`: Maitri response CSVs used only by the backend

## App flow

- `Screen1`: welcome screen
- `StageScreen`: stage selection
- `SchoolChoiceScreen`: school topic routing
- `CollegeChoiceScreen`: college topic routing
- `ConcernScreen`: reflective and topic-aware Sakhi chat
- `Screen2`: about and safeguards

## Backend

The backend keeps the Gemini API key off the mobile client, retrieves relevant anonymized Maitri reflections server-side, and applies Sakhi's shared instructions for:

- language mirroring across English, Hinglish, and Hindi
- stage-aware and topic-aware prompting
- short session memory for continuity
- high-risk safety routing
- graceful fallback on temporary model failures
- lightweight request analytics for debugging

Required environment variables are documented in `app/.env.example`.

## Local setup

1. Copy `app/.env.example` to the backend environment file you use for deployment or local testing.
2. From `app/backend`, run `npm install`.
3. From `app/backend`, run `npm start`.

## Notes

- Raw Maitri CSVs should remain server-side and should not be committed to a public repository if they contain personal data.
- `node_modules/` is intentionally excluded from version control.
- The App Inventor project should point to the deployed backend endpoint, not a local machine URL.

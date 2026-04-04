# Sakhi AI

This repository contains the Sakhi AI MIT App Inventor app and its OpenAI-backed middleware.

## Structure

- `app/Sakhi.aia`: importable MIT App Inventor project
- `app/source/`: extracted App Inventor source files for version control
- `app/backend/`: Node/Express middleware and local test client
- `app/appinventor/README.md`: App Inventor integration notes

## Backend

The backend keeps the OpenAI API key off the mobile client and applies shared Sakhi instructions server-side.

Required environment variables are documented in `app/.env.example`.

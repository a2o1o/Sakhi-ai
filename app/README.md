# Sakhi AI App

This folder contains the App Inventor app and the middleware used to power the Sakhi reflection flow.

## Contents

- `Sakhi.aia`: current MIT App Inventor project
- `source/`: extracted App Inventor files for editing and diffs
- `backend/`: Express server, package files, and test UI
- `.env.example`: backend environment template
- `appinventor/README.md`: notes for wiring the App Inventor client to the backend

## App flow

- `Screen1`: welcome screen
- `StageScreen`: stage selection
- `ConcernScreen`: reflective prompt and AI response
- `Screen2`: about and safeguards

## Important note

The App Inventor project still contains a placeholder backend URL. Update that to your deployed backend endpoint before production use.

# pH Strip Reader API

## How to Calibrate
1. Run `python calibrate.py calib` locally with your 6.1 - 7.9 buffer photos.
2. Commit the generated `chart.json` to this repository.
3. Render will automatically load `chart.json` on startup for maximum accuracy.

## n8n Workflow
Send the `file` and the `boxes` JSON from Gemini to `/analyze`.

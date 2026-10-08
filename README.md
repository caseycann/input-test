# Input Lab

A home page (`index.html`) linking to small browser tools, each in its own folder:

- `pose/`: Webcam → MediaPipe Pose Landmarker → skeleton overlay + raw landmark data.
- `quadrants/`: Camera split into 4, 8 or 16 zones; a button lights when enough of a zone is darker than a chosen grey (default 60% grey / #666).
- `hands/`: Webcam → MediaPipe Gesture Recognizer → per-hand gesture buttons (7 built-in gestures + pinch), skeleton overlay, raw landmark JSON.

To add a tool, create a new folder with its own `index.html`, then copy a card in the root `index.html` and point it at the folder.

Static site, no build step. All inference runs in the browser; no video leaves the device.

## Run locally

Camera access needs `localhost` or HTTPS:

```sh
npx serve .
# or: python3 -m http.server 8000
```

## Deploy to Vercel

```sh
npx vercel        # preview
npx vercel --prod # production
```

Or import the repo in the Vercel dashboard with Framework Preset = "Other" and no build command.

## Pose data format

Per pose, 33 landmarks (BlazePose topology):

- `landmarks`: `x`, `y` normalized 0–1 to the image (un-mirrored), `z` relative depth (hips ≈ 0, smaller = closer to camera), `visibility` 0–1.
- `worldLandmarks`: `x`, `y`, `z` in meters, origin at the center of the hips.

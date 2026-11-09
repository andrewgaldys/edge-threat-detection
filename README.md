# Edge Threat Detection: browser edition

A live threat-detection dashboard that runs entirely in your browser. It uses a YOLOv8n model to find people, cars, dogs, and drones (through the closest COCO class, airplane), and raises an alert when one stays in view for more than 30 analysed frames.

## How it works

GitHub Pages only serves files, so nothing here runs on a server:

- The model runs on your own device through WebAssembly.
- Video from your camera never leaves your browser. The page's Content Security Policy blocks it from sending data to any other site.
- Breach snapshots stay in the page's memory and disappear when you close it.
- Network alerts come from simulated traffic, screened by the same z-score rules as the full edition.

With no camera, the page shows a demonstration scene and still runs real detection on it.

## Using it

- **Source:** pick the demonstration scene or a camera. Your computer's built-in camera is preferred over a nearby phone.
- **Watch list:** untick any class you want to ignore.
- **Perimeter zone:** drag on the video. Only objects centred inside the zone count.
- **Filters:** narrow the alert feed by severity or by source.

## Licence

AGPL-3.0, because the page distributes an Ultralytics YOLOv8 model. See `LICENSE` and `NOTICE.txt`.

# Figure 2 Generation Prompt

## Mode

Built-in OpenAI image generation with three references:

- The first Figure 2 draft as a structural reference.
- The user-edited Figure 1 hero as the visual style reference.
- Real LTCC images of identity 0028 as the same-person reference.

## Prompt

Create a revised, publication-ready 16:9 scientific workflow diagram for an
NVIDIA research blog.

Rebuild and simplify the earlier pipeline diagram. Remove its camera-scene
panels, packet details, coordinates, and dense technical annotations. Match
the hero image's matte-black background, NVIDIA-green headings and thin
outlines, crisp white sans-serif typography, numbered green circles, dashed
directional arrows, restrained technical geometry, generous negative space,
and premium editorial finish.

Use the first three black-shirt views of LTCC identity 0028 as the recognizable
same-person example. Preserve the visible `93` shirt cue so the repeated
identity is immediately clear. Present them as clean person crops without
camera-scene backgrounds.

Explain the ReID workflow at a glance using exactly five numbered stages:

1. Camera Streams
2. YOLO Detection
3. Person Crops
4. ReID + Tracking
5. Global Identity

Connect the stages with fine dashed white arrows.

For Camera Streams, show three minimal camera icons labeled `CAM 1`, `CAM 2`,
and `CAM N`. Do not show camera footage or monitoring screens.

For YOLO Detection, show a clean block labeled `YOLO Person Detection`.
Directly below it, show a restrained NVIDIA-green accelerator symbol and the
text `TensorRT Optimized`. Represent detection conceptually with a simple
person silhouette and thin green outline. Do not show coordinates, confidence
scores, resolutions, classes, or bounding-box data.

For Person Crops, show three large, evenly spaced crops of the same LTCC
subject, all wearing the distinctive black `93` shirt and seen from different
angles. Use the heading `Same Person, Different Views`. Do not use the
white-shirt crop.

For ReID + Tracking, create a polished central processing block labeled
`TAO-Trained Swin ReID`. Directly below it, use the same accelerator symbol and
the text `TensorRT Optimized`. Show the three crops transforming into a compact
green-and-white embedding cluster. Beneath it, show two simple local track
paths labeled `Per-Camera Tracking`. Avoid tensor dimensions, vector lengths,
batch sizes, provider names, framework names, and internal model diagrams.

For Global Identity, show the crop views and track paths converging into one
bold green identifier labeled `ONE GLOBAL ID`. Beneath it, use three simple
output icons labeled `Video`, `Event Logs`, and `Identity Journeys`. Do not
show a dashboard.

Use this bottom conclusion:

`One person. Multiple cameras. One persistent identity.`

Use near-black and charcoal, white and soft-gray text, and restrained NVIDIA
green (`#76B900`) for headings, arrows, outlines, model accents, and the
identity result. Keep NVIDIA branding technical and subtle.

Avoid camera footage, corridor scenes, dashboards, browsers, charts,
terminals, code, packet diagrams, coordinates, numerical boxes, confidence
values, dimensions, ONNX Runtime, WebSocket, Triton, DeepStream, cloud
symbols, face-recognition overlays, biometric face meshes, police imagery,
control-room aesthetics, cyberpunk styling, generic AI brains, giant logos,
advertising slogans, invented metrics, tiny text, duplicate labels,
decorative clutter, spelling errors, and watermarks.

# Final Image Generation Prompts

The following assets were created with the built-in OpenAI image generation
tool. No CLI fallback or external API key was used.

## Figure 1: Final Editorial Hero

```text
Use case: scientific-educational
Asset type: final premium 16:9 cover image for an NVIDIA technical research blog.
Primary request: Create a publication-grade editorial visual about synthetic-data filtering for clothing-change person re-identification. Tell one concise, scientifically faithful story: an annotated synthetic camera scene produces many redundant full-person crops; the research pipeline retains only three representative variants; a Swin ReID embedding model then associates one real LTCC query in black clothing with the same person wearing a white shirt across three cameras.

Input images and roles:
- Image 1: actual project synthetic 3D source frame with JSON-derived person bounding boxes. Preserve it as the dominant left-side scene and keep the green boxes faithful.
- Image 2: actual near-duplicate synthetic full-person crops from one person, camera, frame, and bounding box. Preserve the repeated woman in a pale shirt and jeans; show many variants before selection and exactly three after selection.
- Image 3: actual LTCC same-identity clothing-change observations. Preserve the black-jacket query and three white-shirt gallery views as authentic image panels.

Scene/backdrop: matte charcoal-to-black editorial canvas with subtle fine paper texture and extremely faint geometric registration lines. No environment other than the source frame, no computer screen, no dashboard.
Style/medium: high-end science-journal visual and contemporary NVIDIA technical-blog art direction; elegant photographic collage with precise data-art geometry, controlled depth, crisp source-image panels, and generous negative space. Serious and technically credible, not a presentation slide.
Composition/framing: panoramic 16:9, balanced and filled. The large annotated synthetic scene occupies about 36% at left and bleeds slightly off the edge. From a selected box, a horizontal sequence of eight near-duplicate full-person crops flows toward center. A narrow restrained green selection plane reduces the sequence to exactly three representative crops. Immediately behind them, imply Swin processing with offset square patch windows and a compact 1024-dimensional embedding cloud made of small graphite points. On the right, place one black-jacket LTCC query crop, then one clean green trajectory branching to exactly three white-shirt observations from distinct camera views. The real query-to-gallery identity association must be visually unmistakable. No check marks.
Lighting/mood: premium, rigorous, calm, focused; crisp highlights on image edges, deep matte shadows, no theatrical glow.
Color palette: charcoal, black, cool gray, natural source-image colors, and sparse NVIDIA green (#76B900) only for bounding boxes, the selection plane, selected crop edges, and identity trajectory.
Text: absolutely no text, numbers, letters, labels, captions, legends, logos, or watermarks.
Constraints: Make synthetic data and the eight-to-three filtering event the strongest visual focus. Preserve the distinction between synthetic training imagery and real LTCC evaluation imagery. Show exactly three retained synthetic variants and exactly three real white-shirt gallery matches. Keep faces small and incidental. Do not invent metrics, hardware, datasets, or UI.
Avoid: dashboard, monitor, browser, terminal, flowchart, PowerPoint infographic, boxed sections, folder icons, check marks, badges, buttons, glossy glass UI, holograms, cyberpunk neon, surveillance room, facial scan overlays, police imagery, giant GPU, giant logo, generic stock people, brain icon, humanoid robot, fake camera labels, illegible microtext.
```

References:

1. `references/synthetic_scene_json_boxes.png`
2. `references/synthetic_same_moment_full_person_variants.png`
3. `references/ltcc_identity_0009_cross_camera.png`

## Figure 3: Editorial Filtering Method

```text
Use case: scientific-educational
Asset type: publication-grade 16:9 method figure for a technical research blog.
Primary request: Visualize synthetic-data redundancy and representative filtering using the actual project crops. A dense field of near-duplicate synthetic observations from one person-and-moment group is reduced to exactly three retained variants, communicating that the project removed repetition while keeping useful appearance variation.

Input image role:
- Image 1 contains actual synthetic crops of the same rendered woman, camera, source frame, and bounding box under different generated variants. Preserve her pale shirt, light jeans, pose, crop framing, and recognizably synthetic rendering. Do not replace her with a photorealistic or different person.

Scene/backdrop: premium matte black editorial field with subtle paper texture and a faint scientific grid.
Style/medium: high-end data-art photograph for a science journal; elegant, minimal, precise, tactile, and technically credible. Not an infographic slide.
Composition/framing: panoramic 16:9. Across the left two-thirds, arrange a dense perspective field of many repeated crop panels, using the same woman from Image 1 with the real slight exposure and blur variations. The repeated panels should recede in depth and feel abundant without becoming chaotic. Near center-right, pass the field through one thin translucent NVIDIA-green selection plane. On the right, show exactly three larger retained crop panels, evenly spaced and sharply edged, each taken from Image 1. Use subtle graphite registration marks and three small green selection points; no other interface elements. Leave clean negative space around the retained triplet.
Lighting/mood: controlled museum-like lighting, crisp panel edges, deep matte shadows, quiet scientific confidence.
Color palette: black, graphite, cool gray, source-image colors, sparse NVIDIA green (#76B900) only at the selection plane and retained borders.
Text: no text, numbers, letters, labels, logos, captions, legends, or watermarks.
Constraints: The left side must visibly represent repetitive variants of one underlying moment, not different poses or identities. The right side must contain exactly three retained panels. Preserve the actual synthetic character and crop appearance. No dashboard and no model architecture in this figure.
Avoid: PowerPoint flowchart, boxes with labels, folder icons, funnel clip art, check marks, badges, buttons, dashboard, monitor, terminal, spreadsheet, glossy UI, hologram, cyberpunk neon, photorealistic replacement person, generic stock photography, huge logo, advertising layout, fake metrics, illegible text.
```

Reference:

1. `references/synthetic_same_moment_full_person_variants.png`

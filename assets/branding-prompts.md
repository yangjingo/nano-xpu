# Nano-XPU branding v2

2026-11-13。使用内置 `image_gen` 编辑生成，未使用 CLI/API fallback。

README 使用 `nanoxpu-banner-v2.png`，显示宽度 560px。配套独立图标为 `nanoxpu-icon-v2.png`，带 alpha 透明通道。旧版 Nano-XPU 横幅、图标及未使用的 nano-vllm 标识已删除；历史素材仍可从 Git 提交 `b734b0b797d58e9b5e470dec420c187709d4f4fd` 查阅。

方向：保留四臂 X 与芯片互联意象，简化中心细节，使用深石墨色和低饱和浅绿；去掉原版金属浮雕、重阴影和粒子波纹。横幅只保留名称，不通过图像宣称尚未实现的异构能力。

## Banner prompt

历史参考输入：改版前的 `nanoxpu-banner.png`（编辑目标）、`nanoxpu-icon.png`（标识参考）。这些文件现已移除，记录名称仅用于追溯生成过程。

```text
Edit target: image 1 is the existing Nano-XPU GitHub README wordmark banner. Image 2 is its existing standalone identity mark. Redesign and refine this actual project logo, retaining its identity: a bold geometric X built from four diagonal hardware-like arms, with a small central interconnect/chip motif, and the exact wordmark 'Nano-XPU'. Use case: polished, restrained open-source inference infrastructure branding, immediately readable when displayed at 560 CSS pixels wide in a GitHub README. Preserve the recognizable X silhouette, make its geometry balanced and precise, replace the clutter of miniature squares in the center with one beautifully resolved tiny sage-green square die and just two simple short horizontal connections. Remove the old metallic surface, bevels, photographic effects, deep shadows, dotted particle waves, and enclosing rounded presentation card. Style: crisp flat graphic design, serious engineering, elegantly minimal; original brand identity evolution, not a mockup. Palette: dark graphite #253734 for the X and wordmark, muted sage #9CBBAA only on the center die, near-white #FAFAF8 background. Horizontal composition, wide approximately 3:1 canvas; large refined X mark on the left, generous but controlled gap, exact wordmark 'Nano-XPU' on the right in a beautifully kerned bold contemporary sans serif with confident clean letterforms. No subtitle or any other text. Keep the complete lockup optically centered with around 7 percent outer safety margin; use the canvas efficiently, no large blank top and bottom margins. Clean edges and genuinely flat fills, no texture/no noise/no watermark/no fake UI/no extra symbols. This is the final banner asset itself, not an image of a banner in a scene.
```

## Icon prompt

参考输入：生成后的 `nanoxpu-banner-v2.png`。设置 `transparent_background=true`。

```text
Edit/extract from the provided Nano-XPU brand banner: create the matching standalone square app/repository icon only. Preserve exactly the X emblem at the LEFT of the reference, with its four precise diagonal dark-graphite hardware arms, near-white central interconnect negative space, and small muted-sage square die in the center. Remove all wordmark text and background. Reproduce the same geometry and colors faithfully; do not invent a different symbol. Asset is a square 1:1 transparent PNG, large mark optically centered taking approximately 80 percent of canvas width and height, consistent clear margin on all sides. Use actual alpha transparency around the entire emblem and in the negative-space gaps (the central sage die remains opaque). Crisp flat shapes and edges, no added text, no gray/checkerboard baked background, no texture, no gloss, no bevel, no shadow, no watermark. Final directly usable icon asset, not a presentation mockup.
```

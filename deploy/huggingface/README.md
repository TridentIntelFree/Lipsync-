---
title: Lipsync
emoji: 👄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.21.0
app_file: app.py
pinned: false
short_description: Infer speech from silent video of a speaking face
---

# Lipsync

Infer what someone said from silent video of them speaking.

**Output is a guess that reads as a certainty.** The model gets roughly one word
in five wrong on clean, head-on, well-lit video and worse on anything else, and
many sounds are visually identical — `p`, `b` and `m` are the same picture. The
gaps are filled by a language model that always returns fluent English whether
or not it read anything. Never treat the result as evidence of what a particular
person said.

Source and full documentation: https://github.com/TridentIntelFree/Lipsync-

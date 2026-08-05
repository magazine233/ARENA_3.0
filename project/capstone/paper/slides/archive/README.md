# Archived deck variants

Superseded by the live deliverable, `../TALK_10MIN_formal.*` (Jason's call, 2026-06-12). Kept for reference:

| variant | what it was |
|---|---|
| `TALK_8MIN.*` + `build_deck.py` | the original 8-minute scaffold deck (neutral register) |
| `TALK_8MIN_voice.*` + `build_deck_voice.py` | the 8-minute voice-pass cut (spoken register, 4 landing beats) |
| `TALK_10MIN.*` + `build_deck_10min.py` | the 10-minute voice cut (adds HatCat + geometry near-miss slides) |

All builders run from `project/capstone/` and depend on `../render_slide_assets.py` outputs in `../assets/`.
The formal 10-minute deck carries the same slides, figures, and numbers as the 10-minute voice cut, in the
register of `../../PAPER_DRAFT.md`. To resurrect a variant, `git mv` it back up and rebuild.

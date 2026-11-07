# wowmock

A small headless stand-in for the parts of the World of Warcraft UI API that the addons
in `addons/` use. It lets the addons run outside the game, so they can be tested and
previewed.

- **`wow.lua`** is the mock. It covers frames, buttons, status bars, textures, lines,
  font strings, anchors, scale, events, timers, `OnUpdate` and the tooltip. Where the
  real client is strict, so is the mock:
  - registering an unknown event raises an error;
  - each object type only has its own methods.

  It also reports conflicting anchors, such as a `TOP` plus `CENTER` pair on one
  axis. Text is measured with real glyph widths (padded a little, since the game's
  font runs wider), and strings with a set width wrap. `Dump()` resolves every
  visible region to a screen rectangle.
- **`run.lua`** loads `addons/ProfessionBars` in TOC order and drives it through
  several scenarios:
  - a Mainline client;
  - a Classic-style client;
  - a character with no professions;
  - secret values and unknown professions;
  - a rank that waits on character level, and secondary professions' books and quests;
  - session gains across a `/reload`;
  - a layout stress test with every list shown, and a short screen;
  - camp recipes and long names;
  - the recipe cache as recipes are learned and dropped.

  It checks the results, including that everything in the profession panel fits
  inside it, nothing overlaps, no text is cut off, the panel stays on screen, and
  no error was swallowed by the addon's error guard.
- **`preview.js`** renders a dumped scene to a PNG with headless Chromium
  (Playwright). Icons are replaced by emoji and the game font by Inter, so treat the
  result as a layout and colour preview, not a screenshot.

```sh
# from the repository root
lua5.1 tools/wowmock/run.lua                                  # run the tests
lua5.1 tools/wowmock/run.lua --dump /tmp/scene.json main      # or: panels, styles, themes, gather
node tools/wowmock/preview.js /tmp/scene.json docs/professionbars/main.png

# Full 1080p screen with a stand-in game scene and placeholder HUD
lua5.1 tools/wowmock/run.lua --dump /tmp/screen.json ingame   # or: ingamepanel
node tools/wowmock/preview.js /tmp/screen.json docs/professionbars/ingame.png game
```

The mock is not a full client. It doesn't render real fonts or textures, check
protected or secure behaviour, or know Forever's exact data. `/pb debug` in game is
the source of truth for what the live client returns.

# Profession Bars

Slim, colour-themed skill bars for your professions in World of Warcraft: Forever.
Click a bar and that profession unfolds beside it as a mind map. The map shows:
- your rank, and what it takes to reach the next one;
- a plan for the next skill points, with the reagents you're missing;
- what you can craft right now, and where your gathering skill stands;
- which of your other characters have the same profession.

![Bars and the Blacksmithing mind map](../../docs/professionbars/main.png)

*These images come from the headless test harness (`tools/wowmock`), so they use
stand-in icons and fonts. They are not in-game screenshots.*

## Install

1. Get this folder. On GitHub use **Code → Download ZIP**, then unzip it and find
   `addons/ProfessionBars`.
2. Copy the `ProfessionBars` folder into your Forever client's `Interface/AddOns`
   folder. That client folder sits inside your World of Warcraft directory and has
   `Interface` and `WTF` in it (the beta used `_classic_beta_`). The result must be
   `.../Interface/AddOns/ProfessionBars/ProfessionBars.toc`.
3. On the character select screen, open **AddOns** and tick **Profession Bars**.

The addon targets interface `16001` (Forever 1.60.1). If a later patch marks it as
out of date, tick *Load out of date AddOns* or update the number on the
`## Interface:` line. `/run print(select(4, GetBuildInfo()))` prints the current number.

## Using it

- **Bars.** One bar per profession: primaries first, then Cooking, First Aid and Fishing.
  - The thin gold tick marks where the next rank can be trained. The numbers turn gold
    when you can train it right now.
  - A green **+N** shows the skill you've gained this session.
  - Hover a bar to see which of your other characters have that profession.
- **Click a bar** to unfold its mind map beside the bars. Click the bar again, or
  press Esc, to close it.
- **Moving the map.** Drag it anywhere and it stays there. The profession in the
  middle stays put while branches fold and unfold. `/pb reset` puts it back beside
  the bars.
- **Hover the bars** to show a small header:
  - **edit** lets you click bars to show or hide them;
  - **the colour square** switches theme;
  - **lock** stops the bars from being dragged;
  - **x** hides the bars.

  Drag the header to move the bars.
- **Toggle everything** with `/pb`, a key binding (Key Bindings → AddOns → Profession
  Bars), or the minimap addon menu.

![The four themes](../../docs/professionbars/themes.png)

## The mind map

Click a branch to unfold it, or click the middle to unfold or fold everything. Hover
anything for details.

| Branch | Shows |
| --- | --- |
| **Rank** | Apprentice, Journeyman, Expert and Artisan, and which you have done. |
| **Next step** | The skill and character level needed for the next rank, and whether you can train now. |
| **Plan to N** | The orange crafts that reach your next training level (or your cap), plus a shopping list. Missing reagents are red; crafts your bags already cover say *ready*. |
| **Skill-ups** | Your orange and yellow recipes, with how many you can craft from your bags. |
| **Ready to craft** | Everything you have the reagents for right now. |
| **Recipes** | Known recipes by category. |
| **Camp** | Camp recipes you know. It only appears when the game lists recipes under a camp name. |
| **Ore / Herbs** | Nodes coloured by skill-up chance, plus the next unlocks and the skill still needed. Zone-specific nodes are listed in the tooltip. |
| **Beasts** | The highest level you can skin, the best levels for skill-ups and the next milestones. |
| **Alts** | Your other characters with this profession, in class colours. |

Recipes load the first time you open a profession's window on a character, because
the game only shares recipe colours then. The map tells you when it needs this, and
when the colours are older than your current skill. Bag counts update live. Shift-click
a recipe to link it in chat.

![A gathering profession: Mining](../../docs/professionbars/gather.png)

## Commands

| Command | What it does |
| --- | --- |
| `/pb` | Show or hide the bars. |
| `/pb <profession>` or `/pb map <profession>` | Open a profession's map, e.g. `/pb mining`. |
| `/pb edit` | Choose which professions get a bar. |
| `/pb hide <profession>`, `/pb show <profession\|all>` | Hide or show one bar. |
| `/pb theme [vibrant\|soft\|midnight\|mono]` | Set the theme, or cycle with no name. |
| `/pb color <profession> <#rrggbb\|reset>` | Give one profession your own colour. |
| `/pb lock`, `/pb unlock` | Stop or allow dragging. |
| `/pb scale <0.5-2>`, `/pb mapscale <0.5-2>`, `/pb width <140-400>` | Sizes. |
| `/pb combat` | Hide the bars while in combat (toggle). |
| `/pb alts`, `/pb forget <name>` | List your other characters' professions, or forget one. |
| `/pb reset` | Reset position, size, theme and colours. Alts and recipes are kept. |
| `/pb debug` | Print which game APIs the addon found, what they returned and the last error. |

## Known limitations

- **Not yet run on the live client.** Everything is tested against a mock of the
  game's API (below). If something looks off, `/pb debug` shows what the client
  returned.
- **Plans are a minimum.** They only use orange recipes and what's in your bags.
  They ignore the bank and auction prices, and some recipes turn yellow as you level.
- **Classic numbers.** Rank requirements, trainer levels and node levels are Classic
  values in `Data.lua`. Forever may change some of them, and new zones may add nodes.
- **Big map.** With branches open on both sides the map is about 800 pixels wide;
  `/pb mapscale` shrinks it.
- **English only.** The addon's own text is in English.
- **Alts appear after one login.** Each character shows up once you've logged in on
  it with the addon enabled.

## Notes

- Forever uses the modern (Mainline) addon API. The addon reads skills with
  `GetProfessions`/`GetProfessionInfo` and fills gaps (for example First Aid) from
  `C_TradeSkillUI`.
- Every API call is guarded. If something still fails, it's reported once in chat
  and the rest keeps working, instead of erroring on every event.
- Everything stays on your machine, in `ProfessionBarsDB` (account) and
  `ProfessionBarsCharDB` (per character).

## Development

The addon runs headless against a mock of the WoW API in `tools/wowmock`. From the
repository root:

```sh
lua5.1 tools/wowmock/run.lua        # ~4,800 checks: scanning, bars, plans, map layout, commands
luacheck addons/ProfessionBars      # lint (config in .luacheckrc)
```

`tools/wowmock/README.md` explains how to render the preview images.

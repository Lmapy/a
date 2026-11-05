-- Colour themes. Each theme transforms the per-profession base colour, so a
-- new theme is just a few numbers.
--   sat   : saturation multiplier (1 = as defined, 0 = grey)
--   light : brightness multiplier
--   track : brightness of the empty part of a bar, relative to its colour
--   panel : opacity of the panel backgrounds
--   mono  : optional single colour used for every profession
local _, ns = ...

ns.THEMES = {
	{ key = "vibrant", label = "Vibrant", sat = 1.00, light = 1.00, track = 0.22, panel = 0.55 },
	{ key = "soft", label = "Soft", sat = 0.45, light = 1.12, track = 0.18, panel = 0.45 },
	{ key = "midnight", label = "Midnight", sat = 0.95, light = 0.62, track = 0.34, panel = 0.80 },
	{ key = "mono", label = "Mono", sat = 1.00, light = 1.00, track = 0.22, panel = 0.55, mono = { 0.40, 0.62, 0.96 } },
}

local function clamp(v)
	if v < 0 then
		return 0
	elseif v > 1 then
		return 1
	end
	return v
end

function ns.ThemeIndex(key)
	for i, theme in ipairs(ns.THEMES) do
		if theme.key == key then
			return i
		end
	end
	return nil
end

function ns.GetTheme()
	local index = ns.db and ns.ThemeIndex(ns.db.theme)
	return ns.THEMES[index or 1]
end

function ns.NextThemeKey()
	local index = ns.ThemeIndex(ns.GetTheme().key) or 1
	return ns.THEMES[index % #ns.THEMES + 1].key
end

-- Colour before theming: user override, then the profession's own colour.
function ns.BaseColor(key)
	local custom = ns.db and ns.db.colors and ns.db.colors[key]
	if custom then
		return custom, true
	end
	local prof = ns.PROFESSIONS[key]
	return (prof and prof.color) or ns.DEFAULT_COLOR, false
end

-- On-screen colour of a profession under the active theme.
function ns.Color(key)
	local theme = ns.GetTheme()
	local base, custom = ns.BaseColor(key)
	if theme.mono and not custom then
		base = theme.mono
	end
	local r, g, b = base[1], base[2], base[3]
	local grey = r * 0.30 + g * 0.59 + b * 0.11
	r = grey + (r - grey) * theme.sat
	g = grey + (g - grey) * theme.sat
	b = grey + (b - grey) * theme.sat
	return clamp(r * theme.light), clamp(g * theme.light), clamp(b * theme.light)
end

-- "|cffRRGGBB" escape for coloured chat/tooltip text.
function ns.ColorCode(r, g, b)
	return string.format("|cff%02x%02x%02x", math.floor(clamp(r) * 255 + 0.5), math.floor(clamp(g) * 255 + 0.5), math.floor(clamp(b) * 255 + 0.5))
end

function ns.ParseHex(text)
	if type(text) ~= "string" then
		return nil
	end
	local hex = text:gsub("^#", "")
	if not hex:match("^%x%x%x%x%x%x$") then
		return nil
	end
	return tonumber(hex:sub(1, 2), 16) / 255, tonumber(hex:sub(3, 4), 16) / 255, tonumber(hex:sub(5, 6), 16) / 255
end

-- Gold used for frames in the classic style.
ns.GOLD_EDGE = { 0.86, 0.68, 0.26 }

-- Bottom and top colours of the panel gradient behind classic bars and the map.
function ns.PanelColors()
	if ns.GetTheme().key == "midnight" then
		return { 0.02, 0.02, 0.04 }, { 0.08, 0.09, 0.12 }
	end
	return { 0.05, 0.10, 0.17 }, { 0.14, 0.25, 0.38 }
end

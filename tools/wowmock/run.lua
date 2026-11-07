-- Headless tests for addons/ProfessionBars.
--   lua5.1 tools/wowmock/run.lua                 run every scenario
--   lua5.1 tools/wowmock/run.lua --dump out.json [main|panels|styles|themes|gather|ingame|ingamepanel]
--                                               write a preview scene
-- Run from the repository root.
local Mock = dofile("tools/wowmock/wow.lua")

local ADDON_DIR, ADDON = "addons/ProfessionBars", "ProfessionBars"
local failures, checks = 0, 0
local current = "?"

local function check(condition, message)
	checks = checks + 1
	if not condition then
		failures = failures + 1
		print("  FAIL [" .. current .. "] " .. message)
	end
	return condition
end

local function eq(actual, expected, message)
	return check(actual == expected, message .. " (expected " .. tostring(expected) .. ", got " .. tostring(actual) .. ")")
end

local function copy(value)
	if type(value) ~= "table" then
		return value
	end
	local out = {}
	for k, v in pairs(value) do
		out[k] = copy(v)
	end
	return out
end

---------------------------------------------------------------------------
-- Fake game data
---------------------------------------------------------------------------

-- slot: position in GetProfessions() (prof1, prof2, archaeology, fishing, cooking, firstAid)
local MODERN_PROFS = {
	{ slot = 1, name = "Blacksmithing", icon = 136241, rank = 45, max = 75, line = 164 },
	{ slot = 2, name = "Mining", icon = 136248, rank = 92, max = 150, line = 186, modifier = 0 },
	{ slot = 4, name = "Fishing", icon = 136245, rank = 1, max = 75, line = 356 },
	{ slot = 5, name = "Cooking", icon = 133971, rank = 120, max = 150, line = 185 },
}

-- First Aid only shows up through C_TradeSkillUI (not GetProfessions).
local CHILD_INFOS = {
	{ professionID = 2477, professionName = "Classic First Aid", parentProfessionID = 129, parentProfessionName = "First Aid", skillLevel = 75, maxSkillLevel = 75 },
	{ professionID = 2437, professionName = "Classic Blacksmithing", parentProfessionID = 164, parentProfessionName = "Blacksmithing", skillLevel = 45, maxSkillLevel = 75 },
}

local ITEMS = { copperBar = 2840, roughStone = 2835, coarseStone = 2836, lightLeather = 2318, wool = 2592, coarseThread = 2320, bronzeBar = 2841, weakFlux = 2880 }

-- Blacksmithing recipes: id, name, difficulty enum, category, reagents, learned.
local BS_RECIPES = {
	{ 2660, "Rough Sharpening Stone", 3, 1, { { ITEMS.roughStone, 1 } } },
	{ 3115, "Rough Weightstone", 3, 1, { { ITEMS.roughStone, 1 }, { ITEMS.wool, 1 } } },
	{ 2663, "Copper Bracers", 2, 2, { { ITEMS.copperBar, 2 } } },
	{ 2662, "Copper Chain Pants", 2, 2, { { ITEMS.copperBar, 4 } } },
	{ 3319, "Copper Chain Boots", 1, 2, { { ITEMS.copperBar, 8 } } },
	{ 2737, "Copper Mace", 1, 3, { { ITEMS.copperBar, 6 }, { ITEMS.weakFlux, 1 }, { ITEMS.lightLeather, 2 } } },
	{ 2738, "Copper Axe", 0, 3, { { ITEMS.copperBar, 6 }, { ITEMS.weakFlux, 1 }, { ITEMS.lightLeather, 2 } } },
	{ 3293, "Copper Battle Axe", 0, 3, { { ITEMS.copperBar, 12 }, { ITEMS.weakFlux, 2 } } },
	{ 3320, "Rough Grinding Stone", 0, 1, { { ITEMS.roughStone, 2 } } },
	{ 7408, "Heavy Copper Maul", 0, 3, { { ITEMS.copperBar, 12 }, { ITEMS.weakFlux, 2 }, { ITEMS.lightLeather, 2 } } },
	{ 2665, "Coarse Sharpening Stone", 0, 1, { { ITEMS.coarseStone, 1 } } },
	{ 9983, "Copper Claymore", 0, 3, { { ITEMS.copperBar, 10 }, { ITEMS.weakFlux, 2 } } },
	{ 3292, "Heavy Copper Broadsword", 0, 3, { { ITEMS.copperBar, 14 }, { ITEMS.weakFlux, 2 } }, false },
}
local CATEGORIES = { [1] = "Materials", [2] = "Armor", [3] = "Weapons" }
local ITEM_NAMES = {
	[ITEMS.copperBar] = "Copper Bar", [ITEMS.roughStone] = "Rough Stone", [ITEMS.lightLeather] = "Light Leather",
	[ITEMS.wool] = "Wool Cloth", [ITEMS.weakFlux] = "Weak Flux", [ITEMS.bronzeBar] = "Bronze Bar",
}

local function installModern(m, profs, childInfos)
	local env = m.env
	env.GetProfessions = function()
		local slots = {}
		for _, p in ipairs(profs) do
			slots[p.slot] = p.slot * 10
		end
		return slots[1], slots[2], slots[3], slots[4], slots[5], slots[6]
	end
	env.GetProfessionInfo = function(index)
		for _, p in ipairs(profs) do
			if p.slot * 10 == index then
				return p.name, p.icon, p.rank, p.max, 4, 0, p.line, p.modifier or 0, 0, 0
			end
		end
	end
	local trade = { open = nil }
	m.trade = trade
	env.C_TradeSkillUI = {
		GetChildProfessionInfos = function() return childInfos or {} end,
		GetProfessionInfoBySkillLineID = function() return { skillLevel = 0, maxSkillLevel = 0 } end,
		GetBaseProfessionInfo = function()
			if not trade.open then return { professionID = 0 } end
			return { professionID = trade.open.line, professionName = trade.open.name, skillLevel = trade.open.rank, maxSkillLevel = trade.open.max }
		end,
		IsTradeSkillLinked = function() return trade.linked or false end,
		IsTradeSkillGuild = function() return false end,
		IsNPCCrafting = function() return false end,
		GetAllRecipeIDs = function()
			local ids = {}
			for _, r in ipairs(trade.noAll and {} or trade.recipes or {}) do
				ids[#ids + 1] = r[1]
			end
			return ids
		end,
		GetFilteredRecipeIDs = function()
			local ids = {}
			for _, r in ipairs(trade.recipes or {}) do
				ids[#ids + 1] = r[1]
			end
			return ids
		end,
		GetRecipeInfo = function(id)
			for _, r in ipairs(trade.recipes or {}) do
				if r[1] == id then
					return { recipeID = id, name = r[2], relativeDifficulty = r[3], categoryID = r[4], learned = r[6] ~= false, icon = 134400 }
				end
			end
		end,
		GetRecipeSchematic = function(id)
			for _, r in ipairs(trade.recipes or {}) do
				if r[1] == id then
					local slots = {}
					for i, reagent in ipairs(r[5]) do
						slots[i] = { reagentType = 1, quantityRequired = reagent[2], reagents = { { itemID = reagent[1] } } }
					end
					return { name = r[2], reagentSlotSchematics = slots }
				end
			end
		end,
		GetCategoryInfo = function(id) return { name = CATEGORIES[id] } end,
		GetRecipeLink = function(id)
			for _, r in ipairs(trade.recipes or {}) do
				if r[1] == id then
					return "|cffffd000|Henchant:" .. id .. "|h[" .. r[2] .. "]|h|r"
				end
			end
		end,
	}
end

local function boot(m)
	local ns = m.LoadAddon(ADDON_DIR, ADDON)
	m.Fire("ADDON_LOADED", "SomeOtherAddon")
	m.Fire("ADDON_LOADED", ADDON)
	m.Fire("PLAYER_LOGIN")
	m.Advance(4)
	return ns
end

local function slash(m, text)
	m.env.SlashCmdList.PROFESSIONBARS(text)
	m.Advance(0.6)
end

local function shownBars(ns)
	local list = {}
	for _, prof in ipairs(ns.profs) do
		local bar = ns.Bars.bars[prof.key]
		if bar and bar:IsShown() then
			list[#list + 1] = bar
		end
	end
	table.sort(list, function(a, b) return a:GetTop() > b:GetTop() end)
	return list
end

local function plain(text)
	return (text or ""):gsub("|c%x%x%x%x%x%x%x%x", ""):gsub("|r", "")
end

---------------------------------------------------------------------------
-- Reading the panel
---------------------------------------------------------------------------

local function findRow(ns, label)
	for _, row in ipairs(ns.Detail.rows.active) do
		if plain(row.label:GetText()) == label then
			return row
		end
	end
end

local function rowValue(ns, label)
	local row = findRow(ns, label)
	return row and plain(row.value:GetText())
end

-- "label=value; label; ..." for every row of the open tab, top to bottom.
local function rowList(ns)
	local out = {}
	for i, row in ipairs(ns.Detail.rows.active) do
		local value = plain(row.value:GetText())
		out[i] = plain(row.label:GetText()) .. (value ~= "" and ("=" .. value) or "")
	end
	return table.concat(out, "; ")
end

-- "TITLE | right text; ..." for every section of the open tab.
local function titles(ns)
	local out = {}
	for i, title in ipairs(ns.Detail.titles.active) do
		local right = plain(title.right:GetText())
		out[i] = title.text:GetText() .. (right ~= "" and (" | " .. right) or "")
	end
	return table.concat(out, "; ")
end

local function tabKeys(ns)
	local keys = {}
	for i, tab in ipairs(ns.Detail.view.tabs) do
		keys[i] = tab.key
	end
	return table.concat(keys, ",")
end

local function nextText(ns)
	return ns.Detail.frame.nextLine.text:GetText()
end

local function clickTab(m, ns, key)
	for _, tab in ipairs(ns.Detail.tabs.active) do
		if tab.key == key then
			m.Click(tab)
			return
		end
	end
	check(false, "there is a " .. key .. " tab")
end

local function clickRow(m, ns, label)
	local row = findRow(ns, label)
	if check(row ~= nil, "there is a row '" .. label .. "'") then
		m.Click(row)
	end
end

-- Click every "Show all" row of the open tab.
local function showEverything(m, ns)
	for _ = 1, 10 do
		local more
		for _, row in ipairs(ns.Detail.rows.active) do
			if row.data and row.data.expand then
				more = row
			end
		end
		if not more then
			return
		end
		m.Click(more)
	end
end

local function tooltipText(m)
	return table.concat(m.env.GameTooltip.lines, "\n")
end

local function inside(a, b)
	return a.l >= b.l - 0.5 and a.r <= b.r + 0.5 and a.b >= b.b - 0.5 and a.t <= b.t + 0.5
end

local function overlaps(a, b)
	return a.l < b.r - 0.5 and b.l < a.r - 0.5 and a.b < b.t - 0.5 and b.b < a.t - 0.5
end

-- The open panel is on screen, everything in it fits inside it, nothing
-- overlaps and no text is cut off.
local function checkPanel(m, ns, label)
	local frame = ns.Detail.frame
	local panel = m.Resolve(frame)
	if not check(panel ~= nil and frame:IsShown(), label .. ": the panel is open and anchored") then
		return
	end
	check(panel.l >= -0.5 and panel.r <= m.W + 0.5 and panel.b >= -0.5 and panel.t <= m.H + 0.5, label .. ": the panel is on screen")
	local boxes = {}
	local function add(object, name)
		local rect = m.Resolve(object)
		if check(rect ~= nil, label .. ": " .. name .. " is anchored") then
			boxes[#boxes + 1] = { rect = rect, name = name }
		end
	end
	add(frame.icon, "the icon")
	add(frame.track, "the track")
	add(frame.nextLine, "the next line")
	for _, tab in ipairs(ns.Detail.tabs.active) do
		add(tab, "tab " .. tab.text:GetText())
	end
	for _, title in ipairs(ns.Detail.titles.active) do
		add(title, "title " .. title.text:GetText())
	end
	for _, row in ipairs(ns.Detail.rows.active) do
		add(row, "row '" .. plain(row.label:GetText()) .. "'")
	end
	if frame.footer:IsShown() then
		add(frame.footer, "the footer")
	end
	for i, a in ipairs(boxes) do
		check(inside(a.rect, panel), label .. ": " .. a.name .. " fits inside the panel")
		for j = i + 1, #boxes do
			check(not overlaps(a.rect, boxes[j].rect), label .. ": " .. a.name .. " and " .. boxes[j].name .. " do not overlap")
		end
	end
	-- Header: name and rank stay clear of the value and the close button.
	local rankName, value, close = m.Resolve(frame.rankName), m.Resolve(frame.value), m.Resolve(frame.close)
	check(rankName.r < value.l - 4 and value.r < close.l, label .. ": the header's name, value and close button do not collide")
	-- One-line text fits its space; wrapped text fits its row.
	for _, row in ipairs(ns.Detail.rows.active) do
		local text, rowRect = m.Resolve(row.label), m.Resolve(row)
		local name = plain(row.label:GetText())
		if row.data and row.data.wrap then
			check(text.b >= rowRect.b - 0.5, label .. ": wrapped text '" .. name .. "' fits its row")
		else
			check(row.label:GetUnboundedStringWidth() <= text.r - text.l + 0.5, label .. ": '" .. name .. "' is not cut off")
		end
	end
	local nextRect = m.Resolve(frame.nextLine.text)
	check(frame.nextLine.text:GetUnboundedStringWidth() <= nextRect.r - nextRect.l + 0.5, label .. ": the next step is not cut off")
end

---------------------------------------------------------------------------
-- Scenarios
---------------------------------------------------------------------------

local scenarios = {}

scenarios[#scenarios + 1] = { "mainline client", function()
	local m = Mock.new()
	m.level = 18
	installModern(m, MODERN_PROFS, CHILD_INFOS)
	m.bags = { [ITEMS.copperBar] = 26, [ITEMS.roughStone] = 7, [ITEMS.weakFlux] = 2, [ITEMS.lightLeather] = 3, [ITEMS.wool] = 1 }
	for id, name in pairs(ITEM_NAMES) do
		m.itemNames[id] = name
	end
	local ns = boot(m)
	local detail = ns.Detail
	check(m.ChatHas("click a bar for that profession's plan") ~= nil and ns.db.welcomed, "first login shows a one-time tip")
	local me = ns.db.alts["Tester-Mockrealm"]
	check(me and me.class == "PALADIN" and me.profs.blacksmithing[1] == 45, "this character is remembered account-wide")

	-- Bars
	local bars = shownBars(ns)
	eq(#bars, 5, "one bar per profession")
	local order = {}
	for i, bar in ipairs(bars) do
		order[i] = bar.key
	end
	eq(table.concat(order, ","), "blacksmithing,mining,cooking,firstaid,fishing", "primaries first, then secondaries")
	eq(plain(ns.Bars.bars.blacksmithing.value:GetText()), "45 / 75", "blacksmithing value text")
	eq(ns.Bars.bars.blacksmithing.value:GetText(), "45|cff8c8c96 / 75|r", "the max is dimmed")
	eq(ns.Bars.bars.firstaid.value:GetText(), "|cffffd10075 / 75|r", "first aid (capped, trainable) is gold")
	eq(ns.Bars.bars.firstaid.note:GetText(), "|cffffd100train|r", "a small gold 'train' appears when you can train")
	eq(ns.Bars.bars.blacksmithing.note:GetText(), "", "no note when there is nothing to act on")
	check(ns.Bars.bars.blacksmithing.tick:IsShown(), "training tick is shown")
	eq(math.floor(ns.Bars.bars.blacksmithing.status:GetValue() * 75 + 0.5), 45, "bar fill matches skill")

	-- The default clean style: icon, name and value above a thin line.
	local clean = ns.Bars.bars.blacksmithing
	eq(clean.style, "clean", "clean style is the default")
	local line, cleanName = m.Resolve(clean.status), m.Resolve(clean.name)
	eq(math.floor(line.t - line.b + 0.5), 4, "the bar is a 4px line")
	check(cleanName.b > line.t, "name sits above the line")
	check(math.abs(cleanName.l - line.l) < 0.5, "name and line share a left edge")
	check(not ns.Bars.frame.border[1]:IsShown() and ns.Bars.frame.bg:IsShown(), "flat panel without a border")
	local tick = m.Resolve(clean.tick)
	check(clean.tick:IsShown() and math.abs(tick.l - (line.l + (line.r - line.l) * 50 / 75)) < 1, "gold tick at 50 where Journeyman can be trained")
	check(not ns.Bars.bars.firstaid.tick:IsShown(), "no tick once the threshold is reached")

	-- The framed classic style: label row, framed bar, rank inside, marker.
	slash(m, "style classic")
	local smith = ns.Bars.bars.blacksmithing
	eq(smith.style, "classic", "/pb style classic")
	eq(smith.rank:GetText(), "Apprentice", "rank name sits inside the bar")
	eq(ns.Bars.bars.firstaid.rank:GetText(), "|cffffd100Ready for Journeyman|r", "a trainable rank says so inside the bar")
	eq(#smith.dividers, 7, "bar is split into eight segments")
	local fill = m.Resolve(smith.status)
	local marker = m.Resolve(smith.marker)
	local expected = fill.l + (fill.r - fill.l) * 45 / 75
	check(math.abs((marker.l + marker.r) / 2 - expected) < 1, "gold marker sits at the current skill")
	check(marker.t > fill.t and marker.b < fill.b, "marker reaches past the top and bottom of the bar")
	local nameRect, barFrame = m.Resolve(smith.name), m.Resolve(smith.frame)
	check(nameRect.b > barFrame.t, "name sits above the bar")
	check(ns.Bars.frame.bg.gradient ~= nil, "classic bars sit on a gradient panel")
	m.Click(smith)
	check(smith.selected and smith.frame.border[1].color[2] > 0.85, "selected classic bar gets a bright gold frame")
	check(detail.frame.bg.gradient ~= nil and detail.frame.border[1].color[1] == ns.GOLD_EDGE[1], "the panel matches the classic look")
	m.Click(smith)
	slash(m, "style clean")
	eq(ns.Bars.bars.blacksmithing.style, "clean", "/pb style clean")
	eq(table.concat(ns.scanSources, "+"), "GetProfessions+C_TradeSkillUI", "first aid came from C_TradeSkillUI")
	local r, g, b = unpack(ns.Bars.bars.blacksmithing.status.barColor)
	check(r > 0.8 and g < 0.6 and b < 0.3, "blacksmithing uses its ember colour")
	check(not ns.Bars.header:IsShown(), "header hidden until hover")

	-- Hover shows the header and a tooltip; leaving hides it again.
	m.Hover(ns.Bars.bars.blacksmithing)
	check(ns.Bars.header:IsShown(), "header appears on hover")
	check(m.env.GameTooltip.lines[1] == "Blacksmithing", "bar tooltip title")
	check(m.env.GameTooltip.lines[3]:find("5 more skill", 1, true) ~= nil, "bar tooltip says what is next")
	check(tooltipText(m):find("Click for the Blacksmithing plan", 1, true) ~= nil, "bar tooltip says what a click does")
	m.Leave()
	m.Advance(0.5)
	check(not ns.Bars.header:IsShown(), "header hides after the mouse leaves")

	-- The panel before any recipes are cached.
	m.Click(ns.Bars.bars.blacksmithing)
	check(detail.frame:IsShown(), "clicking a bar opens its panel")
	eq(detail.current, "blacksmithing", "the panel shows the clicked profession")
	eq(tabKeys(ns), "plan", "only the plan before any recipes are known")
	eq(detail.frame.name:GetText(), "Blacksmithing", "header: name")
	eq(detail.frame.rankName:GetText(), "Apprentice", "header: rank")
	eq(plain(detail.frame.value:GetText()), "45 / 75", "header: skill")
	eq(nextText(ns), "5 to 50, then train Journeyman", "one line says what to do next")
	m.Hover(detail.frame.nextLine)
	eq(m.env.GameTooltip.lines[1], "5 more skill, then learn Journeyman from a city trainer.", "the next line explains itself")
	m.Leave()
	check(findRow(ns, "Open your Blacksmithing window once to load your recipes.") ~= nil, "the plan explains how to load recipes")
	local br, bg, bb = ns.Color("blacksmithing")
	check(ns.Bars.bars.blacksmithing.selected, "the bar whose panel is open is marked selected")
	local accent = ns.Bars.bars.blacksmithing.accent
	check(accent:IsShown() and accent.color[1] == br and accent.color[2] == bg and accent.color[3] == bb, "selection is a thin line in the profession's colour")
	check(detail.frame.bg.gradient == nil and detail.frame.border[1].color[4] < 0.1, "the clean panel is flat with a hairline edge")
	check(detail.frame.accent.color[1] == br and detail.frame.accent.color[2] == bg, "the panel's top edge is in the profession's colour")
	-- The track covers 0-300: your skill, the part you can reach and a gold
	-- mark where Journeyman is learned.
	local inner = detail.metrics.INNER
	local trackRect = m.Resolve(detail.frame.track)
	local trackFill, trackAvail, goal = m.Resolve(detail.frame.track.fill), m.Resolve(detail.frame.track.avail), m.Resolve(detail.frame.track.goal)
	eq(math.floor(trackFill.r - trackFill.l + 0.5), math.floor(inner * 45 / 300 + 0.5), "the track fills to 45 of 300")
	eq(math.floor(trackAvail.r - trackAvail.l + 0.5), math.floor(inner * 75 / 300 + 0.5), "the reachable part ends at the cap")
	check(detail.frame.track.goal:IsShown() and math.abs(goal.l + 1 - (trackRect.l + math.floor(inner * 50 / 300 + 0.5))) < 0.5, "a gold mark at 50")
	m.Hover(detail.frame.track)
	check(tooltipText(m):find("Journeyman: up to 150", 1, true) ~= nil and tooltipText(m):find("Gold mark: where Journeyman can be learned (50)", 1, true) ~= nil, "the track's tooltip explains the ranks")
	m.Leave()
	checkPanel(m, ns, "blacksmithing without recipes")

	-- Open the Blacksmithing window: recipes get cached.
	m.trade.open, m.trade.recipes = MODERN_PROFS[1], BS_RECIPES
	m.Fire("TRADE_SKILL_SHOW")
	m.Fire("TRADE_SKILL_LIST_UPDATE")
	m.Advance(1)
	local cache = ns.GetRecipeCache("blacksmithing")
	eq(cache and #cache.list, 12, "learned recipes cached (unlearned skipped)")
	eq(cache.rank, 45, "cache remembers the skill it was scanned at")
	local byName = {}
	for _, recipe in ipairs(cache.list) do
		byName[recipe.name] = recipe
	end
	eq(byName["Copper Axe"].diff, "optimal", "difficulty enum mapped")
	eq(byName["Copper Axe"].cat, "Weapons", "category name read")
	eq(#byName["Copper Mace"].reagents, 3, "reagents read from the schematic")
	eq(ns.CraftableCount(byName["Copper Axe"]), 1, "craftable count limited by flux")
	eq(ns.CraftableCount(byName["Copper Bracers"]), 13, "craftable count from copper bars")

	-- The plan: crafts for the next points, from your bags first.
	eq(tabKeys(ns), "plan,recipes", "recipes add a Recipes tab")
	eq(detail.activeTab, "plan", "the plan comes first")
	eq(titles(ns), "CRAFT FOR SKILL | next 5 points; SHOPPING LIST | all in your bags", "plan sections")
	eq(rowList(ns), "3x Rough Grinding Stone=ready; 1x Copper Axe=ready; 2x Copper Chain Boots=ready; "
		.. "Light Leather=2/2; Rough Stone=6/6; Copper Bar=22/22; Weak Flux=1/1", "orange crafts your bags cover, then yellow ones")
	local boots = findRow(ns, "2x Copper Chain Boots")
	check(boots and boots.label.textColor[1] == 1 and boots.label.textColor[2] == 1 and boots.label.textColor[3] == 0, "yellow crafts are yellow")
	m.Hover(findRow(ns, "3x Rough Grinding Stone"))
	local tip = tooltipText(m)
	check(tip:find("[link]", 1, true) == 1 and tip:find("Orange: +1 skill every craft.", 1, true) and tip:find("You can make 3 from your bags.", 1, true),
		"plan rows show the recipe, why it is there and how many you can make")
	m.Leave()
	m.Hover(boots)
	check(tooltipText(m):find("Yellow: +1 skill about half the time.", 1, true) ~= nil, "yellow crafts say they are a coin flip")
	m.Leave()

	local plan = ns.PlanSkillUps(cache, 5)
	local planned, points = {}, 0
	for _, craft in ipairs(plan.crafts) do
		planned[#planned + 1] = craft.count .. "x " .. craft.recipe.name
		points = points + craft.points
	end
	eq(points, 5, "the plan covers the points needed, counting yellow as half")
	eq(table.concat(planned, ", "), "3x Rough Grinding Stone, 1x Copper Axe, 2x Copper Chain Boots", "plan order")
	local long = ns.PlanSkillUps(cache, 95)
	points = 0
	for _, craft in ipairs(long.crafts) do
		points = points + craft.points
	end
	check(long.horizon == 10 and points >= 10 and points < 11, "a long climb is planned ten points at a time (colours change as you level)")

	-- Out of rough stone: the rest is planned on the cheapest orange recipe
	-- and the shopping list says what to get.
	m.bags[ITEMS.roughStone] = nil
	m.Fire("BAG_UPDATE_DELAYED")
	m.Advance(1)
	eq(titles(ns), "CRAFT FOR SKILL | next 5 points; SHOPPING LIST | 1 to get", "plan says one item is missing")
	eq(rowList(ns), "1x Copper Axe=ready; 2x Copper Chain Boots=ready; 3x Coarse Sharpening Stone=need items; "
		.. "Item 2836=0/3; Light Leather=2/2; Copper Bar=22/22; Weak Flux=1/1", "missing reagents come first, with a placeholder name")
	check(m.itemRequests[ITEMS.coarseStone], "unknown item names are requested from the client")
	m.itemNames[ITEMS.coarseStone] = "Coarse Stone"
	m.Fire("ITEM_DATA_LOAD_RESULT", ITEMS.coarseStone, true)
	m.Advance(1)
	local coarse = findRow(ns, "Coarse Stone")
	check(coarse ~= nil and coarse.value:GetText() == "0/3" and coarse.value.textColor[1] == ns.BAD[1], "the item name arrives and the shortage is red")
	m.Hover(coarse)
	eq(m.env.GameTooltip.lines[1], "[item] " .. ITEMS.coarseStone, "shopping rows show the item tooltip")
	m.Leave()
	checkPanel(m, ns, "plan with a shopping list")
	m.bags[ITEMS.roughStone] = 7
	m.Fire("BAG_UPDATE_DELAYED")
	m.Advance(1)
	eq(titles(ns), "CRAFT FOR SKILL | next 5 points; SHOPPING LIST | all in your bags", "the plan updates as you loot")

	-- Recipes tab: orange first, the ones you can make most of on top.
	clickTab(m, ns, "recipes")
	eq(ns.db.tabs.blacksmithing, "recipes", "the open tab is remembered per profession")
	eq(titles(ns), "KNOWN RECIPES (12) | 6 orange  2 yellow  2 green", "recipe counts by colour")
	eq(rowList(ns), "Rough Grinding Stone=x3; Copper Axe=x1; Copper Battle Axe=x1; Copper Claymore=x1; Heavy Copper Maul=x1; "
		.. "Coarse Sharpening Stone; Copper Chain Boots=x3; Show all 12", "recipe order")
	clickRow(m, ns, "Show all 12")
	eq(#detail.rows.active, 12, "'Show all' lists every recipe")
	checkPanel(m, ns, "every recipe")
	local axe = findRow(ns, "Copper Axe")
	m.Hover(axe)
	check(m.env.GameTooltip.lines[1]:find("[link]", 1, true) == 1, "recipe rows show the recipe tooltip")
	m.Leave()
	m.shift = true
	m.Click(axe)
	m.shift = false
	eq(m.insertedLink, "|cffffd000|Henchant:2738|h[Copper Axe]|h|r", "shift-click links the recipe")

	-- Placement: beside the bars, on the side with room.
	local panelRect, barsRect = m.Resolve(detail.frame), m.Resolve(ns.Bars.frame)
	eq(math.floor(panelRect.l - barsRect.r + 0.5), 8, "the panel opens just right of the bars")
	eq(math.floor(panelRect.t - barsRect.t + 0.5), 0, "the panel lines up with the top of the bars")
	ns.db.point = { m.W - 240, 600 }
	ns.Bars:RestorePosition()
	detail:Refresh()
	panelRect, barsRect = m.Resolve(detail.frame), m.Resolve(ns.Bars.frame)
	eq(math.floor(barsRect.l - panelRect.r + 0.5), 8, "with the bars on the right, the panel opens on their left")
	ns.db.point = nil
	ns.Bars:RestorePosition()
	detail:Refresh()

	-- Other characters.
	ns.db.alts["Bob-Mockrealm"] = { name = "Bob", realm = "Mockrealm", class = "WARRIOR", className = "Warrior", level = 40,
		profs = { blacksmithing = { 150, 150, "Blacksmithing" }, mining = { 210, 225, "Mining" } } }
	detail:Refresh()
	eq(tabKeys(ns), "plan,recipes,alts", "an Alts tab appears once another character has it")
	clickTab(m, ns, "alts")
	local bob = findRow(ns, "Bob")
	check(bob ~= nil and bob.value:GetText() == "150 / 150", "alt and their skill are listed")
	check(bob ~= nil and bob.label.textColor[1] == 0.78, "alts use class colours")
	m.Hover(bob)
	eq(tooltipText(m), "Bob\nLevel 40 Warrior\nBob-Mockrealm", "alt tooltip")
	m.Leave()
	checkPanel(m, ns, "alts")
	m.Hover(ns.Bars.bars.blacksmithing)
	check(tooltipText(m):find("Also on: Bob 150/150", 1, true) ~= nil, "bar tooltip mentions alts")
	m.Leave()
	slash(m, "alts")
	check(m.ChatHas("Bob-Mockrealm: Blacksmithing 150/150, Mining 210/225") ~= nil, "/pb alts lists other characters")
	slash(m, "forget bob")
	check(ns.db.alts["Bob-Mockrealm"] == nil and m.ChatHas("forgot Bob-Mockrealm") ~= nil, "/pb forget removes one")
	check(ns.db.alts["Tester-Mockrealm"] ~= nil, "forget never removes the current character")
	eq(tabKeys(ns), "plan,recipes", "the Alts tab goes with them")
	eq(detail.activeTab, "plan", "and the panel falls back to the first tab")

	-- Bags change -> craftable counts update while the panel is open.
	clickTab(m, ns, "recipes")
	m.bags[ITEMS.weakFlux] = 0
	m.Fire("BAG_UPDATE_DELAYED")
	m.Advance(1)
	eq(rowValue(ns, "Copper Axe"), "", "craftable counts follow the bags")

	-- Switching profession: Mining's Where tab.
	m.Click(ns.Bars.bars.mining)
	eq(detail.current, "mining", "clicking another bar switches the panel")
	eq(tabKeys(ns), "where,nodes", "gathering tabs")
	eq(nextText(ns), "33 to 125, then train Expert", "mining's next step")
	eq(titles(ns), "BEST FOR YOU NOW | skill-up chance; RARE OR ONE ZONE; WHERE TO GO | you're level 18; NEXT UNLOCK", "where sections")
	eq(rowList(ns), "Tin Vein=usually; Copper Vein=sometimes; "
		.. "Lesser Bloodstone=Arathi only; Silver Vein=rare, in Tin spots; Incendicite Vein=Wetlands only; "
		.. "The Barrens=level 10-25; Redridge Mountains=level 15-25; Stonetalon Mountains=level 15-27; Show all 7; "
		.. "Iron Deposit=at 125 (+33)", "best nodes first; rare and one-zone nodes apart; zones that suit your level")
	local silver = findRow(ns, "Silver Vein")
	check(silver and silver.label.textColor[1] == 1 and silver.label.textColor[2] == 0.5, "silver vein is orange at 92 skill")
	m.Hover(silver)
	eq(tooltipText(m), "Silver Vein\nNeeds 75 skill.\nOrange for you: always a skill-up.\nRare, in Tin spots.", "node tooltip")
	m.Leave()
	checkPanel(m, ns, "mining, where")
	clickTab(m, ns, "nodes")
	eq(titles(ns), "EVERY NODE | skill needed", "every node, with the skill it needs")
	eq(rowList(ns), "Copper Vein=1; Tin Vein=65; Incendicite Vein=65 · Wetlands only; Silver Vein=75 · rare, in Tin spots; "
		.. "Lesser Bloodstone=75 · Arathi only; Iron Deposit=125; Indurium Vein=150 · Uldaman only; Show all 13", "node list")
	clickRow(m, ns, "Show all 13")
	checkPanel(m, ns, "mining, every node")
	m.Click(ns.Bars.bars.mining)
	check(not detail.frame:IsShown(), "clicking the open bar closes the panel")
	check(not ns.Bars.bars.mining.selected and not ns.Bars.bars.mining.accent:IsShown(), "selection cleared")

	-- Skill-up: bar animates and turns gold when Journeyman becomes trainable.
	MODERN_PROFS[1].rank = 50
	m.Fire("CHAT_MSG_SKILL", "Your skill in Blacksmithing has increased to 50.")
	m.Advance(0.1)
	m.Advance(0.3)
	local mid = ns.Bars.bars.blacksmithing.status:GetValue()
	check(mid > 45 / 75 and mid < 50 / 75, "bar fill animates toward the new value")
	m.Advance(1)
	eq(ns.Bars.bars.blacksmithing.value:GetText(), "|cffffd10050 / 75|r", "gold once Journeyman is trainable")
	eq(ns.Bars.bars.blacksmithing.note:GetText(), "|cff59ff73+5|r  |cffffd100train|r", "session gain and 'train' notes")
	eq(math.floor(ns.Bars.bars.blacksmithing.status:GetValue() * 75 + 0.5), 50, "bar fill settles on the new value")
	m.Click(ns.Bars.bars.blacksmithing)
	eq(detail.activeTab, "recipes", "the panel reopens on the tab you used last")
	clickTab(m, ns, "plan")
	eq(nextText(ns), "Learn Journeyman now", "the next step says to train now")
	eq(detail.frame.nextLine.text.textColor[2], ns.GOOD[2], "in green")
	check(plain(detail.frame.value:GetText()):find("+5", 1, true) == 1, "the header shows this session's gain")
	eq(detail.frame.footer:GetText(), "Recipe colours are from skill 45. Open Blacksmithing to refresh them.", "the footer warns the colours are stale")
	check(titles(ns):find("CRAFT FOR SKILL | next 10 of 25 points", 1, true) == 1, "once trainable, the plan aims for the cap, ten points at a time")
	checkPanel(m, ns, "blacksmithing at 50")
	m.Click(ns.Bars.bars.blacksmithing)

	-- Linked professions are never cached as ours.
	m.trade.linked = true
	m.trade.recipes = { { 1, "Someone Else's Recipe", 0, 1, {} } }
	m.Fire("TRADE_SKILL_LIST_UPDATE")
	m.Advance(1)
	eq(#ns.GetRecipeCache("blacksmithing").list, 12, "linked trade skills are ignored")
	m.trade.linked = false

	-- Slash commands.
	slash(m, "theme soft")
	eq(ns.db.theme, "soft", "/pb theme soft")
	local sr = ns.Bars.bars.blacksmithing.status.barColor[1]
	check(sr < r, "soft theme desaturates the bars")
	slash(m, "theme")
	eq(ns.db.theme, "midnight", "/pb theme cycles")
	slash(m, "theme nonsense")
	check(m.ChatHas("themes: vibrant, soft, midnight, mono") ~= nil, "/pb theme lists themes on a typo")
	slash(m, "theme mono")
	slash(m, "color mining #ff0000")
	eq(ns.Bars.bars.mining.status.barColor[1], 1, "/pb color sets a custom colour")
	eq(ns.Bars.bars.mining.status.barColor[2], 0, "/pb color (green channel)")
	slash(m, "colour mining reset")
	check(ns.db.colors.mining == nil, "/pb colour reset")
	slash(m, "color mining")
	check(m.ChatHas("usage: /pb color") ~= nil, "/pb color without a value prints usage")
	slash(m, "theme vibrant")

	local height = ns.Bars.frame:GetHeight()
	slash(m, "hide fishing")
	check(not ns.Bars.bars.fishing:IsShown(), "/pb hide fishing")
	check(ns.Bars.frame:GetHeight() < height, "bar stack shrinks")
	slash(m, "show fish")
	check(ns.Bars.bars.fishing:IsShown(), "/pb show with a partial name")
	slash(m, "hide nothing-like-this")
	check(m.ChatHas("no learned profession matches 'nothing-like-this'") ~= nil, "unknown profession message")

	slash(m, "")
	check(not ns.Bars.frame:IsShown(), "/pb toggles the bars off")
	slash(m, "")
	check(ns.Bars.frame:IsShown(), "/pb toggles the bars on")
	m.env.ProfessionBars_Toggle()
	check(not ns.Bars.frame:IsShown(), "key binding toggles")
	m.env.ProfessionBars_OnAddonCompartmentClick()
	check(ns.Bars.frame:IsShown(), "addon compartment toggles")

	slash(m, "lock")
	eq(ns.db.locked, true, "/pb lock")
	m.RunScript(ns.Bars.header, "OnDragStart")
	check(not ns.Bars.frame.moving, "locked bars do not move")
	slash(m, "unlock")
	m.RunScript(ns.Bars.header, "OnDragStart")
	m.dragDX, m.dragDY = 100, -50
	local leftBefore = ns.Bars.frame:GetLeft()
	m.RunScript(ns.Bars.header, "OnDragStop")
	m.dragDX, m.dragDY = nil, nil
	eq(math.floor(ns.Bars.frame:GetLeft() - leftBefore + 0.5), 100, "dragging moves the bars")
	check(ns.db.point ~= nil, "position saved")

	slash(m, "scale 1.25")
	eq(ns.Bars.frame:GetScale(), 1.25, "/pb scale")
	local left = ns.Bars.frame:GetLeft() * 1.25
	eq(math.floor(left - ns.db.point[1] + 0.5), 0, "scaling keeps the bars where they were")
	slash(m, "width 260")
	eq(ns.Bars.bars.cooking:GetWidth(), 260 - 20, "/pb width (clean bars sit inside the panel's padding)")

	-- The slim style, and back.
	slash(m, "style slim")
	eq(ns.db.style, "slim", "/pb style slim")
	local slim = ns.Bars.bars.blacksmithing
	check(slim.style == "slim" and slim:IsShown() and slim.rank == nil, "slim bars replace the framed ones")
	eq(slim:GetWidth(), 260, "slim bars fill the width")
	check(not ns.Bars.frame.bg:IsShown(), "slim bars have no panel")
	check(ns.Bars.frame:GetHeight() < 120, "slim stack is compact")
	m.Click(slim)
	local sr2, sg2, sb2 = ns.Color("blacksmithing")
	check(slim.border[1].color[1] == sr2 and slim.border[1].color[2] == sg2 and slim.border[1].color[3] == sb2, "selected slim bar gets a coloured border")
	check(detail.frame.bg.gradient == nil, "the panel uses its plain look with slim bars")
	checkPanel(m, ns, "beside scaled slim bars")
	m.Click(slim)
	slash(m, "style")
	eq(ns.db.style, "clean", "/pb style cycles clean -> classic -> slim -> clean")
	check(ns.Bars.bars.blacksmithing.style == "clean" and ns.Bars.bars.blacksmithing:IsShown(), "clean bars are back")
	slash(m, "style fancy")
	check(m.ChatHas("styles: clean, classic, slim") ~= nil, "unknown style lists the options")

	-- Idle fade: dim until hovered, back on hover, awake again after a skill-up.
	slash(m, "fade")
	m.Advance(0.5)
	check(math.abs(ns.Bars.frame:GetAlpha() - 0.3) < 0.01, "/pb fade dims the idle bars")
	m.Hover(ns.Bars.bars.cooking)
	m.Advance(0.5)
	check(math.abs(ns.Bars.frame:GetAlpha() - 1) < 0.01, "hovering brings them back")
	m.Leave()
	m.Advance(1)
	check(math.abs(ns.Bars.frame:GetAlpha() - 0.3) < 0.01, "they dim again after the mouse leaves")
	MODERN_PROFS[4].rank = 121
	m.Fire("CHAT_MSG_SKILL")
	m.Advance(1)
	check(math.abs(ns.Bars.frame:GetAlpha() - 1) < 0.01, "a skill-up wakes them")
	m.Advance(5)
	check(math.abs(ns.Bars.frame:GetAlpha() - 0.3) < 0.01, "and they dim again afterwards")
	MODERN_PROFS[4].rank = 120
	slash(m, "fade")
	m.Advance(0.5)
	check(math.abs(ns.Bars.frame:GetAlpha() - 1) < 0.01, "/pb fade again turns it off")
	slash(m, "width abc")
	check(m.ChatHas("usage: /pb width") ~= nil, "/pb width needs a number")
	slash(m, "panelscale 0.9")
	eq(ns.db.panelScale, 0.9, "/pb panelscale")

	slash(m, "combat")
	m.inCombat = true
	m.Fire("PLAYER_REGEN_DISABLED")
	check(not ns.Bars.frame:IsShown(), "hidden in combat when enabled")
	m.inCombat = false
	m.Fire("PLAYER_REGEN_ENABLED")
	check(ns.Bars.frame:IsShown(), "back after combat")
	slash(m, "combat")
	m.Fire("PLAYER_REGEN_DISABLED")
	check(ns.Bars.frame:IsShown(), "stays visible in combat when disabled")
	m.Fire("PLAYER_REGEN_ENABLED")

	-- Edit mode: hidden bars appear dimmed and can be toggled by clicking.
	slash(m, "hide cooking")
	slash(m, "edit")
	check(ns.Bars.editing and ns.Bars.bars.cooking:IsShown(), "edit mode shows hidden bars")
	eq(plain(ns.Bars.bars.cooking.value:GetText()), "hidden", "edit mode labels hidden bars")
	check(ns.Bars.header:IsShown(), "header stays visible while editing")
	m.Click(ns.Bars.bars.cooking)
	check(not ns.char.hidden.cooking, "clicking a bar in edit mode shows it")
	m.Click(ns.Bars.bars.fishing)
	check(ns.char.hidden.fishing, "clicking a bar in edit mode hides it")
	check(not detail.frame:IsShown(), "edit mode clicks do not open the panel")
	slash(m, "edit")
	check(not ns.Bars.editing and not ns.Bars.bars.fishing:IsShown(), "leaving edit mode hides chosen bars")
	slash(m, "show all")
	check(ns.Bars.bars.fishing:IsShown(), "/pb show all")

	-- "/pb <profession>" opens its panel; secondary professions get Ranks.
	slash(m, "mining")
	eq(detail.current, "mining", "/pb <profession> opens its panel")
	eq(detail.frame:GetScale(), 0.9, "the panel uses /pb panelscale")
	checkPanel(m, ns, "mining at 0.9")
	slash(m, "first aid")
	eq(detail.current, "firstaid", "/pb with a two-word name")
	eq(tabKeys(ns), "plan,ranks", "secondary professions get a Ranks tab")
	eq(nextText(ns), "Learn Journeyman now", "first aid at 75/75 can train")
	clickTab(m, ns, "ranks")
	eq(rowList(ns), "Apprentice=75 / 75; Journeyman=learn now; Expert=book at 125; Artisan=quest at 225", "how each rank is learned")
	m.Hover(findRow(ns, "Expert"))
	check(tooltipText(m):find("Learn it from the book Expert First Aid - Under Wraps at skill 125.", 1, true) ~= nil, "Expert First Aid is a book")
	m.Leave()
	m.Hover(findRow(ns, "Artisan"))
	check(tooltipText(m):find("Learn it from the quest Triage at skill 225, character level 35.", 1, true) ~= nil, "Artisan First Aid is a quest")
	m.Leave()
	checkPanel(m, ns, "first aid ranks")
	slash(m, "xyz")
	check(m.ChatHas("'xyz' isn't a command or a profession you have") ~= nil, "an unknown word gets a short answer")
	eq(detail.current, "firstaid", "and leaves the panel alone")
	local found = false
	for _, name in ipairs(m.env.UISpecialFrames) do
		found = found or name == "ProfessionBarsDetail"
	end
	check(found, "Escape closes the panel (UISpecialFrames)")
	detail.frame:Hide()
	check(detail.current == nil, "hiding the panel clears its state")

	-- A failing API is contained and reported once.
	local realLevel = m.env.UnitLevel
	m.env.UnitLevel = function() error("boom") end
	m.Fire("SKILL_LINES_CHANGED")
	m.Advance(1)
	m.Fire("SKILL_LINES_CHANGED")
	m.Advance(1)
	local reports = 0
	for _, text in ipairs(m.chat) do
		if text:find("something failed while updating professions", 1, true) then
			reports = reports + 1
		end
	end
	eq(reports, 1, "an error is reported once, not on every event")
	check(ns.lastError and ns.lastError:find("boom", 1, true) ~= nil, "the error is kept for /pb debug")
	m.env.UnitLevel = realLevel
	m.Fire("SKILL_LINES_CHANGED")
	m.Advance(1)
	check(ns.Bars.bars.blacksmithing:IsShown(), "bars recover once the API works again")
	slash(m, "debug")
	check(m.ChatHas("last error: updating professions") ~= nil, "/pb debug shows the last error")
	ns.lastError = nil

	slash(m, "debug")
	check(m.ChatHas("interface 16001") and m.ChatHas("read from: GetProfessions + C_TradeSkillUI") and m.ChatHas("12 recipes cached"), "/pb debug reports what it sees")
	slash(m, "help")
	check(m.ChatHas("/pb theme") ~= nil and m.ChatHas("/pb panelscale") ~= nil, "/pb help lists commands")
	slash(m, "reset")
	eq(ns.db.theme, "vibrant", "/pb reset restores the theme")
	eq(ns.db.scale, 1, "/pb reset restores the scale")
	eq(ns.db.panelScale, 1, "/pb reset restores the panel scale")
	check(next(ns.db.tabs) == nil, "/pb reset forgets the open tabs")
	check(ns.db.point == nil, "/pb reset forgets the position")
	check(ns.db.alts["Tester-Mockrealm"] ~= nil, "/pb reset keeps the alt data")
	check(#ns.GetRecipeCache("blacksmithing").list == 12, "/pb reset keeps the recipe cache")

	eq(#m.warnings, 0, "no conflicting anchors (" .. table.concat(m.warnings, "; ") .. ")")
	eq(ns.lastError, nil, "no contained errors")
end }

scenarios[#scenarios + 1] = { "classic-style skill lines only", function()
	local m = Mock.new()
	m.level = 40
	m.env.GetNumSkillLines = function() return 7 end
	local lines = {
		{ "Professions", true }, { "Tailoring", false, nil, 120, 0, 0, 150, 1 }, { "Enchanting", false, nil, 230, 0, 5, 300, 1 },
		{ "Secondary Skills", true }, { "First Aid", false, nil, 150, 0, 0, 150, 1 },
		{ "Weapon Skills", true }, { "Swords", false, nil, 200, 0, 0, 200, nil },
	}
	m.env.GetSkillLineInfo = function(i) return unpack(lines[i]) end
	local ns = boot(m)
	eq(#shownBars(ns), 3, "professions read from skill lines (weapon skills ignored)")
	eq(table.concat(ns.scanSources, "+"), "GetSkillLineInfo", "fallback source used")
	eq(plain(ns.Bars.bars.enchanting.value:GetText()), "230 / 300", "artisan value")
	m.Click(ns.Bars.bars.tailoring)
	eq(tabKeys(ns), "plan", "the panel opens without C_TradeSkillUI")
	check(findRow(ns, "This client doesn't share recipe data with addons.") ~= nil, "and says why there are no recipes")
	eq(nextText(ns), "5 to 125, then train Expert", "tailoring's next step")
	checkPanel(m, ns, "tailoring")
	m.Click(ns.Bars.bars.enchanting)
	eq(nextText(ns), "70 to 300", "artisan's next step is the grind to max")
	eq(ns.Detail.frame.rankName:GetText(), "Artisan", "artisan rank")
	m.Click(ns.Bars.bars.firstaid)
	eq(nextText(ns), "Learn Expert now", "first aid at 150/150 can learn Expert")
	eq(ns.Bars.bars.firstaid.note:GetText(), "|cffffd100book|r", "the bar says Expert First Aid is a book, not a trainer")
	m.Hover(ns.Detail.frame.nextLine)
	eq(m.env.GameTooltip.lines[1], "Learn Expert from the book Expert First Aid - Under Wraps. It raises your cap to 225.", "the next line names the book")
	m.Leave()
	checkPanel(m, ns, "first aid")
	eq(#m.warnings, 0, "no conflicting anchors")
	eq(ns.lastError, nil, "no contained errors")
end }

scenarios[#scenarios + 1] = { "no professions yet", function()
	local m = Mock.new()
	installModern(m, {}, {})
	local ns = boot(m)
	eq(#shownBars(ns), 0, "no bars")
	check(ns.Bars.empty:IsShown(), "empty state shown")
	check(ns.Bars.empty.text:GetText():find("No professions yet", 1, true) == 1, "empty state text")
	slash(m, "mining")
	check(m.ChatHas("'mining' isn't a command or a profession you have") ~= nil, "/pb mining with nothing learned")
	check(not ns.Detail.frame:IsShown(), "no panel to open")
	m.Hover(ns.Bars.empty)
	check(ns.Bars.header:IsShown(), "empty state still reveals the header")
	eq(#m.warnings, 0, "no conflicting anchors")
	eq(ns.lastError, nil, "no contained errors")
end }

scenarios[#scenarios + 1] = { "secret values and an unknown profession", function()
	local m = Mock.new()
	m.level = 60
	local profs = {
		{ slot = 1, name = "Jewelcrafting", icon = 134071, rank = 12, max = 75, line = 755 },
		{ slot = 2, name = "Herbalism", icon = 136246, rank = 300, max = 300, line = 182 },
		{ slot = 5, name = "Skinning", icon = 134366, rank = 45, max = 75, line = 393 },
	}
	installModern(m, profs, {})
	local secret = {}
	m.env.issecretvalue = function(v) return v == secret end
	local original = m.env.GetProfessionInfo
	m.env.GetProfessionInfo = function(index)
		local name, icon, rank, max, a, b, line, modifier = original(index)
		if name == "Skinning" then
			modifier = secret
		end
		return name, icon, rank, max, a, b, line, modifier
	end
	local ns = boot(m)
	eq(#shownBars(ns), 3, "unknown professions still get a bar")
	eq(ns.Bars.bars["other:jewelcrafting"].prof.icon, 134071, "unknown profession keeps the game's icon")
	eq(ns.Bars.bars.skinning.prof.modifier, 0, "secret values are treated as missing")
	m.Click(ns.Bars.bars["other:jewelcrafting"])
	eq(tabKeys(ns), "plan", "an unknown profession is treated like a craft")
	check(findRow(ns, "Open your Jewelcrafting window once to load your recipes.") ~= nil, "its plan can still load recipes")
	checkPanel(m, ns, "unknown profession")

	m.Click(ns.Bars.bars.herbalism)
	eq(nextText(ns), "Mastered", "mastered profession")
	eq(tabKeys(ns), "where,nodes", "herbalism tabs")
	eq(rowList(ns), "Maxed out, so nothing gives skill any more.; Black Lotus=rare; Icecap=Winterspring only; "
		.. "Felwood=level 48-55; Un'Goro Crater=level 48-55; Western Plaguelands=level 51-58; Show all 5",
		"a maxed herbalist sees no skill-up chances, just rare herbs and where to farm")
	checkPanel(m, ns, "herbalism")

	m.Click(ns.Bars.bars.skinning)
	eq(tabKeys(ns), "where", "skinning tabs")
	eq(nextText(ns), "5 to 50, then train Journeyman", "skinning's next step")
	eq(rowList(ns), "Level 13-14 beasts=always; Level 1-12 beasts=usually; "
		.. "Darkshore=level 10-20; Loch Modan=level 10-20; Silverpine Forest=level 10-20; Show all 5; "
		.. "Level 15 beasts=at 50 (+5)", "skinning: beast levels, zones for them, next unlock")
	checkPanel(m, ns, "skinning")
	eq(#m.warnings, 0, "no conflicting anchors")
	eq(ns.lastError, nil, "no contained errors")
end }

scenarios[#scenarios + 1] = { "a rank that waits on character level", function()
	local m = Mock.new()
	m.level = 18
	local profs = {
		{ slot = 2, name = "Mining", icon = 136248, rank = 125, max = 150, line = 186 },
		{ slot = 5, name = "Cooking", icon = 133971, rank = 200, max = 225, line = 185 },
	}
	installModern(m, profs, {})
	local ns = boot(m)
	eq(ns.Bars.bars.mining.note:GetText(), "|cffff7766lvl 20|r", "the bar says which level Expert needs")
	m.Click(ns.Bars.bars.mining)
	eq(nextText(ns), "Reach level 20 for Expert", "the panel says it too")
	eq(ns.Detail.frame.nextLine.text.textColor[1], ns.BAD[1], "in red")
	eq(rowValue(ns, "Iron Deposit"), "always", "iron gives skill every time at 125")
	checkPanel(m, ns, "mining waiting on level")
	slash(m, "style classic")
	eq(ns.Bars.bars.mining.rank:GetText(), "|cffff7766Expert at level 20|r", "classic bars say it inside the bar")
	slash(m, "style clean")

	-- Cooking's Artisan rank is a quest, not a trainer.
	slash(m, "cooking")
	eq(nextText(ns), "25 to 225, then do the Artisan quest", "secondary professions are not sent to a trainer")
	clickTab(m, ns, "ranks")
	eq(rowList(ns), "Apprentice=done; Journeyman=done; Expert=200 / 225; Artisan=quest at 225", "cooking ranks")
	profs[2].rank, profs[2].max = 225, 225
	m.Fire("SKILL_LINES_CHANGED")
	m.Advance(1)
	eq(nextText(ns), "Reach level 35 for Artisan", "the Artisan quest needs level 35")
	eq(rowValue(ns, "Artisan"), "at level 35", "and the Ranks tab says so")
	-- Cooking at 125/150: Expert comes from a book.
	profs[2].rank, profs[2].max = 125, 150
	m.Fire("SKILL_LINES_CHANGED")
	m.Advance(1)
	eq(ns.Bars.bars.cooking.note:GetText(), "|cffffd100book|r", "the bar says Expert Cooking is a book")
	eq(nextText(ns), "Learn Expert now", "the panel says it can be learned now")
	eq(rowValue(ns, "Expert"), "learn now", "the Ranks tab too")
	m.Hover(ns.Detail.frame.nextLine)
	eq(m.env.GameTooltip.lines[1], "Learn Expert from the book Expert Cookbook. It raises your cap to 225.", "the next line names the book")
	m.Leave()
	checkPanel(m, ns, "cooking ranks")
	eq(#m.warnings, 0, "no conflicting anchors")
	eq(ns.lastError, nil, "no contained errors")
end }

scenarios[#scenarios + 1] = { "session gains survive a /reload", function()
	local profs = { { slot = 1, name = "Blacksmithing", icon = 136241, rank = 40, max = 75, line = 164 } }
	local function start(saved, now)
		local m = Mock.new()
		m.level = 18
		m.now = now or 0
		installModern(m, profs, {})
		if saved then
			m.env.ProfessionBarsDB, m.env.ProfessionBarsCharDB = copy(saved.db), copy(saved.char)
		end
		return m, boot(m)
	end
	local function logout(m)
		m.Fire("PLAYER_LOGOUT")
		return { db = copy(m.env.ProfessionBarsDB), char = copy(m.env.ProfessionBarsCharDB), at = m.now }
	end
	local m, ns = start({ db = { mapScale = 0.8, expanded = { mining = { ore = true } } }, char = {} })
	check(ns.db.mapScale == nil and ns.db.expanded == nil, "settings left over from the mind map are cleared")
	profs[1].rank = 44
	m.Fire("CHAT_MSG_SKILL")
	m.Advance(1)
	eq(ns.SessionGain(ns.byKey.blacksmithing), 4, "+4 this session")
	local saved = logout(m)
	m, ns = start(saved, saved.at + 5)
	eq(ns.SessionGain(ns.byKey.blacksmithing), 4, "a /reload keeps the session's gains")
	eq(plain(ns.Bars.bars.blacksmithing.note:GetText()), "+4", "and the bar still shows them")
	saved = logout(m)
	m, ns = start(saved, saved.at + 700)
	eq(ns.SessionGain(ns.byKey.blacksmithing), 0, "logging back in after a break starts a new session")
	eq(#m.warnings, 0, "no conflicting anchors")
	eq(ns.lastError, nil, "no contained errors")
end }

scenarios[#scenarios + 1] = { "layout stress: every tab, everything shown", function()
	local m = Mock.new()
	m.level = 30
	local profs = { { slot = 2, name = "Mining", icon = 136248, rank = 140, max = 150, line = 186 } }
	installModern(m, profs, {})
	local ns = boot(m)
	-- Smelting recipes give Mining a third tab.
	local recipes = {}
	for i = 1, 14 do
		recipes[i] = { 7000 + i, "Smelt Test Bar " .. i, i % 4, (i % 3) + 1, { { 2770, 1 } } }
	end
	m.bags[2770] = 40
	m.trade.open, m.trade.recipes = profs[1], recipes
	m.Fire("TRADE_SKILL_LIST_UPDATE")
	m.Advance(1)
	slash(m, "mining")
	eq(tabKeys(ns), "where,nodes,plan", "smelting recipes add a tab")
	eq(ns.Detail.view.tabs[3].label, "Smelting", "mining calls its plan Smelting")
	for _, key in ipairs({ "where", "nodes", "plan" }) do
		clickTab(m, ns, key)
		showEverything(m, ns)
		checkPanel(m, ns, key .. ", everything shown")
	end
	eq(rowList(ns):match("^[^;]+"), "10x Smelt Test Bar 12=ready", "smelting plan")
	-- A short screen scales the panel down to fit.
	local small = Mock.new({ width = 1024, height = 300 })
	small.level = 30
	installModern(small, profs, {})
	local ns2 = boot(small)
	slash(small, "mining")
	clickTab(small, ns2, "nodes")
	showEverything(small, ns2)
	check(ns2.Detail.frame:GetScale() < 1, "the panel scales down on a short screen")
	checkPanel(small, ns2, "short screen")
	eq(#m.warnings + #small.warnings, 0, "no conflicting anchors")
	eq(ns.lastError, nil, "no contained errors")
end }

scenarios[#scenarios + 1] = { "camp recipes, long names and secondary ranks", function()
	local m = Mock.new()
	m.level = 30
	local profs = {
		{ slot = 1, name = "Enchanting", icon = 136244, rank = 90, max = 150, line = 333 },
		{ slot = 4, name = "Fishing", icon = 136245, rank = 40, max = 75, line = 356 },
		{ slot = 5, name = "Cooking", icon = 133971, rank = 60, max = 75, line = 185 },
	}
	installModern(m, profs, {})
	local ns = boot(m)
	eq(ns.ShortRecipeName("Enchant Bracer - Minor Stamina"), "Bracer - Minor Stamina", "long enchant names are shortened")
	m.trade.open, m.trade.recipes = profs[1], { { 7457, "Enchant Bracer - Minor Stamina", 0, 1, { { 10940, 3 } } } }
	m.Fire("TRADE_SKILL_LIST_UPDATE")
	m.Advance(1)
	m.trade.open = profs[3]
	m.trade.recipes = { { 818, "Basic Campfire", 1, 1, { { 4470, 1 } } }, { 2538, "Charred Wolf Meat", 0, 1, { { 2672, 1 } } } }
	m.Fire("TRADE_SKILL_LIST_UPDATE")
	m.Advance(1)

	slash(m, "cooking")
	eq(tabKeys(ns), "plan,recipes,camp,ranks", "camp recipes get their own tab")
	eq(nextText(ns), "Learn Journeyman now", "cooking at 60 can train")
	eq(titles(ns), "CRAFT FOR SKILL | next 10 of 15 points; SHOPPING LIST | 1 to get", "a trainable rank plans toward the cap, ten points at a time")
	eq(rowList(ns), "10x Charred Wolf Meat=need items; Item 2672=0/10", "nothing in the bags: plan the best recipe")
	clickTab(m, ns, "camp")
	eq(rowList(ns), "Basic Campfire", "camp tab")
	checkPanel(m, ns, "cooking camp")

	slash(m, "enchanting")
	eq(nextText(ns), "35 to 125, then train Expert", "enchanting's next step")
	eq(rowList(ns), "10x Bracer - Minor Stamina=need items; Item 10940=0/30", "never a 35x plan: ten points at a time, short names")
	clickTab(m, ns, "recipes")
	check(findRow(ns, "Bracer - Minor Stamina") ~= nil, "recipe rows use the short name")
	checkPanel(m, ns, "enchanting")

	slash(m, "fishing")
	eq(tabKeys(ns), "ranks", "fishing shows its ranks")
	eq(nextText(ns), "10 to 50, then train Journeyman", "fishing's next step")
	eq(rowList(ns), "Apprentice=40 / 75; Journeyman=trainer at 50; Expert=book at 125; Artisan=quest at 225", "fishing ranks")
	m.Hover(findRow(ns, "Artisan"))
	check(tooltipText(m):find("Learn it from the quest Nat Pagle, Angler Extreme at skill 225, character level 35.", 1, true) ~= nil, "Artisan Fishing is a quest")
	m.Leave()
	checkPanel(m, ns, "fishing")
	eq(#m.warnings, 0, "no conflicting anchors")
	eq(ns.lastError, nil, "no contained errors")
end }

scenarios[#scenarios + 1] = { "recipe cache follows what you know", function()
	local m = Mock.new()
	m.level = 20
	local profs = {
		{ slot = 1, name = "Blacksmithing", icon = 136241, rank = 60, max = 75, line = 164 },
		{ slot = 2, name = "Mining", icon = 136248, rank = 40, max = 75, line = 186 },
	}
	installModern(m, profs, {})
	local ns = boot(m)
	local function cachedNames()
		local names = {}
		for _, recipe in ipairs(ns.GetRecipeCache("blacksmithing").list) do
			names[#names + 1] = recipe.name
		end
		table.sort(names)
		return table.concat(names, ",")
	end
	local function scan(recipes)
		m.trade.open, m.trade.recipes = profs[1], recipes
		m.Fire("TRADE_SKILL_LIST_UPDATE")
		m.Advance(1)
	end
	scan({ { 1, "A", 0, 1, { { 2840, 1 } } }, { 2, "B", 0, 1, { { 2840, 1 } } }, { 3, "C", 0, 1, { { 2840, 1 } } } })
	eq(cachedNames(), "A,B,C", "learned recipes cached")

	-- Dropped and relearned: A is unlearned now and B is no longer listed.
	scan({ { 1, "A", 0, 1, { { 2840, 1 } }, false }, { 3, "C", 0, 1, { { 2840, 1 } } } })
	eq(cachedNames(), "C", "unlearned and missing recipes are forgotten")

	-- A filtered list hides recipes without unlearning them: keep those.
	m.trade.noAll = true
	scan({ { 4, "D", 0, 1, { { 2840, 1 } } } })
	eq(cachedNames(), "C,D", "recipes hidden by the window's filters are kept")
	scan({ { 3, "C", 0, 1, { { 2840, 1 } }, false } })
	eq(cachedNames(), "D", "even a filtered list forgets recipes reported as unlearned")
	m.trade.noAll = false

	-- No answers yet (data still loading) never wipes the cache.
	local realInfo = m.env.C_TradeSkillUI.GetRecipeInfo
	m.env.C_TradeSkillUI.GetRecipeInfo = function() return nil end
	scan({ { 4, "D", 0, 1, { { 2840, 1 } } } })
	eq(cachedNames(), "D", "a scan with no answers keeps the cache")
	m.env.C_TradeSkillUI.GetRecipeInfo = realInfo

	-- A miner at 40 is not shown rare or one-zone veins they cannot mine yet.
	m.Click(ns.Bars.bars.mining)
	eq(titles(ns), "BEST FOR YOU NOW | skill-up chance; WHERE TO GO | you're level 20; NEXT UNLOCK", "no rare section until one is in reach")
	eq(rowValue(ns, "Tin Vein"), "at 65 (+25)", "the next unlock is the next common vein")
	check(findRow(ns, "Incendicite Vein") == nil, "locked one-zone veins are not suggested")
	eq(#m.warnings, 0, "no conflicting anchors")
	eq(ns.lastError, nil, "no contained errors")
end }

---------------------------------------------------------------------------
-- Preview scenes for tools/wowmock/preview.js
---------------------------------------------------------------------------

local function sceneMock(theme, style)
	local m = Mock.new()
	m.level = 18
	MODERN_PROFS[1].rank = 43
	installModern(m, MODERN_PROFS, CHILD_INFOS)
	m.bags = { [ITEMS.copperBar] = 26, [ITEMS.roughStone] = 3, [ITEMS.weakFlux] = 2, [ITEMS.lightLeather] = 3, [ITEMS.wool] = 1 }
	for id, name in pairs(ITEM_NAMES) do
		m.itemNames[id] = name
	end
	m.itemNames[ITEMS.coarseStone] = "Coarse Stone"
	local ns = boot(m)
	ns.db.alts["Bob-Mockrealm"] = { name = "Bob", class = "WARRIOR", className = "Warrior", level = 40,
		profs = { blacksmithing = { 150, 150, "Blacksmithing" }, mining = { 210, 225, "Mining" } } }
	ns.db.alts["Ivy-Mockrealm"] = { name = "Ivy", class = "MAGE", className = "Mage", level = 27,
		profs = { blacksmithing = { 92, 150, "Blacksmithing" } } }
	MODERN_PROFS[1].rank = 45
	m.Fire("CHAT_MSG_SKILL")
	m.Advance(1)
	m.trade.open, m.trade.recipes = MODERN_PROFS[1], BS_RECIPES
	m.Fire("TRADE_SKILL_LIST_UPDATE")
	m.Advance(1)
	if theme then
		slash(m, "theme " .. theme)
	end
	if style then
		slash(m, "style " .. style)
	end
	return m, ns
end

local function label(text, x, y, size)
	return { type = "text", rect = { l = x, r = x + 300, b = y, t = y + size + 4 }, text = text, size = size,
		color = { 1, 1, 1, 0.85 }, justifyH = "LEFT", anchorH = "LEFT", fixedWidth = true, alpha = 1 }
end

local function append(ops, more)
	for _, op in ipairs(more) do
		ops[#ops + 1] = op
	end
end

local SCENES = {}

-- Bars plus the Blacksmithing panel, as in the README.
function SCENES.main()
	local m, ns = sceneMock()
	ns.db.point = { 40, 600 }
	ns.Bars:RestorePosition()
	m.Click(ns.Bars.bars.blacksmithing)
	m.Advance(1)
	m.Hover(ns.Bars.bars.blacksmithing)
	m.mouseOver = nil
	m.env.GameTooltip:Hide()
	return m, m.Dump()
end

-- Three panels side by side: a craft's plan, a gatherer's Where tab and a
-- secondary profession's ranks.
function SCENES.panels()
	local ops, m0 = {}, nil
	local views = {
		{ "blacksmithing", "plan", "Blacksmithing: Plan" },
		{ "mining", "where", "Mining: Where" },
		{ "cooking", "ranks", "Cooking: Ranks" },
	}
	for i, view in ipairs(views) do
		local m, ns = sceneMock()
		m0 = m0 or m
		ns.db.tabs[view[1]] = view[2]
		slash(m, view[1])
		ns.Bars.frame:Hide()
		local x = 30 + (i - 1) * 350
		ns.Detail.frame:ClearAllPoints()
		ns.Detail.frame:SetPoint("TOPLEFT", m.UIParent, "BOTTOMLEFT", x, 700)
		ops[#ops + 1] = label(view[3], x, 708, 12)
		append(ops, m.Dump())
	end
	return m0, ops
end

-- The same bars in every theme, side by side.
function SCENES.themes()
	local ops, m0 = {}, nil
	for i, theme in ipairs({ "vibrant", "soft", "midnight", "mono" }) do
		local m, ns = sceneMock(theme)
		m0 = m0 or m
		local x = 30 + (i - 1) * 250
		ns.db.point = { x, 600 }
		ns.Bars:RestorePosition()
		ops[#ops + 1] = label(ns.GetTheme().label, x, 606, 12)
		append(ops, m.Dump())
	end
	return m0, ops
end

-- The three bar styles side by side.
function SCENES.styles()
	local ops, m0 = {}, nil
	local styles = { { "clean", "Clean (default)" }, { "classic", "Classic" }, { "slim", "Slim" } }
	for i, entry in ipairs(styles) do
		local m, ns = sceneMock(nil, entry[1])
		m0 = m0 or m
		local x = 30 + (i - 1) * 270
		ns.db.point = { x, 600 }
		ns.Bars:RestorePosition()
		ops[#ops + 1] = label(entry[2], x, 606, 12)
		append(ops, m.Dump())
	end
	return m0, ops
end

-- Mining's Where tab in the Midnight theme.
function SCENES.gather()
	local m, ns = sceneMock("midnight")
	ns.db.point = { 40, 560 }
	ns.Bars:RestorePosition()
	m.Click(ns.Bars.bars.mining)
	m.Advance(1)
	return m, m.Dump()
end

-- Full-screen scenes at the default positions, for "in game" mock-ups
-- (render with: node tools/wowmock/preview.js scene.json out.png game).
function SCENES.ingame()
	local m = sceneMock()
	m.Advance(1)
	return m, m.Dump()
end

function SCENES.ingamepanel()
	local m, ns = sceneMock()
	m.Hover(ns.Bars.bars.blacksmithing)
	m.Click(ns.Bars.bars.blacksmithing)
	m.Advance(1)
	m.env.GameTooltip:Hide()
	return m, m.Dump()
end

local function dumpScene(path, name)
	local build = assert(SCENES[name or "main"], "unknown scene " .. tostring(name))
	local m, ops = build()
	local file = assert(io.open(path, "w"))
	file:write(Mock.ToJSON({ width = m.W, height = m.H, ops = ops }))
	file:close()
	print("wrote " .. path)
end

if arg[1] == "--dump" then
	dumpScene(assert(arg[2], "--dump needs a path"), arg[3])
	return
end

for _, scenario in ipairs(scenarios) do
	current = scenario[1]
	local ok, err = xpcall(scenario[2], debug.traceback)
	if not ok then
		failures = failures + 1
		print("  ERROR [" .. current .. "] " .. tostring(err))
	end
	print((ok and "ok   " or "fail ") .. current)
end
print(string.format("%d checks, %d failures", checks, failures))
os.exit(failures == 0 and 0 or 1)

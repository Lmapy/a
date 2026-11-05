-- Headless tests for addons/ProfessionBars.
--   lua5.1 tools/wowmock/run.lua                 run every scenario
--   lua5.1 tools/wowmock/run.lua --dump out.json [main|themes|gather]
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

-- Visible map nodes as screen rectangles (no overlaps allowed).
local function mapNodes(m, ns)
	local nodes = {}
	for _, node in ipairs(ns.Map.nodes.active) do
		nodes[#nodes + 1] = { node = node, rect = m.Resolve(node) }
	end
	nodes[#nodes + 1] = { node = ns.Map.center, rect = m.Resolve(ns.Map.center) }
	return nodes
end

local function checkMapGeometry(m, ns, label)
	local nodes = mapNodes(m, ns)
	local panel = m.Resolve(ns.Map.frame)
	for i, a in ipairs(nodes) do
		check(a.rect ~= nil, label .. ": node " .. i .. " is anchored")
		if a.rect then
			check(a.rect.l >= panel.l - 0.5 and a.rect.r <= panel.r + 0.5 and a.rect.b >= panel.b - 0.5 and a.rect.t <= panel.t + 0.5,
				label .. ": node '" .. tostring(a.node.data and a.node.data.label or "centre") .. "' fits inside the panel")
			for j = i + 1, #nodes do
				local b = nodes[j]
				if b.rect then
					local overlap = a.rect.l < b.rect.r - 0.5 and b.rect.l < a.rect.r - 0.5 and a.rect.b < b.rect.t - 0.5 and b.rect.b < a.rect.t - 0.5
					check(not overlap, label .. ": nodes '" .. tostring(a.node.data and a.node.data.label or "centre") .. "' and '" .. tostring(b.node.data and b.node.data.label) .. "' do not overlap")
				end
			end
		end
	end
	local onScreen = panel.l >= -0.5 and panel.r <= m.W + 0.5 and panel.b >= -0.5 and panel.t <= m.H + 0.5
	check(onScreen, label .. ": panel is on screen")
end

local function findNode(ns, label)
	for _, node in ipairs(ns.Map.nodes.active) do
		if node.data and plain(node.data.label) == label then
			return node
		end
	end
end

local function branchKeys(ns)
	local keys = {}
	for _, branch in ipairs(ns.Map.tree.branches) do
		keys[#keys + 1] = branch.key
	end
	return table.concat(keys, ",")
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
	local ns = boot(m)

	-- Bars
	local bars = shownBars(ns)
	eq(#bars, 5, "one bar per profession")
	local order = {}
	for i, bar in ipairs(bars) do
		order[i] = bar.key
	end
	eq(table.concat(order, ","), "blacksmithing,mining,cooking,firstaid,fishing", "primaries first, then secondaries")
	eq(plain(ns.Bars.bars.blacksmithing.value:GetText()), "45/75", "blacksmithing value text")
	eq(ns.Bars.bars.firstaid.value:GetText(), "|cffffd10075/75|r", "first aid (capped, trainable) is gold")
	check(ns.Bars.bars.blacksmithing.tick:IsShown(), "training tick is shown")
	eq(math.floor(ns.Bars.bars.blacksmithing.status:GetValue() * 75 + 0.5), 45, "bar fill matches skill")
	eq(table.concat(ns.scanSources, "+"), "GetProfessions+C_TradeSkillUI", "first aid came from C_TradeSkillUI")
	local r, g, b = unpack(ns.Bars.bars.blacksmithing.status.barColor)
	check(r > 0.8 and g < 0.6 and b < 0.3, "blacksmithing uses its ember colour")
	check(not ns.Bars.header:IsShown(), "header hidden until hover")

	-- Hover shows the header and a tooltip; leaving hides it again.
	m.Hover(ns.Bars.bars.blacksmithing)
	check(ns.Bars.header:IsShown(), "header appears on hover")
	check(m.env.GameTooltip.lines[1] == "Blacksmithing", "bar tooltip title")
	check(m.env.GameTooltip.lines[3]:find("5 more skill", 1, true) ~= nil, "bar tooltip says what is next")
	m.Leave()
	m.Advance(0.5)
	check(not ns.Bars.header:IsShown(), "header hides after the mouse leaves")

	-- Map before any recipes are cached.
	m.Click(ns.Bars.bars.blacksmithing)
	check(ns.Map.frame:IsShown(), "clicking a bar opens the map")
	eq(ns.Map.current, "blacksmithing", "map shows the clicked profession")
	eq(branchKeys(ns), "rank,next,skillups", "branches without recipe data")
	check(findNode(ns, "Open your Blacksmithing window") ~= nil, "map explains how to load recipes")
	local br, bg, bb = ns.Color("blacksmithing")
	check(ns.Bars.bars.blacksmithing.border[1].color[1] == br and ns.Bars.bars.blacksmithing.border[1].color[2] == bg and ns.Bars.bars.blacksmithing.border[1].color[3] == bb, "selected bar gets a coloured border")

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

	eq(branchKeys(ns), "rank,next,skillups,craft,recipes", "branches with recipe data")
	checkMapGeometry(m, ns, "blacksmithing default")
	local skillups = findNode(ns, "Skill-ups")
	check(skillups and skillups.data.open, "skill-ups branch opens by default")
	eq(skillups.sign, -1, "content branches sit on the left")
	eq(plain(skillups.value:GetText()), "6 · 2 · 2", "skill-up counts by colour")
	local axe = findNode(ns, "Copper Axe")
	check(axe ~= nil and axe.value:GetText() == "x1", "orange recipe leaf with craftable count")
	check(findNode(ns, "+3 more") ~= nil, "long branches end with '+N more'")

	-- Unfold "Ready to craft" and check the layout again.
	local craft = findNode(ns, "Ready to craft")
	m.Click(craft)
	m.Advance(0.5)
	check(ns.db.expanded.blacksmithing.craft == true, "unfolding is remembered")
	local grinding = findNode(ns, "Rough Grinding Stone")
	check(grinding ~= nil and grinding.value:GetText() == "x3", "ready-to-craft leaves shown with counts")
	check(findNode(ns, "+6 more") ~= nil, "ready-to-craft lists skill-up recipes first, the rest under '+N more'")
	checkMapGeometry(m, ns, "blacksmithing with craft open")

	-- Centre click folds everything, second click unfolds everything.
	m.Click(ns.Map.center)
	check(findNode(ns, "Rough Grinding Stone") == nil and findNode(ns, "Copper Axe") == nil, "centre click folds every branch")
	m.Click(ns.Map.center)
	check(findNode(ns, "Apprentice") ~= nil and findNode(ns, "Armor") ~= nil, "second centre click unfolds every branch")
	checkMapGeometry(m, ns, "blacksmithing fully open")

	-- Recipe tooltip and shift-click link.
	axe = findNode(ns, "Copper Axe")
	m.Hover(axe)
	check(m.env.GameTooltip.lines[1]:find("[link]", 1, true) == 1, "recipe leaf shows the recipe tooltip")
	m.Leave()
	m.shift = true
	m.Click(axe)
	m.shift = false
	eq(m.insertedLink, "|cffffd000|Henchant:2738|h[Copper Axe]|h|r", "shift-click links the recipe")

	-- Bags change -> craftable counts update while the map is open.
	m.bags[ITEMS.weakFlux] = 0
	m.Fire("BAG_UPDATE_DELAYED")
	m.Advance(1)
	axe = findNode(ns, "Copper Axe")
	eq(axe and axe.value:GetText(), "", "craftable count follows the bags")

	-- Switching profession and closing.
	m.Click(ns.Bars.bars.mining)
	eq(ns.Map.current, "mining", "clicking another bar switches the map")
	eq(branchKeys(ns), "rank,next,nodes", "mining branches")
	local silver, iron = findNode(ns, "Silver Vein"), findNode(ns, "Iron Deposit")
	check(silver and silver.label.textColor[1] == 1 and silver.label.textColor[2] == 0.5, "silver vein is orange at 92 skill")
	check(findNode(ns, "Tin Vein") ~= nil and findNode(ns, "Incendicite Vein") == nil, "common veins are leaves, zone-specific ones are not")
	m.Hover(findNode(ns, "Ore"))
	check(table.concat(m.env.GameTooltip.lines, "\n"):find("Incendicite Vein (65)", 1, true) ~= nil, "zone-specific veins are listed in the tooltip")
	m.Leave()
	eq(plain(ns.Map.center.sub:GetText()), "92/150 · Journeyman", "centre node shows skill and rank")
	check(iron and iron.value:GetText() == "+33", "iron deposit shows the skill still needed")
	checkMapGeometry(m, ns, "mining")
	m.Click(ns.Bars.bars.mining)
	check(not ns.Map.frame:IsShown(), "clicking the open bar closes the map")
	check(ns.Bars.bars.mining.border[1].color[1] == 0, "selection border cleared")

	-- Skill-up: bar animates and turns gold when Journeyman becomes trainable.
	MODERN_PROFS[1].rank = 50
	m.Fire("CHAT_MSG_SKILL", "Your skill in Blacksmithing has increased to 50.")
	m.Advance(0.1)
	m.Advance(0.3)
	local mid = ns.Bars.bars.blacksmithing.status:GetValue()
	check(mid > 45 / 75 and mid < 50 / 75, "bar fill animates toward the new value")
	m.Advance(1)
	eq(ns.Bars.bars.blacksmithing.value:GetText(), "|cffffd10050/75|r", "gold once Journeyman is trainable")
	eq(math.floor(ns.Bars.bars.blacksmithing.status:GetValue() * 75 + 0.5), 50, "bar fill settles on the new value")
	m.Click(ns.Bars.bars.blacksmithing)
	check(findNode(ns, "Visit a trainer").data.value == "now", "next step says to train now")
	check(ns.Map.frame.footer:GetText():find("from skill 45", 1, true) ~= nil, "footer warns the recipe colours are stale")
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
	eq(ns.Bars.bars.cooking:GetWidth(), 260, "/pb width")
	slash(m, "width abc")
	check(m.ChatHas("usage: /pb width") ~= nil, "/pb width needs a number")
	slash(m, "mapscale 0.9")
	eq(ns.db.mapScale, 0.9, "/pb mapscale")

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
	check(not ns.Map.frame:IsShown(), "edit mode clicks do not open the map")
	slash(m, "edit")
	check(not ns.Bars.editing and not ns.Bars.bars.fishing:IsShown(), "leaving edit mode hides chosen bars")
	slash(m, "show all")
	check(ns.Bars.bars.fishing:IsShown(), "/pb show all")

	slash(m, "mining")
	eq(ns.Map.current, "mining", "/pb <profession> opens its map")
	slash(m, "map first aid")
	eq(ns.Map.current, "firstaid", "/pb map with a two-word name")
	slash(m, "map xyz")
	check(m.ChatHas("no learned profession matches 'xyz'") ~= nil, "/pb map with no match reports it")
	eq(ns.Map.current, "firstaid", "a bad /pb map leaves the map alone")
	local found = false
	for _, name in ipairs(m.env.UISpecialFrames) do
		found = found or name == "ProfessionBarsMap"
	end
	check(found, "Escape closes the map (UISpecialFrames)")
	ns.Map.frame:Hide()
	check(ns.Map.current == nil, "hiding the map clears its state")

	slash(m, "debug")
	check(m.ChatHas("interface 16001") and m.ChatHas("read from: GetProfessions + C_TradeSkillUI") and m.ChatHas("12 recipes cached"), "/pb debug reports what it sees")
	slash(m, "help")
	check(m.ChatHas("/pb theme") ~= nil, "/pb help lists commands")
	slash(m, "reset")
	eq(ns.db.theme, "vibrant", "/pb reset restores the theme")
	eq(ns.db.scale, 1, "/pb reset restores the scale")
	check(ns.db.point == nil, "/pb reset forgets the position")

	eq(#m.warnings, 0, "no conflicting anchors (" .. table.concat(m.warnings, "; ") .. ")")
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
	eq(ns.Bars.bars.enchanting.value:GetText(), "230/300", "artisan value")
	m.Click(ns.Bars.bars.tailoring)
	eq(branchKeys(ns), "rank,next,skillups", "map opens without C_TradeSkillUI")
	checkMapGeometry(m, ns, "tailoring")
	m.Click(ns.Bars.bars.enchanting)
	check(findNode(ns, "Max skill 300") ~= nil, "artisan next step is the grind to max")
	eq(#m.warnings, 0, "no conflicting anchors")
end }

scenarios[#scenarios + 1] = { "no professions yet", function()
	local m = Mock.new()
	installModern(m, {}, {})
	local ns = boot(m)
	eq(#shownBars(ns), 0, "no bars")
	check(ns.Bars.empty:IsShown(), "empty state shown")
	check(ns.Bars.empty.text:GetText():find("No professions yet", 1, true) == 1, "empty state text")
	slash(m, "map")
	check(m.ChatHas("no professions learned yet") ~= nil, "/pb map with nothing learned")
	m.Hover(ns.Bars.empty)
	check(ns.Bars.header:IsShown(), "empty state still reveals the header")
	eq(#m.warnings, 0, "no conflicting anchors")
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
	eq(branchKeys(ns), "rank,next", "unknown profession map")
	checkMapGeometry(m, ns, "unknown profession")
	m.Click(ns.Bars.bars.herbalism)
	check(findNode(ns, "Every rank learned") ~= nil, "mastered profession")
	check(findNode(ns, "Black Lotus") ~= nil, "herbalism at 300 lists top herbs")
	m.Click(ns.Bars.bars.skinning)
	eq(branchKeys(ns), "rank,next,skinning", "skinning map")
	check(findNode(ns, "Skin up to level 14") ~= nil, "skinning reach at 45 skill")
	check(findNode(ns, "Best: level 13-14") ~= nil, "best skinning band")
	check(findNode(ns, "Good: level 1-12") ~= nil, "good skinning band")
	check(findNode(ns, "Level 15 beasts") ~= nil, "next skinning milestone")
	checkMapGeometry(m, ns, "skinning")
	eq(#m.warnings, 0, "no conflicting anchors")
end }

scenarios[#scenarios + 1] = { "layout stress: every branch full and open", function()
	local m = Mock.new()
	m.level = 30
	local profs = { { slot = 2, name = "Mining", icon = 136248, rank = 140, max = 150, line = 186 } }
	installModern(m, profs, {})
	local ns = boot(m)
	-- Smelting recipes give mining all six branches.
	local recipes = {}
	for i = 1, 14 do
		recipes[i] = { 7000 + i, "Smelt Test Bar " .. i, i % 4, (i % 3) + 1, { { 2770, 1 } } }
	end
	m.bags[2770] = 40
	m.trade.open, m.trade.recipes = profs[1], recipes
	m.Fire("TRADE_SKILL_LIST_UPDATE")
	m.Advance(1)
	slash(m, "map mining")
	m.Click(ns.Map.center)
	if findNode(ns, "Copper Vein") == nil and findNode(ns, "Tin Vein") == nil and findNode(ns, "Apprentice") == nil then
		m.Click(ns.Map.center)
	end
	eq(#ns.Map.tree.branches, 6, "six branches")
	for _, branch in ipairs(ns.Map.tree.branches) do
		check(branch.open, "branch " .. branch.key .. " is open")
	end
	checkMapGeometry(m, ns, "everything open")
	-- A small screen forces the panel to scale down to fit.
	local small = Mock.new({ width = 1024, height = 420 })
	small.level = 30
	installModern(small, profs, {})
	local ns2 = boot(small)
	small.trade.open, small.trade.recipes = profs[1], recipes
	small.Fire("TRADE_SKILL_LIST_UPDATE")
	small.Advance(1)
	ns2.db.expanded.mining = { rank = true, next = true, nodes = true, skillups = true, craft = true, recipes = true }
	slash(small, "map mining")
	check(ns2.Map.frame:GetScale() < 1, "panel scales down on a short screen")
	checkMapGeometry(small, ns2, "short screen")
	eq(#m.warnings + #small.warnings, 0, "no conflicting anchors")
end }

---------------------------------------------------------------------------
-- Preview scene for tools/wowmock/preview.js
---------------------------------------------------------------------------

local function sceneMock(theme)
	local m = Mock.new()
	m.level = 18
	MODERN_PROFS[1].rank = 45
	installModern(m, MODERN_PROFS, CHILD_INFOS)
	m.bags = { [ITEMS.copperBar] = 26, [ITEMS.roughStone] = 7, [ITEMS.weakFlux] = 2, [ITEMS.lightLeather] = 3, [ITEMS.wool] = 1 }
	local ns = boot(m)
	m.trade.open, m.trade.recipes = MODERN_PROFS[1], BS_RECIPES
	m.Fire("TRADE_SKILL_LIST_UPDATE")
	m.Advance(1)
	if theme then
		slash(m, "theme " .. theme)
	end
	return m, ns
end

local function label(text, x, y, size)
	return { type = "text", rect = { l = x, r = x + 200, b = y, t = y + size + 4 }, text = text, size = size,
		color = { 1, 1, 1, 0.85 }, justifyH = "LEFT", anchorH = "LEFT", fixedWidth = true, alpha = 1 }
end

local SCENES = {}

-- Bars plus the Blacksmithing map, as in the screenshots in the README.
function SCENES.main()
	local m, ns = sceneMock()
	ns.db.point = { 40, 560 }
	ns.Bars:RestorePosition()
	ns.db.mapPoint = { 300 + ns.Map.metrics.WIDTH / 2, 400 }
	ns.Map:RestorePosition()
	m.Click(ns.Bars.bars.blacksmithing)
	m.Advance(1)
	m.Hover(ns.Bars.bars.blacksmithing)
	m.mouseOver = nil
	m.env.GameTooltip:Hide()
	return m, m.Dump()
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
		for _, op in ipairs(m.Dump()) do
			ops[#ops + 1] = op
		end
	end
	return m0, ops
end

-- A gathering profession's map (Mining) in the Midnight theme.
function SCENES.gather()
	local m, ns = sceneMock("midnight")
	ns.db.point = { 40, 560 }
	ns.Bars:RestorePosition()
	ns.db.mapPoint = { 300 + ns.Map.metrics.WIDTH / 2, 420 }
	ns.Map:RestorePosition()
	m.Click(ns.Bars.bars.mining)
	m.Click(ns.Map.nodes.active[1]) -- unfold Rank too
	m.Advance(1)
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

-- What the profession panel shows, as plain data. Detail.lua draws it.
--   view    = { step, next = { text, color, tooltip }, tabs = { tab }, footer }
--   tab     = { key, label, sections = { section } }
--   section = { key, title, right, rightColor, limit, rows = { row } }
--   row     = { label, value, color, valueColor, tooltip, recipe, itemID, wrap }
local _, ns = ...

local WHITE, GREY, DIM, GOOD, BAD = ns.WHITE, ns.GREY, ns.DIM, ns.GOOD, ns.BAD
local DIFF = ns.DIFFICULTY

local function line(text, color)
	color = color or GREY
	return { text, color[1], color[2], color[3] }
end

local function playerLevel()
	return (UnitLevel and UnitLevel("player")) or 1
end

local function message(text)
	return { label = text, color = GREY, wrap = true }
end

---------------------------------------------------------------------------
-- The plan: crafts for the next few skill points, and what to get for them
---------------------------------------------------------------------------

-- Plans look this many points ahead: recipe colours change as you level,
-- so anything longer would be guesswork.
ns.PLAN_HORIZON = 10
-- Expected skill points per craft: orange always gives one, yellow about
-- half the time.
local POINTS = { optimal = 1, medium = 0.5 }

-- Returns { points, horizon, crafts = { { recipe, count, ready, points } },
-- reagents = { { itemID, need, have, short } } }. Crafts your bags already
-- cover come first (orange before yellow, fewest reagents first); anything
-- left is planned on the best recipe. Reagents you are short of come first.
function ns.PlanSkillUps(cache, points)
	local horizon = math.min(points, ns.PLAN_HORIZON)
	local plan = { points = points, horizon = horizon, crafts = {}, reagents = {} }
	if not cache or horizon <= 0 then
		return plan
	end
	local candidates = {}
	for _, recipe in ipairs(cache.list) do
		if POINTS[recipe.diff] and recipe.reagents and #recipe.reagents > 0 then
			candidates[#candidates + 1] = recipe
		end
	end
	if #candidates == 0 then
		return plan
	end
	local function units(recipe)
		local total = 0
		for _, reagent in ipairs(recipe.reagents) do
			total = total + reagent[2]
		end
		return total
	end
	table.sort(candidates, function(a, b)
		local pa, pb = POINTS[a.diff], POINTS[b.diff]
		if pa ~= pb then
			return pa > pb
		end
		local ua, ub = units(a), units(b)
		if ua ~= ub then
			return ua < ub
		end
		return a.name < b.name
	end)

	local stock = {}
	local function have(itemID)
		if stock[itemID] == nil then
			stock[itemID] = ns.ItemCount(itemID)
		end
		return stock[itemID]
	end
	local function makeable(recipe)
		local best
		for _, reagent in ipairs(recipe.reagents) do
			local can = math.floor(math.max(0, have(reagent[1])) / reagent[2])
			if not best or can < best then
				best = can
			end
		end
		return best or 0
	end
	local entries, left = {}, horizon
	local function add(recipe, count, fromBags)
		local entry = entries[recipe]
		if not entry then
			entry = { recipe = recipe, count = 0, ready = 0, points = 0 }
			entries[recipe] = entry
			plan.crafts[#plan.crafts + 1] = entry
		end
		entry.count = entry.count + count
		entry.points = entry.points + count * POINTS[recipe.diff]
		if fromBags then
			entry.ready = entry.ready + count
		end
		for _, reagent in ipairs(recipe.reagents) do
			stock[reagent[1]] = have(reagent[1]) - reagent[2] * count
		end
		left = left - count * POINTS[recipe.diff]
	end
	for _, recipe in ipairs(candidates) do
		if left <= 0 then
			break
		end
		local count = math.min(math.ceil(left / POINTS[recipe.diff] - 1e-9), makeable(recipe))
		if count > 0 then
			add(recipe, count, true)
		end
	end
	if left > 0 then
		local best = candidates[1]
		add(best, math.ceil(left / POINTS[best.diff] - 1e-9), false)
	end

	local need, order = {}, {}
	for _, entry in ipairs(plan.crafts) do
		for _, reagent in ipairs(entry.recipe.reagents) do
			if not need[reagent[1]] then
				need[reagent[1]] = 0
				order[#order + 1] = reagent[1]
			end
			need[reagent[1]] = need[reagent[1]] + reagent[2] * entry.count
		end
	end
	for _, itemID in ipairs(order) do
		local inBags = ns.ItemCount(itemID)
		plan.reagents[#plan.reagents + 1] = { itemID = itemID, need = need[itemID], have = inBags, short = math.max(0, need[itemID] - inBags) }
	end
	table.sort(plan.reagents, function(a, b)
		if (a.short > 0) ~= (b.short > 0) then
			return a.short > 0
		end
		return a.itemID < b.itemID
	end)
	return plan
end

local function planTarget(prof, step)
	if step.state == "skill" then
		return step.nextRank.skill
	elseif prof.rank < prof.max then
		return prof.max
	end
	return nil
end

local function craftRow(craft)
	local recipe = craft.recipe
	local value, valueColor = "need items", BAD
	if craft.ready >= craft.count then
		value, valueColor = "ready", GOOD
	elseif craft.ready > 0 then
		value, valueColor = craft.ready .. " ready", GOOD
	end
	local why = recipe.diff == "optimal" and "Orange: +1 skill every craft." or "Yellow: +1 skill about half the time."
	return {
		label = craft.count .. "x " .. ns.ShortRecipeName(recipe.name), color = DIFF[recipe.diff],
		value = value, valueColor = valueColor, recipe = recipe, count = ns.CraftableCount(recipe),
		tooltip = { line(why, WHITE) },
	}
end

local function reagentRow(reagent)
	local enough = reagent.short == 0
	return {
		label = ns.ItemName(reagent.itemID) or ("Item " .. reagent.itemID), color = enough and GREY or WHITE,
		value = math.min(reagent.have, reagent.need) .. "/" .. reagent.need, valueColor = enough and GOOD or BAD,
		itemID = reagent.itemID,
		tooltip = { line("In your bags: " .. reagent.have .. ". The plan needs " .. reagent.need .. ".", WHITE) },
	}
end

local function planSections(prof, step, cache)
	if not cache or #cache.list == 0 then
		local text = C_TradeSkillUI and ("Open your " .. prof.name .. " window once to load your recipes.")
			or "This client doesn't share recipe data with addons."
		return { { key = "crafts", title = "Craft for skill", rows = { message(text) } } }
	end
	local target = planTarget(prof, step)
	if not target then
		local text = step.state == "mastered" and "Maxed out: nothing left to level."
			or ("You're at your cap. " .. (ns.DescribeStep(step, prof)) .. ".")
		return { { key = "crafts", title = "Craft for skill", rows = { message(text) } } }
	end
	local plan = ns.PlanSkillUps(cache, target - prof.rank)
	local crafts = {}
	for i, craft in ipairs(plan.crafts) do
		crafts[i] = craftRow(craft)
	end
	if #crafts == 0 then
		crafts[1] = message("No orange or yellow recipes left. Learn new ones at a trainer.")
	end
	local ahead
	if plan.points > plan.horizon then
		ahead = "next " .. plan.horizon .. " of " .. plan.points .. " points"
	else
		ahead = "next " .. plan.points .. (plan.points == 1 and " point" or " points")
	end
	local sections = { { key = "crafts", title = "Craft for skill", right = ahead, rows = crafts, limit = 5 } }
	if #plan.reagents > 0 then
		local rows, short = {}, 0
		for _, reagent in ipairs(plan.reagents) do
			rows[#rows + 1] = reagentRow(reagent)
			if reagent.short > 0 then
				short = short + 1
			end
		end
		sections[2] = {
			key = "shopping", title = "Shopping list", rows = rows, limit = 5,
			right = short > 0 and (short .. " to get") or "all in your bags", rightColor = short > 0 and BAD or GOOD,
		}
	end
	return sections
end

---------------------------------------------------------------------------
-- Recipes and camp
---------------------------------------------------------------------------

local function sortRecipes(a, b)
	local da, db = ns.DIFFICULTY_ORDER[a.recipe.diff] or 9, ns.DIFFICULTY_ORDER[b.recipe.diff] or 9
	if da ~= db then
		return da < db
	end
	if a.count ~= b.count then
		return a.count > b.count
	end
	return a.recipe.name < b.recipe.name
end

local function recipeRows(items)
	table.sort(items, sortRecipes)
	local rows = {}
	for i, item in ipairs(items) do
		rows[i] = {
			label = ns.ShortRecipeName(item.recipe.name), color = DIFF[item.recipe.diff] or WHITE,
			value = item.count > 0 and ("x" .. item.count) or "", valueColor = GOOD,
			recipe = item.recipe, count = item.count,
		}
	end
	return rows
end

local function recipeSections(cache, memo)
	local counts = { optimal = 0, medium = 0, easy = 0, trivial = 0 }
	local items = {}
	for _, recipe in ipairs(cache.list) do
		counts[recipe.diff] = (counts[recipe.diff] or 0) + 1
		items[#items + 1] = { recipe = recipe, count = ns.CraftableCount(recipe, memo) }
	end
	local parts = {}
	for _, diff in ipairs({ "optimal", "medium", "easy" }) do
		if counts[diff] > 0 then
			local c = DIFF[diff]
			parts[#parts + 1] = ns.ColorCode(c[1], c[2], c[3]) .. counts[diff] .. " " .. ns.DIFFICULTY_WORD[diff] .. "|r"
		end
	end
	return { {
		key = "known", title = "Known recipes (" .. #cache.list .. ")", right = table.concat(parts, "  "),
		rows = recipeRows(items), limit = 8,
	} }
end

-- Forever's camps: recipes the game names or files as camp objects.
local CAMP_NAMES = {
	["sharpening wheel"] = true, ["master forge"] = true, ["arcane forge"] = true, ["mana well"] = true,
	["rock garden"] = true, ["greenhouse"] = true, ["fish bowl"] = true, ["fishing rack"] = true,
	["fishing hut"] = true, ["alchemy laboratory"] = true, ["anarchist's workbench"] = true,
}

local function isCampRecipe(recipe)
	local name = recipe.name:lower()
	return CAMP_NAMES[name] or name:find("camp", 1, true) ~= nil or (recipe.cat and recipe.cat:lower():find("camp", 1, true) ~= nil)
end

local function campSections(cache, memo)
	if not cache then
		return nil
	end
	local items = {}
	for _, recipe in ipairs(cache.list) do
		if isCampRecipe(recipe) then
			items[#items + 1] = { recipe = recipe, count = ns.CraftableCount(recipe, memo) }
		end
	end
	if #items == 0 then
		return nil
	end
	return { { key = "camp", title = "Camp recipes", rows = recipeRows(items) } }
end

---------------------------------------------------------------------------
-- Ranks (secondary professions learn them from books and quests)
---------------------------------------------------------------------------

local function rankSections(prof, step)
	local rows = {}
	for i, rank in ipairs(step.rules) do
		local row = { label = rank.name, tooltip = { line("Skill up to " .. rank.cap .. ".", WHITE) } }
		if rank.skill then
			local level = rank.level > 0 and (", character level " .. rank.level) or ""
			row.tooltip[2] = line("Learn it from " .. ns.HowToLearn(prof, rank) .. " at skill " .. rank.skill .. level .. ".")
		end
		if i < step.index then
			row.value, row.valueColor, row.color = "done", GOOD, GREY
		elseif i == step.index then
			row.value, row.valueColor, row.color = prof.rank .. " / " .. rank.cap, WHITE, WHITE
		elseif i == step.index + 1 and step.state == "train" then
			row.value, row.valueColor, row.color = "learn now", GOOD, WHITE
		elseif i == step.index + 1 and step.state == "level" then
			row.value, row.valueColor, row.color = "at level " .. rank.level, BAD, WHITE
		else
			local how = (rank.how == "book" or rank.how == "quest") and rank.how or "trainer"
			row.value, row.valueColor, row.color = how .. " at " .. rank.skill, GREY, GREY
		end
		rows[i] = row
	end
	return { { key = "ranks", title = "How to rank up", rows = rows } }
end

---------------------------------------------------------------------------
-- Gathering and skinning: what's worth it now, and where to go
---------------------------------------------------------------------------

-- Zones as rows: the ones that suit your level first.
local function zoneRows(zones, level)
	local list = {}
	for _, zone in ipairs(zones) do
		local range = ns.ZONE_LEVELS[zone]
		list[#list + 1] = { zone = zone, range = range, fits = range == nil or range[1] <= level + 3 }
	end
	table.sort(list, function(a, b)
		if a.fits ~= b.fits then
			return a.fits
		end
		local la, lb = a.range and a.range[1] or 0, b.range and b.range[1] or 0
		if la ~= lb then
			return la < lb
		end
		return a.zone < b.zone
	end)
	local rows = {}
	for i, item in ipairs(list) do
		rows[i] = {
			label = item.zone, color = item.fits and WHITE or GREY,
			value = item.range and ("level " .. item.range[1] .. "-" .. item.range[2]) or "",
			valueColor = item.fits and GREY or BAD,
			tooltip = { line(item.fits and "Suits your level." or "Well above your level for now.", WHITE) },
		}
	end
	return rows
end

-- At the cap nothing gives skill, whatever its colour.
local function cappedMessage(prof, step)
	if prof.rank < prof.max then
		return nil
	elseif step.state == "mastered" then
		return message("Maxed out, so nothing gives skill any more.")
	end
	return message("You're at your cap: nothing gives skill until you learn " .. step.nextRank.name .. ".")
end

local function nodeTooltip(node, diff)
	local tip = { line("Needs " .. node[1] .. " skill.", WHITE) }
	if ns.CHANCE_WORD[diff] then
		tip[2] = line(ns.DIFFICULTY_WORD[diff]:gsub("^%l", string.upper) .. " for you: " .. ns.CHANCE_WORD[diff] .. " a skill-up.", DIFF[diff])
	end
	if node.note then
		tip[#tip + 1] = line(node.note:gsub("^%l", string.upper) .. ".")
	end
	return tip
end

local function whereGatherSections(prof, level, step)
	local best, special, nextNode = {}, {}, nil
	for _, node in ipairs(ns.GATHER[prof.key] or {}) do
		local diff = ns.GatherDifficulty(prof.rank, node[1])
		if diff == "locked" then
			if not node.special and not nextNode then
				nextNode = node
			end
		elseif diff ~= "trivial" then
			local list = node.special and special or best
			list[#list + 1] = { node = node, diff = diff }
		end
	end
	local function bestFirst(a, b)
		local da, db = ns.DIFFICULTY_ORDER[a.diff], ns.DIFFICULTY_ORDER[b.diff]
		if da ~= db then
			return da < db
		end
		if a.node[1] ~= b.node[1] then
			return a.node[1] > b.node[1]
		end
		return a.node[2] < b.node[2]
	end
	table.sort(best, bestFirst)
	table.sort(special, bestFirst)

	local rows = {}
	for i, item in ipairs(best) do
		rows[i] = {
			label = item.node[2], color = DIFF[item.diff], value = ns.CHANCE_WORD[item.diff],
			valueColor = GREY, tooltip = nodeTooltip(item.node, item.diff),
		}
	end
	local capped = cappedMessage(prof, step)
	if capped then
		rows = { capped }
	elseif #rows == 0 then
		rows[1] = message("Nothing in range gives skill-ups. Train the next rank or move up a tier.")
	end
	local sections = { { key = "best", title = "Best for you now", right = not capped and "skill-up chance" or nil, rows = rows, limit = 4 } }
	if #special > 0 then
		local specialRows = {}
		for i, item in ipairs(special) do
			specialRows[i] = {
				label = item.node[2], color = DIFF[item.diff], value = item.node.note, valueColor = GREY,
				tooltip = nodeTooltip(item.node, item.diff),
			}
		end
		sections[#sections + 1] = { key = "special", title = "Rare or one zone", rows = specialRows, limit = 3 }
	end
	for _, bracket in ipairs(ns.GATHER_ZONES[prof.key] or {}) do
		if prof.rank >= bracket[1] and prof.rank < bracket[2] then
			local zoneList = zoneRows(bracket[4], level)
			for _, row in ipairs(zoneList) do
				row.tooltip[2] = line("Good for " .. bracket[3] .. " at your skill.")
			end
			sections[#sections + 1] = { key = "zones", title = "Where to go", right = "you're level " .. level, rows = zoneList, limit = 4 }
			break
		end
	end
	if nextNode then
		sections[#sections + 1] = { key = "next", title = "Next unlock", rows = { {
			label = nextNode[2], color = WHITE, value = "at " .. nextNode[1] .. " (+" .. (nextNode[1] - prof.rank) .. ")", valueColor = GREY,
		} } }
	end
	return sections
end

local function nodeSections(prof)
	local rows = {}
	for i, node in ipairs(ns.GATHER[prof.key] or {}) do
		local diff = ns.GatherDifficulty(prof.rank, node[1])
		rows[i] = {
			label = node[2], color = DIFF[diff] or DIM,
			value = node[1] .. (node.note and (" · " .. node.note) or ""), valueColor = diff == "locked" and DIM or GREY,
			tooltip = nodeTooltip(node, diff),
		}
	end
	return { { key = "all", title = "Every node", right = "skill needed", rows = rows, limit = 8 } }
end

local function rangeText(lo, hi)
	return lo == hi and tostring(lo) or (lo .. "-" .. hi)
end

local function whereSkinSections(prof, level, step)
	local top = math.min(63, ns.SkinMaxLevel(prof.rank))
	local bands = { optimal = {}, medium = {} }
	for beast = 1, top do
		local band = bands[ns.GatherDifficulty(prof.rank, ns.SkinReq(beast))]
		if band then
			band.lo = band.lo or beast
			band.hi = beast
		end
	end
	local rows = {}
	for _, diff in ipairs({ "optimal", "medium" }) do
		local band = bands[diff]
		if band.lo then
			rows[#rows + 1] = {
				label = "Level " .. rangeText(band.lo, band.hi) .. " beasts", color = DIFF[diff],
				value = ns.CHANCE_WORD[diff], valueColor = GREY,
			}
		end
	end
	local capped = cappedMessage(prof, step)
	if capped then
		rows = { capped }
	end
	local sections = { { key = "best", title = "Best for you now", right = not capped and "skill-up chance" or nil, rows = rows } }
	local band = bands.optimal.lo and bands.optimal or bands.medium
	if band.lo then
		local zones = {}
		for zone, range in pairs(ns.ZONE_LEVELS) do
			if range[1] <= band.hi and range[2] >= band.lo then
				zones[#zones + 1] = zone
			end
		end
		sections[#sections + 1] = {
			key = "zones", title = "Where to go", right = "beasts " .. rangeText(band.lo, band.hi),
			rows = zoneRows(zones, level), limit = 4,
		}
	end
	if top < 63 then
		local need = ns.SkinReq(top + 1)
		sections[#sections + 1] = { key = "next", title = "Next unlock", rows = { {
			label = "Level " .. (top + 1) .. " beasts", color = WHITE, value = "at " .. need .. " (+" .. (need - prof.rank) .. ")", valueColor = GREY,
		} } }
	end
	return sections
end

---------------------------------------------------------------------------
-- Your other characters
---------------------------------------------------------------------------

local function altSections(prof)
	local alts = ns.AltsWith(prof.key)
	if #alts == 0 then
		return nil
	end
	local rows = {}
	for i, alt in ipairs(alts) do
		local colors = RAID_CLASS_COLORS
		local c = colors and alt.class and colors[alt.class]
		local color = c and { c.r, c.g, c.b } or WHITE
		local who = alt.className and ((alt.level and ("Level " .. alt.level .. " ") or "") .. alt.className) or alt.id
		rows[i] = {
			label = alt.name, color = color, value = alt.rank .. " / " .. alt.max, valueColor = WHITE,
			tooltip = { line(who, WHITE), line(alt.id) },
		}
	end
	return { { key = "alts", title = "Your other characters", rows = rows } }
end

---------------------------------------------------------------------------

function ns.BuildDetail(prof)
	local level = playerLevel()
	local step = ns.NextStep(prof, level)
	local short, long, color = ns.DescribeStep(step, prof)
	local view = { step = step, next = { text = short, color = color, tooltip = long }, tabs = {} }
	local cache = ns.GetRecipeCache(prof.key)
	local hasRecipes = cache and #cache.list > 0
	local memo = {}
	local function tab(key, label, sections)
		if sections and #sections > 0 then
			view.tabs[#view.tabs + 1] = { key = key, label = label, sections = sections }
		end
	end
	if prof.type == "craft" or prof.type == "other" then
		tab("plan", "Plan", planSections(prof, step, cache))
		if hasRecipes then
			tab("recipes", "Recipes", recipeSections(cache, memo))
			tab("camp", "Camp", campSections(cache, memo))
		end
	elseif prof.type == "gather" then
		tab("where", "Where", whereGatherSections(prof, level, step))
		tab("nodes", "Nodes", nodeSections(prof))
		if hasRecipes then
			tab("plan", prof.key == "mining" and "Smelting" or "Plan", planSections(prof, step, cache))
		end
	elseif prof.type == "skin" then
		tab("where", "Where", whereSkinSections(prof, level, step))
	end
	if prof.kind == "secondary" or #view.tabs == 0 then
		tab("ranks", "Ranks", rankSections(prof, step))
	end
	tab("alts", "Alts", altSections(prof))
	if hasRecipes and cache.rank and cache.rank < prof.rank then
		view.footer = "Recipe colours are from skill " .. cache.rank .. ". Open " .. prof.name .. " to refresh them."
	end
	return view
end

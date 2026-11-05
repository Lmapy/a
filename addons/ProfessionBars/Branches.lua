-- Builds the mind map content for one profession as plain data:
--   { branches = { { key, label, value, valueColor, side, defaultOpen,
--                    tooltip, children = { { label, value, color,
--                    valueColor, stripe, tooltip, recipe, itemID } } } },
--     footer = "..." }
-- MindMap.lua decides where things go and draws them.
local _, ns = ...

local WHITE, GREY, DIM, GOOD, BAD, GOLD = ns.WHITE, ns.GREY, ns.DIM, ns.GOOD, ns.BAD, ns.GOLD
local DIFF = ns.DIFFICULTY

local function line(text, color)
	color = color or GREY
	return { text, color[1], color[2], color[3] }
end

local function playerLevel()
	return (UnitLevel and UnitLevel("player")) or 1
end

---------------------------------------------------------------------------
-- Progress: rank ladder and the next step
---------------------------------------------------------------------------

local function rankBranch(prof, step, accent)
	local children = {}
	for i, rank in ipairs(ns.RANKS) do
		local child = { label = rank.name, tooltip = { line(rank.name .. ": skill up to " .. rank.cap, WHITE) } }
		if rank.skill then
			child.tooltip[2] = line("Learn at skill " .. rank.skill .. " and character level " .. rank.level .. ".")
		end
		if i < step.index then
			child.value, child.valueColor, child.stripe = "done", GOOD, GOOD
		elseif i == step.index then
			child.value, child.valueColor, child.stripe = prof.rank .. "/" .. prof.max, accent, accent
		else
			child.value = rank.skill .. " · lvl " .. rank.level
			child.color, child.valueColor, child.stripe = GREY, DIM, DIM
		end
		children[i] = child
	end
	return {
		key = "rank", label = "Rank", value = step.rank.name, valueColor = accent, side = "right",
		tooltip = { line("Profession ranks raise your skill cap. Each one is learned from a trainer.") },
		children = children,
	}
end

local function requirement(label, have, need)
	local done = have >= need
	return {
		label = label,
		value = math.min(have, need) .. "/" .. need,
		valueColor = done and GOOD or WHITE,
		stripe = done and GOOD or DIM,
		color = done and WHITE or GREY,
	}
end

local function nextBranch(prof, step)
	local short, long, color = ns.DescribeStep(step, prof)
	local children = {}
	local nextRank = step.nextRank
	if nextRank then
		children[#children + 1] = requirement("Skill " .. nextRank.skill, prof.rank, nextRank.skill)
		children[#children + 1] = requirement("Character level " .. nextRank.level, playerLevel(), nextRank.level)
		local where = prof.kind == "primary" and "Visit a trainer" or "Trainer, book or quest"
		children[#children + 1] = {
			label = where,
			value = step.state == "train" and "now" or "",
			valueColor = GOOD,
			stripe = step.state == "train" and GOOD or DIM,
			color = step.state == "train" and WHITE or GREY,
			tooltip = { line(prof.kind == "primary" and "Any " .. prof.name .. " trainer can teach the next rank." or "Secondary skills may need a book or a quest for higher ranks.") },
		}
		children[#children + 1] = {
			label = nextRank.name .. " cap", value = tostring(nextRank.cap), valueColor = GOLD, stripe = GOLD,
			tooltip = { line("Learning " .. nextRank.name .. " raises your cap from " .. prof.max .. " to " .. nextRank.cap .. ".") },
		}
	elseif step.state == "grind" then
		children[#children + 1] = requirement("Max skill " .. prof.max, prof.rank, prof.max)
	else
		children[#children + 1] = { label = "Every rank learned", value = "max", valueColor = GOLD, stripe = GOLD }
	end
	local tooltip = { line(long, WHITE) }
	local gain = ns.SessionGain(prof)
	if gain > 0 then
		tooltip[#tooltip + 1] = line("+" .. gain .. " skill this session.", GOOD)
	end
	return {
		key = "next", label = "Next step", value = short, valueColor = color, side = "right", defaultOpen = true,
		tooltip = tooltip, children = children,
	}
end

---------------------------------------------------------------------------
-- Recipes: skill-ups, the plan, what you can craft, categories, camp
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

local function recipeLeaf(item)
	local recipe = item.recipe
	return {
		label = ns.ShortRecipeName(recipe.name),
		color = DIFF[recipe.diff] or WHITE,
		value = item.count > 0 and ("x" .. item.count) or "",
		valueColor = GOOD,
		stripe = DIFF[recipe.diff] or DIM,
		recipe = recipe,
		count = item.count,
	}
end

local function recipeLeaves(items)
	table.sort(items, sortRecipes)
	local children = {}
	for i, item in ipairs(items) do
		children[i] = recipeLeaf(item)
	end
	return children
end

local function noRecipes(prof)
	return {
		key = "skillups", label = "Skill-ups", value = "not loaded", valueColor = DIM, side = "left", defaultOpen = true,
		tooltip = { line("Recipes load the first time you open your " .. prof.name .. " window.") },
		children = { { label = "Open your " .. prof.name .. " window", color = GREY, stripe = DIM } },
	}
end

local function skillUpsBranch(prof, cache, memo)
	if not cache or #cache.list == 0 then
		return noRecipes(prof)
	end
	local counts = { optimal = 0, medium = 0, easy = 0, trivial = 0 }
	local items = {}
	for _, recipe in ipairs(cache.list) do
		counts[recipe.diff] = (counts[recipe.diff] or 0) + 1
		if recipe.diff == "optimal" or recipe.diff == "medium" then
			items[#items + 1] = { recipe = recipe, count = ns.CraftableCount(recipe, memo) }
		end
	end
	local children = recipeLeaves(items)
	if #children == 0 then
		children[1] = { label = "No orange or yellow recipes", color = GREY, stripe = DIM }
		children[2] = { label = "Learn new ones at a trainer", color = GREY, stripe = DIM }
	end
	local function n(diff)
		local c = DIFF[diff]
		return ns.ColorCode(c[1], c[2], c[3]) .. counts[diff] .. "|r"
	end
	return {
		key = "skillups", label = "Skill-ups", value = n("optimal") .. " · " .. n("medium") .. " · " .. n("easy"),
		side = "left",
		tooltip = {
			line(counts.optimal .. " orange: always a skill-up", DIFF.optimal),
			line(counts.medium .. " yellow: usually a skill-up", DIFF.medium),
			line(counts.easy .. " green: sometimes a skill-up", DIFF.easy),
			line(counts.trivial .. " grey: no skill-ups", DIFF.trivial),
		},
		children = children,
	}
end

-- Orange recipes give a point every craft, so plans only use those. Recipes
-- you can already make (from your bags) go first, cheapest first; whatever
-- is left is planned on the recipe with the fewest reagents. Returns
-- { points, crafts = { { recipe, count, ready } }, reagents = { { itemID,
-- need, have, short } } } with reagents you are short of first. `ready` is
-- how many of a craft your bags already cover.
function ns.PlanSkillUps(cache, points)
	local plan = { points = points, crafts = {}, reagents = {} }
	if not cache or points <= 0 then
		return plan
	end
	local orange = {}
	for _, recipe in ipairs(cache.list) do
		if recipe.diff == "optimal" and recipe.reagents and #recipe.reagents > 0 then
			orange[#orange + 1] = recipe
		end
	end
	if #orange == 0 then
		return plan
	end
	local function units(recipe)
		local total = 0
		for _, reagent in ipairs(recipe.reagents) do
			total = total + reagent[2]
		end
		return total
	end
	table.sort(orange, function(a, b)
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
	local entries, left = {}, points
	local function add(recipe, count, fromBags)
		local entry = entries[recipe]
		if not entry then
			entry = { recipe = recipe, count = 0, ready = 0 }
			entries[recipe] = entry
			plan.crafts[#plan.crafts + 1] = entry
		end
		entry.count = entry.count + count
		if fromBags then
			entry.ready = entry.ready + count
		end
		for _, reagent in ipairs(recipe.reagents) do
			stock[reagent[1]] = have(reagent[1]) - reagent[2] * count
		end
		left = left - count
	end
	for _, recipe in ipairs(orange) do
		if left <= 0 then
			break
		end
		local count = math.min(left, makeable(recipe))
		if count > 0 then
			add(recipe, count, true)
		end
	end
	if left > 0 then
		add(orange[1], left, false)
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

local function reagentLeaf(reagent)
	local enough = reagent.short == 0
	return {
		label = ns.ItemName(reagent.itemID) or ("Item " .. reagent.itemID),
		value = math.min(reagent.have, reagent.need) .. "/" .. reagent.need,
		valueColor = enough and GOOD or BAD,
		color = enough and GREY or WHITE,
		stripe = enough and GOOD or BAD,
		itemID = reagent.itemID,
		tooltip = { line("In your bags: " .. reagent.have .. ". Needed for the plan: " .. reagent.need .. ".", WHITE) },
	}
end

local function planBranch(prof, step, cache)
	local target = planTarget(prof, step)
	if not target or not cache or #cache.list == 0 then
		return nil
	end
	local plan = ns.PlanSkillUps(cache, target - prof.rank)
	local branch = {
		key = "plan", label = "Plan to " .. target, side = "left", defaultOpen = prof.type == "craft",
		tooltip = {
			line("Orange recipes give a point every craft. This plan reaches " .. target .. " from your bags first, then with the recipe that needs the fewest reagents.", WHITE),
			line("Red items are what to buy or farm. Some recipes turn yellow as you level, so treat it as a minimum."),
		},
	}
	if #plan.crafts == 0 then
		branch.value, branch.valueColor = "no orange", DIM
		branch.children = {
			{ label = "No orange recipes left", color = GREY, stripe = DIM },
			{ label = "Train or find new recipes", color = GREY, stripe = DIM },
		}
		return branch
	end
	-- What you are short of first, then the crafts, then what you have.
	local children, short = {}, 0
	for _, reagent in ipairs(plan.reagents) do
		if reagent.short > 0 then
			short = short + 1
			children[#children + 1] = reagentLeaf(reagent)
		end
	end
	for _, craft in ipairs(plan.crafts) do
		local ready = craft.ready >= craft.count and "ready" or (craft.ready > 0 and (craft.ready .. " ready") or "")
		children[#children + 1] = {
			label = craft.count .. "x " .. ns.ShortRecipeName(craft.recipe.name),
			value = ready, valueColor = GOOD, color = DIFF.optimal, stripe = DIFF.optimal,
			recipe = craft.recipe, count = ns.CraftableCount(craft.recipe),
			tooltip = { line("Craft " .. craft.count .. " for +" .. craft.count .. " skill.", WHITE) },
		}
	end
	for _, reagent in ipairs(plan.reagents) do
		if reagent.short == 0 then
			children[#children + 1] = reagentLeaf(reagent)
		end
	end
	branch.children = children
	branch.value = short > 0 and (short .. (short == 1 and " item short" or " items short")) or "all in bags"
	branch.valueColor = short > 0 and BAD or GOOD
	return branch
end

local function craftableBranch(prof, cache, memo)
	if not cache or #cache.list == 0 then
		return nil
	end
	local items = {}
	for _, recipe in ipairs(cache.list) do
		local count = ns.CraftableCount(recipe, memo)
		if count > 0 then
			items[#items + 1] = { recipe = recipe, count = count }
		end
	end
	local children = recipeLeaves(items)
	if #children == 0 then
		children[1] = { label = "Nothing yet - gather reagents", color = GREY, stripe = DIM }
	end
	return {
		key = "craft", label = "Ready to craft", value = #items > 0 and (#items .. " recipes") or "none",
		valueColor = #items > 0 and GOOD or DIM, side = "left",
		tooltip = { line("Recipes you have every reagent for, counting your bags right now.") },
		children = children,
	}
end

local function recipesBranch(prof, cache)
	if not cache or #cache.list == 0 then
		return nil
	end
	local counts, order = {}, {}
	for _, recipe in ipairs(cache.list) do
		local cat = recipe.cat or "Other"
		if not counts[cat] then
			counts[cat] = 0
			order[#order + 1] = cat
		end
		counts[cat] = counts[cat] + 1
	end
	table.sort(order, function(a, b)
		if counts[a] ~= counts[b] then
			return counts[a] > counts[b]
		end
		return a < b
	end)
	local children = {}
	for i, cat in ipairs(order) do
		children[i] = { label = cat, value = tostring(counts[cat]), valueColor = WHITE }
	end
	return {
		key = "recipes", label = "Recipes", value = #cache.list .. " known", valueColor = WHITE, side = "left",
		tooltip = { line("Known recipes by category.") }, children = children,
	}
end

-- Forever's camps: any known recipe the game files under a camp category
-- or names as one. Hidden when there are none, so it never guesses.
local function campBranch(prof, cache, memo)
	if not cache then
		return nil
	end
	local items = {}
	for _, recipe in ipairs(cache.list) do
		local text = ((recipe.cat or "") .. " " .. recipe.name):lower()
		if text:find("camp", 1, true) then
			items[#items + 1] = { recipe = recipe, count = ns.CraftableCount(recipe, memo) }
		end
	end
	if #items == 0 then
		return nil
	end
	return {
		key = "camp", label = "Camp", value = #items .. " known", valueColor = WHITE, side = "left",
		tooltip = { line("Camp recipes you know, as your profession window lists them.", WHITE) },
		children = recipeLeaves(items),
	}
end

---------------------------------------------------------------------------
-- Gathering
---------------------------------------------------------------------------

local function nodesBranch(prof)
	local nodes = ns.GATHER[prof.key]
	if not nodes then
		return nil
	end
	-- Common nodes become leaves; zone-specific ones go in the tooltip.
	local open, locked, rare = {}, {}, {}
	for _, node in ipairs(nodes) do
		local diff = ns.GatherDifficulty(prof.rank, node[1])
		if node.rare then
			if diff ~= "trivial" then
				rare[#rare + 1] = line(node[2] .. " (" .. node[1] .. ")", DIFF[diff])
			end
		elseif diff == "locked" then
			locked[#locked + 1] = node
		elseif diff ~= "trivial" then
			open[#open + 1] = { node = node, diff = diff }
		end
	end
	local children = {}
	for i = math.max(1, #open - 2), #open do
		local item = open[i]
		children[#children + 1] = {
			label = item.node[2], value = tostring(item.node[1]), color = DIFF[item.diff], stripe = DIFF[item.diff], valueColor = GREY,
			tooltip = { line("Needs " .. item.node[1] .. " skill.", WHITE) },
		}
	end
	for i = 1, math.min(3, #locked) do
		local node = locked[i]
		children[#children + 1] = {
			label = node[2], value = "+" .. (node[1] - prof.rank), color = GREY, stripe = DIM, valueColor = DIM,
			tooltip = { line("Unlocks at " .. node[1] .. " skill (" .. (node[1] - prof.rank) .. " more).", WHITE) },
		}
	end
	local tooltip = {
		line("Best skill-ups first. Orange always gives a point, yellow usually, green sometimes.", WHITE),
		line("Classic node levels; new Forever zones may add more."),
	}
	if #rare > 0 then
		tooltip[#tooltip + 1] = line("Zone-specific nodes in range:", WHITE)
		for _, entry in ipairs(rare) do
			tooltip[#tooltip + 1] = entry
		end
	end
	local nextNode = locked[1]
	return {
		key = "nodes", label = prof.key == "mining" and "Ore" or "Herbs",
		value = nextNode and ("next at " .. nextNode[1]) or "all unlocked",
		valueColor = WHITE, side = "left", defaultOpen = true, tooltip = tooltip, children = children,
	}
end

local function rangeText(lo, hi)
	if lo == hi then
		return "level " .. lo
	end
	return "level " .. lo .. "-" .. hi
end

local function skinningBranch(prof)
	local top = math.min(63, ns.SkinMaxLevel(prof.rank))
	local bands = { optimal = {}, medium = {} }
	for level = 1, top do
		local diff = ns.GatherDifficulty(prof.rank, ns.SkinReq(level))
		if bands[diff] then
			local band = bands[diff]
			band.lo = band.lo or level
			band.hi = level
		end
	end
	local children = {
		{ label = "Skin up to " .. rangeText(top, top), color = WHITE, stripe = GOOD, tooltip = { line("Highest level beast you can skin at " .. prof.rank .. " skill.", WHITE) } },
	}
	if bands.optimal.lo then
		children[#children + 1] = { label = "Best: " .. rangeText(bands.optimal.lo, bands.optimal.hi), color = DIFF.optimal, stripe = DIFF.optimal }
	end
	if bands.medium.lo then
		children[#children + 1] = { label = "Good: " .. rangeText(bands.medium.lo, bands.medium.hi), color = DIFF.medium, stripe = DIFF.medium }
	end
	local level = top + 1
	while #children < 6 and level <= 63 do
		local need = ns.SkinReq(level)
		children[#children + 1] = {
			label = "Level " .. level .. " beasts", value = "at " .. need, color = GREY, stripe = DIM, valueColor = DIM,
			tooltip = { line(need - prof.rank .. " more skill.", WHITE) },
		}
		level = (level % 5 == 0) and level + 5 or (math.floor(level / 5) + 1) * 5
	end
	return {
		key = "skinning", label = "Beasts", value = "up to lvl " .. top, valueColor = WHITE, side = "left", defaultOpen = true,
		tooltip = { line("Which beasts you can skin, and where the best skill-ups are.", WHITE) },
		children = children,
	}
end

---------------------------------------------------------------------------
-- Your other characters
---------------------------------------------------------------------------

local function classColor(classFile)
	local colors = RAID_CLASS_COLORS
	local c = colors and classFile and colors[classFile]
	if c then
		return { c.r, c.g, c.b }
	end
	return WHITE
end

local function altsBranch(prof)
	local alts = ns.AltsWith(prof.key)
	if #alts == 0 then
		return nil
	end
	local children = {}
	for i, alt in ipairs(alts) do
		local who = alt.className and ((alt.level and ("Level " .. alt.level .. " ") or "") .. alt.className) or alt.id
		children[i] = {
			label = alt.name, color = classColor(alt.class), value = alt.rank .. "/" .. alt.max, valueColor = WHITE,
			stripe = classColor(alt.class), tooltip = { line(who, WHITE), line(alt.id) },
		}
	end
	return {
		key = "alts", label = "Alts", value = #alts .. (#alts == 1 and " character" or " characters"), valueColor = WHITE,
		side = "right", tooltip = { line("Your other characters with " .. prof.name .. ", as of their last login.") },
		children = children,
	}
end

---------------------------------------------------------------------------

local function add(list, branch)
	if branch then
		list[#list + 1] = branch
	end
end

function ns.BuildMindMap(prof)
	local r, g, b = ns.Color(prof.key)
	local accent = { math.min(1, r * 1.15 + 0.08), math.min(1, g * 1.15 + 0.08), math.min(1, b * 1.15 + 0.08) }
	local step = ns.NextStep(prof.rank, prof.max, playerLevel())
	local tree = { branches = {}, step = step }
	local branches = tree.branches
	add(branches, rankBranch(prof, step, accent))
	add(branches, nextBranch(prof, step))

	local cache = ns.GetRecipeCache(prof.key)
	local memo = {}
	if prof.type == "gather" then
		add(branches, nodesBranch(prof))
	elseif prof.type == "skin" then
		add(branches, skinningBranch(prof))
	end
	local hasRecipes = cache and #cache.list > 0
	if prof.type == "craft" or hasRecipes then
		add(branches, skillUpsBranch(prof, cache, memo))
		add(branches, planBranch(prof, step, cache))
		add(branches, craftableBranch(prof, cache, memo))
		add(branches, recipesBranch(prof, cache))
		add(branches, campBranch(prof, cache, memo))
	end
	add(branches, altsBranch(prof))

	if hasRecipes and cache.rank and cache.rank < prof.rank then
		tree.footer = "Recipe colours are from skill " .. cache.rank .. ". Open " .. prof.name .. " to refresh them."
	elseif hasRecipes then
		tree.footer = "Counts follow your bags live. Hover recipes and reagents for details; shift-click a recipe to link it."
	elseif prof.type == "craft" then
		tree.footer = "Open your " .. prof.name .. " window once to load recipes into the map."
	elseif prof.type == "gather" or prof.type == "skin" then
		tree.footer = "Orange always gives a skill-up, yellow usually, green sometimes."
	else
		tree.footer = "Click the centre to unfold or fold every branch."
	end
	return tree
end

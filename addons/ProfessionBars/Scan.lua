-- Reading the game: which professions you have (skill levels) and, when a
-- profession window is open, which recipes you know. Every API call is
-- guarded so a missing or changed function only loses that piece of data.
local _, ns = ...

local P = ns.PROFESSIONS
local byID, byName = {}, {}
for key, prof in pairs(P) do
	byID[prof.id] = key
	byName[prof.name:lower()] = key
end

-- Which API paths produced the last scan (shown by /pb debug).
ns.scanSources = {}

-- Map a skill line id and/or (localised) name to our profession key.
-- Unknown professions get an "other:<name>" key so they still show up.
function ns.KeyFor(id, name)
	id = ns.PlainNumber(id)
	if id and byID[id] then
		return byID[id]
	end
	name = ns.PlainString(name)
	if name then
		return byName[name:lower()] or ("other:" .. name:lower())
	end
	return nil
end

-- Remember localised names seen next to a known id, so later name-only
-- lookups (for example the trade skill window) resolve correctly.
local function learnName(key, name)
	if key and name and P[key] then
		byName[name:lower()] = key
	end
end

local function makeEntry(key, name, icon, rank, maxRank, modifier, skillLine)
	local static = P[key]
	return {
		key = key,
		name = name or (static and static.name) or key,
		icon = icon or (static and static.icon) or ns.UNKNOWN_ICON,
		rank = rank or 0,
		max = maxRank or 0,
		modifier = modifier or 0,
		skillLine = skillLine or (static and static.id),
		kind = (static and static.kind) or "other",
		type = (static and static.type) or "other",
	}
end

local function add(list, seen, entry)
	if not entry.key or seen[entry.key] or entry.max <= 0 then
		return false
	end
	seen[entry.key] = true
	list[#list + 1] = entry
	return true
end

-- 1) GetProfessions() + GetProfessionInfo(): the Mainline way.
local function scanProfessionIndices(list, seen)
	if type(GetProfessions) ~= "function" or type(GetProfessionInfo) ~= "function" then
		return false
	end
	local ok, a, b, c, d, e, f, g = pcall(GetProfessions)
	if not ok then
		return false
	end
	local found = false
	local indices = { a, b, c, d, e, f, g }
	for i = 1, 7 do
		local index = indices[i]
		if index then
			local okInfo, name, icon, rank, maxRank, _, _, skillLine, modifier = pcall(GetProfessionInfo, index)
			if okInfo then
				name = ns.PlainString(name)
				skillLine = ns.PlainNumber(skillLine)
				local key = ns.KeyFor(skillLine, name)
				learnName(key, name)
				local entry = makeEntry(key, name, ns.Plain(icon), ns.PlainNumber(rank), ns.PlainNumber(maxRank), ns.PlainNumber(modifier), skillLine)
				found = add(list, seen, entry) or found
			end
		end
	end
	return found
end

-- 2) C_TradeSkillUI profession info tables. Only fills in *known*
-- professions the first pass missed (for example First Aid).
local function fromInfo(list, seen, info)
	if type(info) ~= "table" then
		return false
	end
	local rank, maxRank = ns.PlainNumber(info.skillLevel), ns.PlainNumber(info.maxSkillLevel)
	if not rank or not maxRank or rank <= 0 or maxRank <= 0 then
		return false
	end
	local parentID = ns.PlainNumber(info.parentProfessionID)
	if parentID == 0 then
		parentID = nil
	end
	local key = ns.KeyFor(parentID, ns.PlainString(info.parentProfessionName))
	if not key or not P[key] then
		key = ns.KeyFor(info.professionID, info.professionName)
	end
	if not key or not P[key] or seen[key] then
		return false
	end
	return add(list, seen, makeEntry(key, nil, nil, rank, maxRank, ns.PlainNumber(info.skillModifier), P[key].id))
end

local function scanTradeSkillInfo(list, seen)
	local T = C_TradeSkillUI
	if not T then
		return false
	end
	local found = false
	if T.GetChildProfessionInfos then
		local ok, infos = pcall(T.GetChildProfessionInfos)
		if ok and type(infos) == "table" then
			for _, info in ipairs(infos) do
				found = fromInfo(list, seen, info) or found
			end
		end
	end
	if T.GetProfessionInfoBySkillLineID then
		for key, prof in pairs(P) do
			if not seen[key] then
				local ok, info = pcall(T.GetProfessionInfoBySkillLineID, prof.id)
				if ok then
					found = fromInfo(list, seen, info) or found
				end
			end
		end
	end
	return found
end

-- 3) Classic-style skill lines, only if nothing else worked.
local function scanSkillLines(list, seen)
	if type(GetNumSkillLines) ~= "function" or type(GetSkillLineInfo) ~= "function" then
		return false
	end
	local found = false
	for i = 1, GetNumSkillLines() do
		local name, isHeader, _, rank, _, modifier, maxRank = GetSkillLineInfo(i)
		name = ns.PlainString(name)
		if name and not isHeader then
			local key = ns.KeyFor(nil, name)
			if key and P[key] then
				found = add(list, seen, makeEntry(key, name, nil, ns.PlainNumber(rank), ns.PlainNumber(maxRank), ns.PlainNumber(modifier))) or found
			end
		end
	end
	return found
end

local KIND_ORDER = { primary = 1, secondary = 2, other = 3 }
local function byKindThenName(a, b)
	local ka, kb = KIND_ORDER[a.kind] or 3, KIND_ORDER[b.kind] or 3
	if ka ~= kb then
		return ka < kb
	end
	return a.name < b.name
end

function ns.ScanProfessions()
	local list, seen, sources = {}, {}, {}
	if scanProfessionIndices(list, seen) then
		sources[#sources + 1] = "GetProfessions"
	end
	if scanTradeSkillInfo(list, seen) then
		sources[#sources + 1] = "C_TradeSkillUI"
	end
	if #list == 0 and scanSkillLines(list, seen) then
		sources[#sources + 1] = "GetSkillLineInfo"
	end
	table.sort(list, byKindThenName)
	ns.scanSources = sources
	return list
end

---------------------------------------------------------------------------
-- Recipes (cached per character whenever one of your profession windows
-- is open, because the game only shares recipe colours then).
---------------------------------------------------------------------------

local DIFF_BY_ENUM = {}
do
	local E = Enum and Enum.TradeskillRelativeDifficulty
	DIFF_BY_ENUM[(E and E.Optimal) or 0] = "optimal"
	DIFF_BY_ENUM[(E and E.Medium) or 1] = "medium"
	DIFF_BY_ENUM[(E and E.Easy) or 2] = "easy"
	DIFF_BY_ENUM[(E and E.Trivial) or 3] = "trivial"
end

function ns.GetRecipeCache(key, create)
	local all = ns.char and ns.char.recipes
	if not all then
		return nil
	end
	if not all[key] and create then
		all[key] = { list = {} }
	end
	return all[key]
end

local categoryNames = {}
local function categoryName(id)
	id = ns.PlainNumber(id)
	if not id then
		return nil
	end
	if categoryNames[id] == nil then
		categoryNames[id] = false
		if C_TradeSkillUI.GetCategoryInfo then
			local ok, info = pcall(C_TradeSkillUI.GetCategoryInfo, id)
			if ok and type(info) == "table" then
				categoryNames[id] = ns.PlainString(info.name) or false
			end
		end
	end
	return categoryNames[id] or nil
end

-- Basic reagents of a recipe as { { itemID, count }, ... }.
local function readReagents(recipeID)
	local reagents = {}
	local T = C_TradeSkillUI
	if not T.GetRecipeSchematic then
		return reagents
	end
	local ok, schematic = pcall(T.GetRecipeSchematic, recipeID, false)
	if not ok or type(schematic) ~= "table" or type(schematic.reagentSlotSchematics) ~= "table" then
		return reagents
	end
	local basic = Enum and Enum.CraftingReagentType and Enum.CraftingReagentType.Basic
	for _, slot in ipairs(schematic.reagentSlotSchematics) do
		local isBasic = type(slot) == "table" and (basic == nil or slot.reagentType == nil or slot.reagentType == basic)
		if isBasic and type(slot.reagents) == "table" then
			local first = slot.reagents[1]
			local itemID = type(first) == "table" and ns.PlainNumber(first.itemID)
			if itemID then
				reagents[#reagents + 1] = { itemID, math.max(1, ns.PlainNumber(slot.quantityRequired) or 1) }
			end
		end
	end
	return reagents
end

local function tradeFlag(name)
	local fn = C_TradeSkillUI[name]
	if not fn then
		return false
	end
	local ok, value = pcall(fn)
	return ok and ns.Plain(value) and true or false
end

-- Scan the open profession window into the cache. Returns the profession
-- key when something was stored.
function ns.ScanOpenProfession()
	local T = C_TradeSkillUI
	if not T or not T.GetRecipeInfo or not T.GetBaseProfessionInfo then
		return nil
	end
	-- Someone else's profession (link, guild, NPC crafting): not ours to cache.
	if tradeFlag("IsTradeSkillLinked") or tradeFlag("IsTradeSkillGuild") or tradeFlag("IsNPCCrafting") then
		return nil
	end
	local ok, base = pcall(T.GetBaseProfessionInfo)
	if not ok or type(base) ~= "table" then
		return nil
	end
	local key = ns.KeyFor(base.professionID, base.professionName)
	if not key then
		return nil
	end
	-- GetAllRecipeIDs lists every recipe; GetFilteredRecipeIDs only what the
	-- window's filters show, so missing recipes there may still be known.
	local ids, complete
	for _, fnName in ipairs({ "GetAllRecipeIDs", "GetFilteredRecipeIDs" }) do
		if not ids and T[fnName] then
			local okIDs, result = pcall(T[fnName])
			if okIDs and type(result) == "table" and #result > 0 then
				ids, complete = result, fnName == "GetAllRecipeIDs"
			end
		end
	end
	if not ids then
		return nil
	end

	local cache = ns.GetRecipeCache(key, true)
	local existing = {}
	for _, entry in ipairs(cache.list) do
		existing[entry.id] = entry
	end
	local found, known, unlearned = 0, {}, {}
	for _, rawID in ipairs(ids) do
		local recipeID = ns.PlainNumber(rawID)
		local okInfo, info = false, nil
		if recipeID then
			okInfo, info = pcall(T.GetRecipeInfo, recipeID)
		end
		if okInfo and type(info) == "table" then
			if not ns.Plain(info.learned) then
				unlearned[recipeID] = true
			elseif not ns.Plain(info.isDummyRecipe) and not ns.Plain(info.isGatheringRecipe) and not ns.Plain(info.isRecraft) then
				known[recipeID] = true
				local entry = existing[recipeID]
				if not entry then
					entry = { id = recipeID }
					existing[recipeID] = entry
					cache.list[#cache.list + 1] = entry
				end
				entry.name = ns.PlainString(info.name) or entry.name or ("Recipe " .. recipeID)
				entry.icon = ns.Plain(info.icon) or entry.icon
				entry.diff = DIFF_BY_ENUM[ns.PlainNumber(info.relativeDifficulty) or -1] or "trivial"
				entry.cat = categoryName(info.categoryID) or entry.cat
				if not entry.reagents or #entry.reagents == 0 then
					entry.reagents = readReagents(recipeID)
				end
				if not entry.link and T.GetRecipeLink then
					local okLink, link = pcall(T.GetRecipeLink, recipeID)
					entry.link = okLink and ns.PlainString(link) or nil
				end
				found = found + 1
			end
		end
	end
	-- Nothing answered yet (data still loading): keep what we had.
	if found == 0 and next(unlearned) == nil then
		return nil
	end
	-- Forget recipes you no longer know (for example after dropping and
	-- relearning the profession): anything the game now reports as unlearned,
	-- and, when the list is complete, anything missing from it.
	local kept = {}
	for _, entry in ipairs(cache.list) do
		if known[entry.id] or (not complete and not unlearned[entry.id]) then
			kept[#kept + 1] = entry
		end
	end
	cache.list = kept
	cache.rank = ns.PlainNumber(base.skillLevel) or (ns.byKey and ns.byKey[key] and ns.byKey[key].rank) or cache.rank
	cache.time = time()
	return key
end

-- How many of an item are in your bags.
function ns.ItemCount(itemID, memo)
	if memo and memo[itemID] then
		return memo[itemID]
	end
	local fn = (C_Item and C_Item.GetItemCount) or GetItemCount
	local count = 0
	if fn then
		local ok, value = pcall(fn, itemID)
		count = (ok and ns.PlainNumber(value)) or 0
	end
	if memo then
		memo[itemID] = count
	end
	return count
end

-- How many times a cached recipe can be crafted from what is in your bags
-- right now. `memo` caches item counts across calls in one redraw.
function ns.CraftableCount(recipe, memo)
	local reagents = recipe.reagents
	if not reagents or #reagents == 0 then
		return 0
	end
	local best
	for _, reagent in ipairs(reagents) do
		local can = math.floor(ns.ItemCount(reagent[1], memo) / reagent[2])
		if not best or can < best then
			best = can
		end
	end
	return best or 0
end

-- Item names load asynchronously; until then this returns nil and asks the
-- client to fetch them (ITEM_DATA_LOAD_RESULT then triggers a redraw).
local itemNames, requested = {}, {}
function ns.ItemName(itemID)
	if itemNames[itemID] then
		return itemNames[itemID]
	end
	local name
	if C_Item and C_Item.GetItemNameByID then
		local ok, value = pcall(C_Item.GetItemNameByID, itemID)
		name = ok and ns.PlainString(value) or nil
	end
	if name then
		itemNames[itemID] = name
	elseif not requested[itemID] and C_Item and C_Item.RequestLoadItemDataByID then
		requested[itemID] = true
		pcall(C_Item.RequestLoadItemDataByID, itemID)
	end
	return name
end

---------------------------------------------------------------------------
-- Session gains and your other characters
---------------------------------------------------------------------------

ns.sessionStart = {}

-- Remember each profession's skill the first time it is seen this session.
function ns.TrackSession(list)
	for _, prof in ipairs(list) do
		local start = ns.sessionStart[prof.key]
		if not start or prof.rank < start then
			ns.sessionStart[prof.key] = prof.rank
		end
	end
end

function ns.SessionGain(prof)
	local start = ns.sessionStart[prof.key]
	return start and math.max(0, prof.rank - start) or 0
end

-- An account-wide summary so every character can see the others' skills.
function ns.RememberCharacter(list)
	local name = UnitName and ns.PlainString(UnitName("player"))
	if not name or not ns.db then
		return
	end
	local realm = GetRealmName and ns.PlainString(GetRealmName()) or nil
	local id = realm and (name .. "-" .. realm) or name
	local className, classFile
	if UnitClass then
		className, classFile = UnitClass("player")
	end
	local profs = {}
	for _, prof in ipairs(list) do
		profs[prof.key] = { prof.rank, prof.max, prof.name }
	end
	ns.db.alts[id] = {
		name = name, realm = realm, class = ns.PlainString(classFile), className = ns.PlainString(className),
		level = UnitLevel and ns.PlainNumber(UnitLevel("player")), seen = time(), profs = profs,
	}
	ns.playerID = id
end

-- Your other characters that know a profession, highest skill first.
function ns.AltsWith(key)
	local out = {}
	for id, alt in pairs(ns.db and ns.db.alts or {}) do
		local prof = id ~= ns.playerID and alt.profs and alt.profs[key]
		if prof then
			out[#out + 1] = {
				id = id, name = alt.name or id, class = alt.class, className = alt.className,
				level = alt.level, rank = prof[1], max = prof[2],
			}
		end
	end
	table.sort(out, function(a, b)
		if a.rank ~= b.rank then
			return a.rank > b.rank
		end
		return a.name < b.name
	end)
	return out
end

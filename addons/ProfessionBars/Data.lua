-- Static profession knowledge: ids, base colours, how each rank is learned,
-- gathering nodes and zones. These are Classic values (Forever is built on
-- Classic); anything Forever changes can be corrected here.
local _, ns = ...

-- Skill line ids are stable across every WoW client. `color` is the base
-- colour; the active theme derives the on-screen colour from it.
ns.PROFESSIONS = {
	alchemy = { id = 171, name = "Alchemy", kind = "primary", type = "craft", icon = "Interface\\Icons\\Trade_Alchemy", color = { 0.20, 0.80, 0.46 } },
	blacksmithing = { id = 164, name = "Blacksmithing", kind = "primary", type = "craft", icon = "Interface\\Icons\\Trade_BlackSmithing", color = { 0.93, 0.42, 0.15 } },
	enchanting = { id = 333, name = "Enchanting", kind = "primary", type = "craft", icon = "Interface\\Icons\\Trade_Engraving", color = { 0.64, 0.44, 0.98 } },
	engineering = { id = 202, name = "Engineering", kind = "primary", type = "craft", icon = "Interface\\Icons\\Trade_Engineering", color = { 0.25, 0.78, 0.84 } },
	herbalism = { id = 182, name = "Herbalism", kind = "primary", type = "gather", icon = "Interface\\Icons\\Trade_Herbalism", color = { 0.55, 0.83, 0.24 } },
	leatherworking = { id = 165, name = "Leatherworking", kind = "primary", type = "craft", icon = "Interface\\Icons\\Trade_LeatherWorking", color = { 0.80, 0.61, 0.40 } },
	mining = { id = 186, name = "Mining", kind = "primary", type = "gather", icon = "Interface\\Icons\\Trade_Mining", color = { 0.56, 0.66, 0.80 } },
	skinning = { id = 393, name = "Skinning", kind = "primary", type = "skin", icon = "Interface\\Icons\\INV_Misc_Pelt_Wolf_01", color = { 0.70, 0.43, 0.30 } },
	tailoring = { id = 197, name = "Tailoring", kind = "primary", type = "craft", icon = "Interface\\Icons\\Trade_Tailoring", color = { 0.90, 0.44, 0.76 } },
	cooking = { id = 185, name = "Cooking", kind = "secondary", type = "craft", icon = "Interface\\Icons\\INV_Misc_Food_15", color = { 0.98, 0.74, 0.30 } },
	firstaid = { id = 129, name = "First Aid", kind = "secondary", type = "craft", icon = "Interface\\Icons\\Spell_Holy_SealOfSacrifice", color = { 0.92, 0.26, 0.32 } },
	fishing = { id = 356, name = "Fishing", kind = "secondary", type = "fish", icon = "Interface\\Icons\\Trade_Fishing", color = { 0.24, 0.56, 0.96 } },
}

ns.DEFAULT_COLOR = { 0.62, 0.64, 0.70 }
ns.UNKNOWN_ICON = "Interface\\Icons\\INV_Misc_QuestionMark"

-- How each rank is learned: the skill and character level it needs, and
-- where. Secondary professions learn Expert from a book and Artisan from a
-- quest that needs 225 skill. Classic values; Forever may differ.
ns.RANK_RULES = {
	primary = {
		{ name = "Apprentice", cap = 75 },
		{ name = "Journeyman", cap = 150, skill = 50, level = 10, how = "trainer" },
		{ name = "Expert", cap = 225, skill = 125, level = 20, how = "trainer" },
		{ name = "Artisan", cap = 300, skill = 200, level = 35, how = "artisan" },
	},
	secondary = {
		{ name = "Apprentice", cap = 75 },
		{ name = "Journeyman", cap = 150, skill = 50, level = 10, how = "trainer" },
		{ name = "Expert", cap = 225, skill = 125, level = 0, how = "book" },
		{ name = "Artisan", cap = 300, skill = 225, level = 35, how = "quest" },
	},
}

-- The books and quests behind the secondary professions' higher ranks.
ns.RANK_SOURCES = {
	cooking = { Expert = "Expert Cookbook", Artisan = "Clamlette Surprise" },
	firstaid = { Expert = "Expert First Aid - Under Wraps", Artisan = "Triage" },
	fishing = { Expert = "Expert Fishing - The Bass and You", Artisan = "Nat Pagle, Angler Extreme" },
}

-- The game's own recipe/node difficulty colours.
ns.DIFFICULTY = {
	optimal = { 1.00, 0.50, 0.25 }, -- orange: always a skill-up
	medium = { 1.00, 1.00, 0.00 }, -- yellow: usually
	easy = { 0.25, 0.75, 0.25 }, -- green: sometimes
	trivial = { 0.55, 0.55, 0.55 }, -- grey: never
	locked = { 0.42, 0.42, 0.46 },
}
ns.DIFFICULTY_ORDER = { optimal = 1, medium = 2, easy = 3, trivial = 4, locked = 5 }
ns.DIFFICULTY_WORD = { optimal = "orange", medium = "yellow", easy = "green", trivial = "grey" }
ns.CHANCE_WORD = { optimal = "always", medium = "usually", easy = "sometimes", trivial = "never" }

-- Gathering nodes: { skill, name }. `note` explains a special node; nodes
-- with `special` are rare spawns or grow in one zone only, so they are never
-- suggested as farming targets.
ns.GATHER = {
	mining = {
		{ 1, "Copper Vein" }, { 65, "Tin Vein" },
		{ 65, "Incendicite Vein", special = true, note = "Wetlands only" },
		{ 75, "Silver Vein", special = true, note = "rare, in Tin spots" },
		{ 75, "Lesser Bloodstone", special = true, note = "Arathi only" },
		{ 125, "Iron Deposit" },
		{ 150, "Indurium Vein", special = true, note = "Uldaman only" },
		{ 155, "Gold Vein", special = true, note = "rare, in Iron spots" },
		{ 175, "Mithril Deposit" },
		{ 230, "Truesilver Deposit", special = true, note = "rare, in Mithril spots" },
		{ 230, "Dark Iron Deposit", special = true, note = "Blackrock area only" },
		{ 245, "Small Thorium Vein" }, { 275, "Rich Thorium Vein" },
	},
	herbalism = {
		{ 1, "Peacebloom" }, { 1, "Silverleaf" }, { 15, "Earthroot" }, { 50, "Mageroyal" },
		{ 70, "Briarthorn" }, { 85, "Stranglekelp", note = "underwater" }, { 100, "Bruiseweed" },
		{ 115, "Wild Steelbloom" }, { 120, "Grave Moss", note = "graveyards" }, { 125, "Kingsblood" },
		{ 150, "Liferoot" }, { 160, "Fadeleaf" }, { 170, "Goldthorn" },
		{ 185, "Khadgar's Whisker" }, { 195, "Wintersbite", special = true, note = "Alterac only" },
		{ 205, "Firebloom" }, { 210, "Purple Lotus" }, { 220, "Arthas' Tears" }, { 230, "Sungrass" },
		{ 235, "Blindweed" }, { 245, "Ghost Mushroom", note = "caves" }, { 250, "Gromsblood" },
		{ 260, "Golden Sansam" }, { 270, "Dreamfoil" }, { 280, "Mountain Silversage" },
		{ 285, "Plaguebloom" }, { 290, "Icecap", special = true, note = "Winterspring only" },
		{ 300, "Black Lotus", special = true, note = "rare" },
	},
}

-- Where to gather at each skill bracket (Classic routes): { from, to, what, zones }.
-- Forever's new zones aren't included yet.
ns.GATHER_ZONES = {
	mining = {
		{ 1, 65, "Copper", { "Elwynn Forest", "Dun Morogh", "Durotar", "Mulgore", "Tirisfal Glades", "Darkshore" } },
		{ 65, 125, "Tin", { "Redridge Mountains", "Wetlands", "Hillsbrad Foothills", "The Barrens", "Stonetalon Mountains", "Ashenvale", "Duskwood" } },
		{ 125, 175, "Iron", { "Arathi Highlands", "Desolace", "Stranglethorn Vale", "Thousand Needles", "Alterac Mountains", "Dustwallow Marsh" } },
		{ 175, 245, "Mithril", { "Badlands", "Searing Gorge", "The Hinterlands", "Tanaris", "Feralas", "Azshara" } },
		{ 245, 301, "Thorium", { "Un'Goro Crater", "Burning Steppes", "Western Plaguelands", "Eastern Plaguelands", "Winterspring", "Silithus" } },
	},
	herbalism = {
		{ 1, 70, "Peacebloom, Silverleaf, Earthroot, Mageroyal", { "Elwynn Forest", "Dun Morogh", "Durotar", "Mulgore", "Tirisfal Glades", "Teldrassil", "Westfall", "Loch Modan", "Darkshore", "The Barrens", "Silverpine Forest" } },
		{ 70, 125, "Briarthorn, Bruiseweed, Kingsblood", { "Redridge Mountains", "Wetlands", "Hillsbrad Foothills", "Ashenvale", "Stonetalon Mountains", "Duskwood" } },
		{ 125, 205, "Liferoot, Fadeleaf, Goldthorn, Khadgar's Whisker", { "Arathi Highlands", "Stranglethorn Vale", "Swamp of Sorrows", "Dustwallow Marsh", "The Hinterlands" } },
		{ 205, 270, "Firebloom, Purple Lotus, Sungrass, Gromsblood", { "Tanaris", "Feralas", "The Hinterlands", "Searing Gorge", "Felwood" } },
		{ 270, 301, "Dreamfoil, Mountain Silversage, Plaguebloom", { "Un'Goro Crater", "Felwood", "Western Plaguelands", "Eastern Plaguelands", "Winterspring" } },
	},
}

-- Classic zone level ranges, used to say which zones suit your level.
ns.ZONE_LEVELS = {
	["Elwynn Forest"] = { 1, 10 }, ["Dun Morogh"] = { 1, 10 }, ["Durotar"] = { 1, 10 }, ["Mulgore"] = { 1, 10 },
	["Tirisfal Glades"] = { 1, 10 }, ["Teldrassil"] = { 1, 10 },
	["Westfall"] = { 10, 20 }, ["Loch Modan"] = { 10, 20 }, ["Darkshore"] = { 10, 20 }, ["Silverpine Forest"] = { 10, 20 },
	["The Barrens"] = { 10, 25 }, ["Redridge Mountains"] = { 15, 25 }, ["Stonetalon Mountains"] = { 15, 27 },
	["Ashenvale"] = { 18, 30 }, ["Duskwood"] = { 18, 30 }, ["Wetlands"] = { 20, 30 }, ["Hillsbrad Foothills"] = { 20, 30 },
	["Thousand Needles"] = { 25, 35 }, ["Alterac Mountains"] = { 30, 40 }, ["Arathi Highlands"] = { 30, 40 },
	["Desolace"] = { 30, 40 }, ["Stranglethorn Vale"] = { 30, 45 }, ["Dustwallow Marsh"] = { 35, 45 },
	["Badlands"] = { 35, 45 }, ["Swamp of Sorrows"] = { 35, 45 }, ["Feralas"] = { 40, 50 },
	["The Hinterlands"] = { 40, 50 }, ["Tanaris"] = { 40, 50 }, ["Searing Gorge"] = { 43, 50 },
	["Azshara"] = { 45, 55 }, ["Felwood"] = { 48, 55 }, ["Un'Goro Crater"] = { 48, 55 },
	["Burning Steppes"] = { 50, 58 }, ["Western Plaguelands"] = { 51, 58 }, ["Eastern Plaguelands"] = { 53, 60 },
	["Winterspring"] = { 53, 60 }, ["Silithus"] = { 55, 60 },
}

-- The rank rules that apply to a profession.
function ns.RankRules(prof)
	return ns.RANK_RULES[(prof and prof.kind == "secondary") and "secondary" or "primary"]
end

-- Index of the rank whose cap matches the current maximum skill.
function ns.RankIndex(rules, maxSkill)
	for i, rank in ipairs(rules) do
		if maxSkill <= rank.cap then
			return i
		end
	end
	return #rules
end

-- What it takes to reach the next rank.
-- state: "train" (go learn it now), "skill" (need more skill), "level"
-- (need a higher character level), "grind" (last rank, not maxed yet) or
-- "mastered".
function ns.NextStep(prof, level)
	local rules = ns.RankRules(prof)
	local index = ns.RankIndex(rules, prof.max)
	local step = { index = index, rules = rules, rank = rules[index], nextRank = rules[index + 1] }
	local nextRank = step.nextRank
	if not nextRank then
		step.needSkill = math.max(0, prof.max - prof.rank)
		step.state = step.needSkill == 0 and "mastered" or "grind"
		return step
	end
	step.needSkill = math.max(0, nextRank.skill - prof.rank)
	step.needLevel = math.max(0, nextRank.level - level)
	if step.needSkill == 0 and step.needLevel == 0 then
		step.state = "train"
	elseif step.needSkill > 0 then
		step.state = "skill"
	else
		step.state = "level"
	end
	return step
end

-- Where a rank is learned, in a few words ("a city trainer", "the book
-- Expert Cookbook", "the quest Triage").
function ns.HowToLearn(prof, rank)
	local source = ns.RANK_SOURCES[prof.key] and ns.RANK_SOURCES[prof.key][rank.name]
	if rank.how == "book" then
		return source and ("the book " .. source) or "a book"
	elseif rank.how == "quest" then
		return source and ("the quest " .. source) or "a quest"
	elseif rank.how == "artisan" then
		return "an Artisan trainer (only a few teach it)"
	end
	return prof.kind == "secondary" and "a trainer" or "a city trainer"
end

ns.GOOD = { 0.35, 1.00, 0.45 }
ns.BAD = { 1.00, 0.45, 0.40 }
ns.GOLD = { 1.00, 0.82, 0.00 }
ns.WHITE = { 1.00, 1.00, 1.00 }
ns.GREY = { 0.62, 0.62, 0.66 }
ns.DIM = { 0.42, 0.42, 0.46 }

-- How the next rank is learned, as the end of a short sentence.
local function learnPhrase(rank)
	if rank.how == "book" then
		return "read the " .. rank.name .. " book"
	elseif rank.how == "quest" then
		return "do the " .. rank.name .. " quest"
	end
	return "train " .. rank.name
end

-- A one-line headline (bars and the panel), a longer tooltip sentence and a
-- colour for a step.
function ns.DescribeStep(step, prof)
	local nextRank = step.nextRank
	if step.state == "train" then
		return "Learn " .. nextRank.name .. " now",
			"Learn " .. nextRank.name .. " from " .. ns.HowToLearn(prof, nextRank) .. ". It raises your cap to " .. nextRank.cap .. ".", ns.GOOD
	elseif step.state == "skill" then
		return step.needSkill .. " to " .. nextRank.skill .. ", then " .. learnPhrase(nextRank),
			step.needSkill .. " more skill, then learn " .. nextRank.name .. " from " .. ns.HowToLearn(prof, nextRank) .. ".", ns.WHITE
	elseif step.state == "level" then
		return "Reach level " .. nextRank.level .. " for " .. nextRank.name,
			"You have the skill for " .. nextRank.name .. "; it needs character level " .. nextRank.level .. ".", ns.BAD
	elseif step.state == "grind" then
		return step.needSkill .. " to " .. prof.max, step.needSkill .. " points left to reach " .. prof.max .. ".", ns.WHITE
	end
	return "Mastered", "Every rank learned and maxed.", ns.GOLD
end

-- Expected skill-up colour for gathering a node that needs `req` skill.
function ns.GatherDifficulty(skill, req)
	local diff = skill - req
	if diff < 0 then
		return "locked"
	elseif diff < 25 then
		return "optimal"
	elseif diff < 50 then
		return "medium"
	elseif diff < 100 then
		return "easy"
	end
	return "trivial"
end

-- Classic skinning: level 1-10 mobs need 1 skill, 11-20 need (level-10)*10,
-- above that level*5.
function ns.SkinReq(level)
	if level <= 10 then
		return 1
	elseif level <= 20 then
		return (level - 10) * 10
	end
	return level * 5
end

function ns.SkinMaxLevel(skill)
	if skill < 100 then
		return math.floor(skill / 10) + 10
	end
	return math.floor(skill / 5)
end

-- Shorter labels for long recipe names ("Enchant Bracer - Minor Stamina").
function ns.ShortRecipeName(name)
	return (name:gsub("^Enchant ", ""))
end

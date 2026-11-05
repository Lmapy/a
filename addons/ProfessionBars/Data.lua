-- Static profession knowledge: ids, base colours, rank thresholds and
-- gathering tables. Everything here is plain data plus small pure helpers.
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

-- Classic rank caps and what it takes to *learn* each rank (skill and
-- character level). Forever is built on Classic; edit here if it differs.
ns.RANKS = {
	{ name = "Apprentice", cap = 75 },
	{ name = "Journeyman", cap = 150, skill = 50, level = 10 },
	{ name = "Expert", cap = 225, skill = 125, level = 20 },
	{ name = "Artisan", cap = 300, skill = 200, level = 35 },
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

-- Classic gathering requirements: { skill, name }. `rare` marks nodes that
-- only spawn in one zone or dungeon; the map lists them in the tooltip.
ns.GATHER = {
	mining = {
		{ 1, "Copper Vein" }, { 65, "Tin Vein" }, { 65, "Incendicite Vein", rare = true },
		{ 75, "Silver Vein" }, { 75, "Lesser Bloodstone", rare = true }, { 125, "Iron Deposit" },
		{ 150, "Indurium Vein", rare = true }, { 155, "Gold Vein" }, { 175, "Mithril Deposit" },
		{ 230, "Truesilver Deposit" }, { 230, "Dark Iron Deposit", rare = true },
		{ 245, "Small Thorium Vein" }, { 275, "Rich Thorium Vein" },
	},
	herbalism = {
		{ 1, "Peacebloom" }, { 1, "Silverleaf" }, { 15, "Earthroot" }, { 50, "Mageroyal" },
		{ 70, "Briarthorn" }, { 85, "Stranglekelp" }, { 100, "Bruiseweed" },
		{ 115, "Wild Steelbloom" }, { 120, "Grave Moss" }, { 125, "Kingsblood" },
		{ 150, "Liferoot" }, { 160, "Fadeleaf" }, { 170, "Goldthorn" },
		{ 185, "Khadgar's Whisker" }, { 195, "Wintersbite" }, { 205, "Firebloom" },
		{ 210, "Purple Lotus" }, { 220, "Arthas' Tears" }, { 230, "Sungrass" },
		{ 235, "Blindweed" }, { 245, "Ghost Mushroom" }, { 250, "Gromsblood" },
		{ 260, "Golden Sansam" }, { 270, "Dreamfoil" }, { 280, "Mountain Silversage" },
		{ 285, "Plaguebloom" }, { 290, "Icecap" }, { 300, "Black Lotus" },
	},
}

-- Index of the rank whose cap matches the current maximum skill.
function ns.RankIndex(maxSkill)
	for i, rank in ipairs(ns.RANKS) do
		if maxSkill <= rank.cap then
			return i
		end
	end
	return #ns.RANKS
end

-- What it takes to reach the next rank.
-- state: "train" (go learn it now), "skill" (need more skill), "level"
-- (need a higher character level), "grind" (last rank, not maxed yet) or
-- "mastered".
function ns.NextStep(rank, maxSkill, level)
	local index = ns.RankIndex(maxSkill)
	local step = { index = index, rank = ns.RANKS[index], nextRank = ns.RANKS[index + 1] }
	local nextRank = step.nextRank
	if not nextRank then
		step.needSkill = math.max(0, maxSkill - rank)
		step.state = step.needSkill == 0 and "mastered" or "grind"
		return step
	end
	step.needSkill = math.max(0, nextRank.skill - rank)
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

ns.GOOD = { 0.35, 1.00, 0.45 }
ns.BAD = { 1.00, 0.45, 0.40 }
ns.GOLD = { 1.00, 0.82, 0.00 }
ns.WHITE = { 1.00, 1.00, 1.00 }
ns.GREY = { 0.62, 0.62, 0.66 }
ns.DIM = { 0.42, 0.42, 0.46 }

-- Short (bar/branch sized) and long (tooltip sized) wording for a step.
function ns.DescribeStep(step, prof)
	local nextRank = step.nextRank
	if step.state == "train" then
		return "Train " .. nextRank.name, "Ready to learn " .. nextRank.name .. " " .. prof.name .. " (cap " .. nextRank.cap .. ").", ns.GOOD
	elseif step.state == "skill" then
		return step.needSkill .. " to " .. nextRank.name, step.needSkill .. " more skill to unlock " .. nextRank.name .. " training.", ns.WHITE
	elseif step.state == "level" then
		return "Need level " .. nextRank.level, "Reach character level " .. nextRank.level .. " to learn " .. nextRank.name .. ".", ns.BAD
	elseif step.state == "grind" then
		return step.needSkill .. " to max", step.needSkill .. " points left to reach " .. prof.max .. ".", ns.WHITE
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

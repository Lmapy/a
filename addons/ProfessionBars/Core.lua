-- Saved settings, events, slash commands, key binding and the minimap
-- addon-compartment entry.
local ADDON, ns = ...

local DEFAULTS = {
	shown = true,
	locked = false,
	theme = "vibrant",
	scale = 1,
	width = 220,
	mapScale = 1,
	hideInCombat = false,
	colors = {}, -- [key] = { r, g, b } custom colours
	expanded = {}, -- [key][branch] = true/false
	alts = {}, -- ["Name-Realm"] = { class, level, profs = { [key] = { rank, max } } }
	welcomed = false,
}
-- What /pb reset puts back (it keeps your alts and recipe data).
local LOOK = { "shown", "locked", "theme", "scale", "width", "mapScale", "hideInCombat", "colors", "expanded" }
local CHAR_DEFAULTS = {
	hidden = {}, -- [key] = true for professions without a bar
	recipes = {}, -- [key] = { list = {...}, rank = n, time = t }
}

local function fill(db, defaults)
	for key, value in pairs(defaults) do
		if db[key] == nil then
			db[key] = type(value) == "table" and {} or value
		end
	end
	return db
end

local function initDB()
	if ns.db then
		return
	end
	ProfessionBarsDB = fill(ProfessionBarsDB or {}, DEFAULTS)
	ProfessionBarsCharDB = fill(ProfessionBarsCharDB or {}, CHAR_DEFAULTS)
	ns.db, ns.char = ProfessionBarsDB, ProfessionBarsCharDB
end

---------------------------------------------------------------------------
-- State changes (used by the UI and by slash commands)
---------------------------------------------------------------------------

function ns.UpdateProfessions()
	ns.profs = ns.ScanProfessions()
	ns.byKey = {}
	for _, prof in ipairs(ns.profs) do
		ns.byKey[prof.key] = prof
	end
	ns.TrackSession(ns.profs)
	ns.RememberCharacter(ns.profs)
	ns.Bars:Refresh()
	ns.Map:OnProfessionsChanged()
end

function ns.UpdateRecipes()
	local key = ns.ScanOpenProfession()
	if key then
		ns.Map:Refresh()
	end
end

function ns.RefreshAll()
	ns.Bars:Refresh()
	ns.Map:Refresh()
end

function ns.ApplyVisibility()
	local show = ns.db.shown and not (ns.db.hideInCombat and ns.inCombat)
	ns.Bars:SetShown(show)
	if not show then
		ns.Map:Close()
	end
end

function ns.Toggle(show)
	if show == nil then
		show = not ns.db.shown
	end
	ns.db.shown = show and true or false
	ns.ApplyVisibility()
end

function ns.SetLocked(locked)
	ns.db.locked = locked and true or false
	ns.Bars:UpdateHeader()
end

function ns.SetTheme(key)
	if not ns.ThemeIndex(key) then
		return false
	end
	ns.db.theme = key
	ns.RefreshAll()
	return true
end

function ns.SetHidden(key, hidden)
	ns.char.hidden[key] = hidden and true or nil
	if hidden and ns.Map:IsOpenFor(key) then
		ns.Map:Close()
	end
	ns.Bars:Refresh()
end

---------------------------------------------------------------------------
-- Events
---------------------------------------------------------------------------

local events = CreateFrame("Frame")

-- Some events only exist on some clients; registering an unknown one errors.
local function listen(name)
	return pcall(events.RegisterEvent, events, name)
end

local PROFESSION_EVENTS = {
	"SKILL_LINES_CHANGED", "CHAT_MSG_SKILL", "LEARNED_SPELL_IN_SKILL_LINE", "LEARNED_SPELL_IN_TAB",
	"PLAYER_LEVEL_UP", "TRADE_SKILL_LIST_UPDATE",
}
local RECIPE_EVENTS = { "TRADE_SKILL_SHOW", "TRADE_SKILL_LIST_UPDATE", "TRADE_SKILL_DATA_SOURCE_CHANGED" }
local OTHER_EVENTS = {
	"BAG_UPDATE_DELAYED", "PLAYER_REGEN_DISABLED", "PLAYER_REGEN_ENABLED",
	"ITEM_DATA_LOAD_RESULT", "GET_ITEM_INFO_RECEIVED",
}

local handlers = {}
local isProfessionEvent, isRecipeEvent = {}, {}
for _, name in ipairs(PROFESSION_EVENTS) do
	isProfessionEvent[name] = true
end
for _, name in ipairs(RECIPE_EVENTS) do
	isRecipeEvent[name] = true
end

function handlers.ADDON_LOADED(name)
	if name == ADDON then
		initDB()
	end
end

function handlers.PLAYER_LOGIN()
	initDB()
	ns.inCombat = InCombatLockdown and InCombatLockdown() or false
	ns.Bars:Create()
	ns.Map:Create()
	for _, list in ipairs({ PROFESSION_EVENTS, RECIPE_EVENTS, OTHER_EVENTS }) do
		for _, name in ipairs(list) do
			listen(name)
		end
	end
	ns.UpdateProfessions()
	ns.ApplyVisibility()
	-- Some clients fill in profession data a moment after login.
	ns.Throttle("professions", 3, ns.UpdateProfessions)
	if not ns.db.welcomed then
		ns.db.welcomed = true
		ns.Print("click a bar to open that profession's map. Hover the bars for options, /pb help for commands.")
	end
end

local function refreshOpenMap()
	if ns.Map.current then
		ns.Throttle("the map", 0.5, function() ns.Map:Refresh() end)
	end
end

-- Bag counts drive "ready to craft" and the plan; item names load lazily.
handlers.BAG_UPDATE_DELAYED = refreshOpenMap
handlers.ITEM_DATA_LOAD_RESULT = refreshOpenMap
handlers.GET_ITEM_INFO_RECEIVED = refreshOpenMap

function handlers.PLAYER_REGEN_DISABLED()
	ns.inCombat = true
	if ns.db.hideInCombat then
		ns.ApplyVisibility()
	end
end

function handlers.PLAYER_REGEN_ENABLED()
	ns.inCombat = false
	if ns.db.hideInCombat then
		ns.ApplyVisibility()
	end
end

local function onEvent(event, ...)
	local handler = handlers[event]
	if handler then
		handler(...)
	end
	if not ns.db or not ns.Bars.frame then
		return
	end
	if isProfessionEvent[event] then
		ns.Throttle("professions", 0.3, ns.UpdateProfessions)
	end
	if isRecipeEvent[event] then
		ns.Throttle("recipes", 0.4, ns.UpdateRecipes)
	end
end

events:SetScript("OnEvent", function(_, event, ...)
	ns.SafeCall("handling " .. event, onEvent, event, ...)
end)
events:RegisterEvent("ADDON_LOADED")
events:RegisterEvent("PLAYER_LOGIN")

---------------------------------------------------------------------------
-- Slash commands
---------------------------------------------------------------------------

local function findProfession(query, includeUnlearned)
	query = query:lower()
	for _, prof in ipairs(ns.profs or {}) do
		if prof.key == query or prof.name:lower() == query then
			return prof.key, prof.name
		end
	end
	for _, prof in ipairs(ns.profs or {}) do
		if prof.key:find(query, 1, true) or prof.name:lower():find(query, 1, true) then
			return prof.key, prof.name
		end
	end
	if includeUnlearned then
		for key, prof in pairs(ns.PROFESSIONS) do
			if key:find(query, 1, true) or prof.name:lower():find(query, 1, true) then
				return key, prof.name
			end
		end
	end
	return nil
end

local function clampNumber(text, low, high)
	local value = tonumber(text)
	if not value then
		return nil
	end
	return math.max(low, math.min(high, value))
end

local HELP = {
	"|cffffd100/pb|r  show or hide the bars",
	"|cffffd100/pb map <profession>|r  open a profession's mind map",
	"|cffffd100/pb edit|r  pick which professions get a bar (or hover the bars and click 'edit')",
	"|cffffd100/pb hide <profession>|r, |cffffd100/pb show <profession|all>|r",
	"|cffffd100/pb theme [vibrant|soft|midnight|mono]|r  set or cycle the colour theme",
	"|cffffd100/pb color <profession> <#rrggbb|reset>|r  your own colour for one profession",
	"|cffffd100/pb lock|r, |cffffd100/pb unlock|r  stop or allow dragging",
	"|cffffd100/pb scale <0.5-2>|r, |cffffd100/pb mapscale <0.5-2>|r, |cffffd100/pb width <140-400>|r",
	"|cffffd100/pb combat|r  hide the bars during combat (toggle)",
	"|cffffd100/pb alts|r, |cffffd100/pb forget <name>|r  your other characters' professions",
	"|cffffd100/pb reset|r  reset position and look (keeps alts and recipes)",
	"|cffffd100/pb debug|r  show what the addon can read (handy for bug reports)",
}

local function yesNo(value)
	return value and "|cff55ff55yes|r" or "|cffff5555no|r"
end

local function printDebug()
	local version, build, _, interface = "?", "?", nil, "?"
	if GetBuildInfo then
		version, build, _, interface = GetBuildInfo()
	end
	ns.Print("client " .. tostring(version) .. " (" .. tostring(build) .. "), interface " .. tostring(interface))
	local T = C_TradeSkillUI
	ns.Print("GetProfessions " .. yesNo(type(GetProfessions) == "function")
		.. ", GetProfessionInfo " .. yesNo(type(GetProfessionInfo) == "function")
		.. ", GetSkillLineInfo " .. yesNo(type(GetSkillLineInfo) == "function"))
	ns.Print("C_TradeSkillUI " .. yesNo(T) .. (T and (": GetChildProfessionInfos " .. yesNo(T.GetChildProfessionInfos)
		.. ", GetProfessionInfoBySkillLineID " .. yesNo(T.GetProfessionInfoBySkillLineID)
		.. ", GetAllRecipeIDs " .. yesNo(T.GetAllRecipeIDs)
		.. ", GetRecipeInfo " .. yesNo(T.GetRecipeInfo)
		.. ", GetRecipeSchematic " .. yesNo(T.GetRecipeSchematic)) or ""))
	ns.Print("read from: " .. (#ns.scanSources > 0 and table.concat(ns.scanSources, " + ") or "nothing"))
	local others = 0
	for id in pairs(ns.db.alts) do
		if id ~= ns.playerID then
			others = others + 1
		end
	end
	ns.Print("other characters remembered: " .. others .. (ns.lastError and ("; last error: " .. ns.lastError) or "; no errors"))
	for _, prof in ipairs(ns.profs or {}) do
		local cache = ns.GetRecipeCache(prof.key)
		ns.Print(string.format("  %s (%s, id %s): %d/%d%s, %s", prof.name, prof.key, tostring(prof.skillLine),
			prof.rank, prof.max, prof.modifier > 0 and (" +" .. prof.modifier) or "",
			cache and (#cache.list .. " recipes cached") or "no recipes cached"))
	end
end

local commands = {}

function commands.help()
	ns.Print("commands:")
	for _, line in ipairs(HELP) do
		DEFAULT_CHAT_FRAME:AddMessage("  " .. line)
	end
end

function commands.show(arg)
	if arg == "" then
		ns.Toggle(true)
	elseif arg == "all" then
		for key in pairs(ns.char.hidden) do
			ns.char.hidden[key] = nil
		end
		ns.Bars:Refresh()
		ns.Print("every profession has a bar again.")
	else
		local key, name = findProfession(arg)
		if not key then
			return ns.Print("no learned profession matches '" .. arg .. "'.")
		end
		ns.SetHidden(key, false)
		ns.Print(name .. " bar shown.")
	end
end

function commands.hide(arg)
	if arg == "" then
		return ns.Toggle(false)
	end
	local key, name = findProfession(arg)
	if not key then
		return ns.Print("no learned profession matches '" .. arg .. "'.")
	end
	ns.SetHidden(key, true)
	ns.Print(name .. " bar hidden. /pb show " .. arg .. " brings it back.")
end

function commands.toggle()
	ns.Toggle()
end

function commands.map(arg)
	local key
	if arg == "" then
		key = ns.profs and ns.profs[1] and ns.profs[1].key
	else
		key = findProfession(arg)
	end
	if not key then
		return ns.Print(arg == "" and "no professions learned yet." or ("no learned profession matches '" .. arg .. "'."))
	end
	ns.Toggle(true)
	ns.Map:Open(key)
end

function commands.edit()
	ns.Toggle(true)
	ns.Bars:SetEditing(not ns.Bars.editing)
end

function commands.theme(arg)
	local key = arg ~= "" and arg or ns.NextThemeKey()
	if not ns.SetTheme(key) then
		local names = {}
		for _, theme in ipairs(ns.THEMES) do
			names[#names + 1] = theme.key
		end
		return ns.Print("themes: " .. table.concat(names, ", "))
	end
	ns.Print("theme: " .. ns.GetTheme().label)
end

function commands.color(arg)
	local query, value = arg:match("^(.-)%s+(%S+)$")
	local key, name
	if query and query ~= "" then
		key, name = findProfession(query, true)
	end
	if not key then
		return ns.Print("usage: /pb color <profession> <#rrggbb|reset>")
	end
	if value == "reset" then
		ns.db.colors[key] = nil
	else
		local r, g, b = ns.ParseHex(value)
		if not r then
			return ns.Print("colours look like #ff8800.")
		end
		ns.db.colors[key] = { r, g, b }
	end
	ns.RefreshAll()
	ns.Print(name .. " colour " .. (ns.db.colors[key] and "set" or "reset") .. ".")
end

function commands.lock()
	ns.SetLocked(true)
	ns.Print("bars locked.")
end

function commands.unlock()
	ns.SetLocked(false)
	ns.Print("bars unlocked - hover them and drag the header.")
end

function commands.scale(arg)
	local value = clampNumber(arg, 0.5, 2)
	if not value then
		return ns.Print("usage: /pb scale 0.5-2 (now " .. ns.db.scale .. ")")
	end
	ns.db.scale = value
	ns.Bars:Refresh()
end

function commands.mapscale(arg)
	local value = clampNumber(arg, 0.5, 2)
	if not value then
		return ns.Print("usage: /pb mapscale 0.5-2 (now " .. ns.db.mapScale .. ")")
	end
	ns.db.mapScale = value
	ns.Map:Refresh()
end

function commands.width(arg)
	local value = clampNumber(arg, 140, 400)
	if not value then
		return ns.Print("usage: /pb width 140-400 (now " .. ns.db.width .. ")")
	end
	ns.db.width = math.floor(value)
	ns.Bars:Refresh()
end

function commands.combat()
	ns.db.hideInCombat = not ns.db.hideInCombat
	ns.ApplyVisibility()
	ns.Print("hide in combat: " .. (ns.db.hideInCombat and "on" or "off"))
end

function commands.alts()
	local count = 0
	for id, alt in pairs(ns.db.alts) do
		if id ~= ns.playerID then
			local parts = {}
			for _, prof in pairs(alt.profs or {}) do
				parts[#parts + 1] = (prof[3] or "?") .. " " .. prof[1] .. "/" .. prof[2]
			end
			table.sort(parts)
			ns.Print("  " .. id .. ": " .. (#parts > 0 and table.concat(parts, ", ") or "no professions"))
			count = count + 1
		end
	end
	if count == 0 then
		ns.Print("no other characters yet - log in on them once with the addon enabled.")
	else
		ns.Print("/pb forget <name> removes one.")
	end
end

function commands.forget(arg)
	local query = arg:lower()
	for id, alt in pairs(ns.db.alts) do
		if query ~= "" and id ~= ns.playerID and (id:lower() == query or (alt.name or ""):lower() == query) then
			ns.db.alts[id] = nil
			ns.RefreshAll()
			return ns.Print("forgot " .. id .. ".")
		end
	end
	ns.Print("usage: /pb forget <character name> (see /pb alts)")
end

function commands.reset()
	for _, key in ipairs(LOOK) do
		local value = DEFAULTS[key]
		ns.db[key] = type(value) == "table" and {} or value
	end
	ns.db.point, ns.db.mapPoint = nil, nil
	ns.Bars:RestorePosition()
	ns.Map:RestorePosition()
	ns.ApplyVisibility()
	ns.RefreshAll()
	ns.Print("position and look reset.")
end

function commands.debug()
	printDebug()
end

local function handleSlash(message)
	local command, arg = (message or ""):match("^%s*(%S*)%s*(.-)%s*$")
	command = (command or ""):lower()
	if command == "" then
		return commands.toggle()
	end
	local fn = commands[command] or commands[command:gsub("colour", "color")]
	if fn then
		return fn(arg or "")
	end
	-- "/pb mining" opens that profession's map.
	if findProfession(message) then
		return commands.map(message:match("^%s*(.-)%s*$"))
	end
	commands.help()
end

SLASH_PROFESSIONBARS1 = "/pb"
SLASH_PROFESSIONBARS2 = "/profbars"
SlashCmdList.PROFESSIONBARS = function(message)
	ns.SafeCall("running /pb " .. tostring(message), handleSlash, message)
end

---------------------------------------------------------------------------
-- Key binding and minimap addon compartment
---------------------------------------------------------------------------

BINDING_HEADER_PROFESSIONBARS = "Profession Bars"
BINDING_NAME_PROFESSIONBARS_TOGGLE = "Show/hide profession bars"

function ProfessionBars_Toggle()
	ns.Toggle()
end

function ProfessionBars_OnAddonCompartmentClick()
	ns.Toggle()
end

function ProfessionBars_OnAddonCompartmentEnter(_, button)
	GameTooltip:SetOwner(button, "ANCHOR_LEFT")
	GameTooltip:AddLine("Profession Bars", 1, 1, 1)
	GameTooltip:AddLine("Click to show or hide. Type /pb help for options.", 0.65, 0.65, 0.7)
	GameTooltip:Show()
end

function ProfessionBars_OnAddonCompartmentLeave()
	GameTooltip:Hide()
end

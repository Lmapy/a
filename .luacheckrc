-- Lint config for the WoW addons in addons/ and the headless mock in tools/.
std = "lua51"
max_line_length = false
unused_args = false

files["addons/"] = {
	globals = {
		"ProfessionBarsDB", "ProfessionBarsCharDB", "SlashCmdList", "UISpecialFrames",
		"SLASH_PROFESSIONBARS1", "SLASH_PROFESSIONBARS2",
		"BINDING_HEADER_PROFESSIONBARS", "BINDING_NAME_PROFESSIONBARS_TOGGLE",
		"ProfessionBars_Toggle", "ProfessionBars_OnAddonCompartmentClick",
		"ProfessionBars_OnAddonCompartmentEnter", "ProfessionBars_OnAddonCompartmentLeave",
	},
	read_globals = {
		"CreateFrame", "UIParent", "GameTooltip", "DEFAULT_CHAT_FRAME", "C_Timer", "C_TradeSkillUI",
		"C_Item", "Enum", "GetProfessions", "GetProfessionInfo", "GetNumSkillLines", "GetSkillLineInfo",
		"GetItemCount", "UnitLevel", "InCombatLockdown", "IsShiftKeyDown", "GetBuildInfo",
		"ChatEdit_InsertLink", "ChatFrameUtil", "issecretvalue", "GameFontHighlightSmall", "time", "GetTime",
		"UnitName", "UnitClass", "GetRealmName", "RAID_CLASS_COLORS", "CreateColor",
	},
}
files["tools/"] = { globals = { "_G" }, allow_defined_top = true }

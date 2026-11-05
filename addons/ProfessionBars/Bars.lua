-- The bar stack: one colour-themed bar per profession, in the minimal
-- "clean" style (default), the framed "classic" style or the compact "slim"
-- style. Click a bar to open its mind map. Hovering reveals a small header
-- with controls.
local _, ns = ...

local Bars = { bars = {} }
ns.Bars = Bars

local BAR_H, HEADER_H = 18, 16
local FLAT = "Interface\\Buttons\\WHITE8X8"
local SWATCH_KEYS = { "blacksmithing", "alchemy", "enchanting", "fishing" }

local function playerLevel()
	return (UnitLevel and UnitLevel("player")) or 1
end

---------------------------------------------------------------------------
-- Header (only visible while hovering or editing)
---------------------------------------------------------------------------

local function TextButton(parent, width, onClick, tooltip)
	local button = CreateFrame("Button", nil, parent)
	button:SetSize(width, HEADER_H)
	button.text = button:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	button.text:SetPoint("CENTER")
	button.text:SetTextColor(0.65, 0.65, 0.7)
	button:SetScript("OnClick", onClick)
	button:SetScript("OnEnter", function(self)
		self.text:SetTextColor(1, 1, 1)
		Bars:ShowHeader()
		GameTooltip:SetOwner(self, "ANCHOR_TOP")
		GameTooltip:AddLine(type(tooltip) == "function" and tooltip() or tooltip, 1, 1, 1)
		GameTooltip:Show()
	end)
	button:SetScript("OnLeave", function(self)
		self.text:SetTextColor(0.65, 0.65, 0.7)
		GameTooltip:Hide()
		Bars:MaybeHideHeader()
	end)
	return button
end

function Bars:CreateHeader()
	local header = CreateFrame("Frame", nil, self.frame)
	header:SetPoint("BOTTOMLEFT", self.frame, "TOPLEFT", 0, 2)
	header:SetPoint("BOTTOMRIGHT", self.frame, "TOPRIGHT", 0, 2)
	header:SetHeight(HEADER_H)
	header:EnableMouse(true)
	header:RegisterForDrag("LeftButton")
	header:SetScript("OnDragStart", function()
		if not ns.db.locked then
			self.frame:StartMoving()
		end
	end)
	header:SetScript("OnDragStop", function()
		self.frame:StopMovingOrSizing()
		self:SavePosition()
	end)
	header:SetScript("OnEnter", function() self:ShowHeader() end)
	header:SetScript("OnLeave", function() self:MaybeHideHeader() end)
	header.bg = ns.AddBackground(header, 0, 0, 0, 0.6)

	header.title = header:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
	header.title:SetPoint("LEFT", 5, 0)
	header.title:SetText("Professions")

	header.close = ns.CreateCloseButton(header, HEADER_H)
	header.close:SetPoint("RIGHT", -1, 0)
	header.close:SetScript("OnClick", function()
		ns.Toggle(false)
		ns.Print("Bars hidden. Type /pb (or use your key binding) to bring them back.")
	end)
	header.close:HookScript("OnEnter", function() self:ShowHeader() end)
	header.close:HookScript("OnLeave", function() self:MaybeHideHeader() end)

	header.lock = TextButton(header, 34, function() ns.SetLocked(not ns.db.locked) end, function()
		return ns.db.locked and "Unlock to drag the bars by this header" or "Lock the bars in place"
	end)
	header.lock:SetPoint("RIGHT", header.close, "LEFT", -2, 0)

	header.theme = CreateFrame("Button", nil, header)
	header.theme:SetSize(12, 12)
	header.theme:SetPoint("RIGHT", header.lock, "LEFT", -4, 0)
	header.theme.cells = {}
	for i = 1, 4 do
		local cell = header.theme:CreateTexture(nil, "ARTWORK")
		cell:SetSize(6, 6)
		cell:SetPoint("TOPLEFT", ((i - 1) % 2) * 6, -math.floor((i - 1) / 2) * 6)
		header.theme.cells[i] = cell
	end
	header.theme:SetScript("OnClick", function() ns.SetTheme(ns.NextThemeKey()) end)
	header.theme:SetScript("OnEnter", function(button)
		self:ShowHeader()
		GameTooltip:SetOwner(button, "ANCHOR_TOP")
		GameTooltip:AddLine("Theme: " .. ns.GetTheme().label, 1, 1, 1)
		GameTooltip:AddLine("Click to try the next one", 0.65, 0.65, 0.7)
		GameTooltip:Show()
	end)
	header.theme:SetScript("OnLeave", function()
		GameTooltip:Hide()
		self:MaybeHideHeader()
	end)

	header.edit = TextButton(header, 30, function() self:SetEditing(not self.editing) end, function()
		return self.editing and "Finish choosing bars" or "Choose which professions get a bar"
	end)
	header.edit:SetPoint("RIGHT", header.theme, "LEFT", -4, 0)

	header:Hide()
	self.header = header
end

function Bars:UpdateHeader()
	local header = self.header
	header.lock.text:SetText(ns.db.locked and "unlock" or "lock")
	header.edit.text:SetText(self.editing and "done" or "edit")
	header.title:SetText(self.editing and "Click bars to show/hide" or "Professions")
	for i, cell in ipairs(header.theme.cells) do
		cell:SetColorTexture(ns.Color(SWATCH_KEYS[i]))
	end
	header.bg:SetColorTexture(0, 0, 0, self:Style() == "clean" and 0.3 or (ns.GetTheme().panel + 0.1))
	if self.editing then
		header:Show()
	end
end

function Bars:ShowHeader()
	self.hoverToken = (self.hoverToken or 0) + 1
	self.header:Show()
	self.hovered = true
	self:ApplyFade()
end

-- Optional idle fade (/pb fade): the bars dim until you hover them, edit
-- them or skill up.
function Bars:ApplyFade()
	local frame = self.frame
	if not frame then
		return
	end
	local awake = not ns.db.fade or self.hovered or self.editing or (self.wakeUntil and GetTime() < self.wakeUntil)
	local target = awake and 1 or 0.3
	local current = frame:GetAlpha()
	if math.abs(current - target) > 0.01 then
		ns.Tween(frame, 0.25, current, target, ns.SetAlphaTo)
	end
end

function Bars:Wake(seconds)
	self.wakeUntil = GetTime() + seconds
	self:ApplyFade()
	C_Timer.After(seconds + 0.05, function() self:ApplyFade() end)
end

-- Hide the header shortly after the mouse leaves, unless it came back.
function Bars:MaybeHideHeader()
	self.hoverToken = (self.hoverToken or 0) + 1
	local token = self.hoverToken
	C_Timer.After(0.35, function()
		if token ~= self.hoverToken or self.editing then
			return
		end
		if self.frame:IsMouseOver() or self.header:IsMouseOver() then
			return
		end
		self.header:Hide()
		self.hovered = false
		self:ApplyFade()
	end)
end

---------------------------------------------------------------------------
-- Bars
---------------------------------------------------------------------------

local function BarOnEnter(self)
	Bars:ShowHeader()
	local prof = self.prof
	if not prof then
		return
	end
	local r, g, b = ns.Color(prof.key)
	GameTooltip:SetOwner(self, "ANCHOR_RIGHT")
	GameTooltip:AddLine(prof.name, r, g, b)
	local step = ns.NextStep(prof.rank, prof.max, playerLevel())
	GameTooltip:AddDoubleLine(step.rank.name, prof.rank .. " / " .. prof.max, 1, 1, 1, 1, 1, 1)
	if prof.modifier and prof.modifier > 0 then
		GameTooltip:AddDoubleLine("Bonus from gear or race", "+" .. prof.modifier, 0.65, 0.65, 0.7, 0.35, 1, 0.45)
	end
	local gain = ns.SessionGain(prof)
	if gain > 0 then
		GameTooltip:AddDoubleLine("This session", "+" .. gain, 0.65, 0.65, 0.7, 0.35, 1, 0.45)
	end
	local _, long, color = ns.DescribeStep(step, prof)
	GameTooltip:AddLine(long, color[1], color[2], color[3], true)
	local alts = ns.AltsWith(prof.key)
	if #alts > 0 then
		local names = {}
		for i = 1, math.min(3, #alts) do
			names[i] = alts[i].name .. " " .. alts[i].rank .. "/" .. alts[i].max
		end
		GameTooltip:AddLine("Also on: " .. table.concat(names, ", ") .. (#alts > 3 and ", ..." or ""), 0.65, 0.65, 0.7, true)
	end
	GameTooltip:AddLine(" ")
	if Bars.editing then
		GameTooltip:AddLine("Click to show or hide this bar", 0.65, 0.65, 0.7)
	else
		GameTooltip:AddLine("Click to open the " .. prof.name .. " map", 0.65, 0.65, 0.7)
	end
	GameTooltip:Show()
end

local function BarOnLeave()
	GameTooltip:Hide()
	Bars:MaybeHideHeader()
end

local function BarOnClick(self)
	if Bars.editing then
		ns.SetHidden(self.key, not ns.char.hidden[self.key])
	else
		ns.SafeCall("opening the map", ns.Map.Toggle, ns.Map, self.key)
	end
end

---------------------------------------------------------------------------
-- Bar styles
--   clean (default): icon, name and value above a thin colour line
--   classic:         name and value above a gold-framed, glossy bar with the
--                    rank written inside and a gold marker at your skill
--   slim:            one compact bar per profession
---------------------------------------------------------------------------

local GLOSS = "Interface\\TargetingFrame\\UI-StatusBar"
local GOLD, GOLD_DARK, GOLD_BRIGHT = ns.GOLD_EDGE, { 0.42, 0.31, 0.10 }, { 1.00, 0.88, 0.50 }
local SEGMENTS = 8
local STYLES = {
	clean = { row = 26, gap = 8, pad = 10, icon = 18, textX = 26, bar = 4 },
	classic = { row = 37, gap = 7, pad = 8, bar = 18 },
	slim = { row = BAR_H, gap = 3, pad = 0 },
}

function Bars:Style()
	return (ns.db and STYLES[ns.db.style]) and ns.db.style or "clean"
end

-- Width of a bar's fill area for the current style and width setting.
function Bars:FillWidth(style)
	if style == "clean" then
		return ns.db.width - 2 * STYLES.clean.pad - STYLES.clean.textX
	elseif style == "classic" then
		return ns.db.width - 2 * STYLES.classic.pad - 4
	end
	return ns.db.width - BAR_H - 1
end

local function setupBar(bar)
	bar:RegisterForClicks("LeftButtonUp")
	bar.flash = bar.status:CreateTexture(nil, "OVERLAY")
	bar.flash:SetAllPoints()
	bar.flash:SetColorTexture(1, 1, 1, 1)
	bar.flash:SetAlpha(0)
	bar.hl = bar:CreateTexture(nil, "HIGHLIGHT")
	bar.hl:SetAllPoints(bar.status)
	bar.hl:SetColorTexture(1, 1, 1, 0.08)
	bar.status.bar = bar
	bar:SetScript("OnEnter", BarOnEnter)
	bar:SetScript("OnLeave", BarOnLeave)
	bar:SetScript("OnClick", BarOnClick)
	return bar
end

function Bars:CreateSlimBar()
	local bar = CreateFrame("Button", nil, self.frame)
	bar.style = "slim"
	bar:SetHeight(BAR_H)
	ns.AddBackground(bar, 0, 0, 0, 0.6)
	ns.AddBorder(bar, 0, 0, 0, 0.9)

	bar.icon = bar:CreateTexture(nil, "ARTWORK")
	bar.icon:SetPoint("TOPLEFT", 1, -1)
	bar.icon:SetSize(BAR_H - 2, BAR_H - 2)
	bar.icon:SetTexCoord(0.08, 0.92, 0.08, 0.92)

	local status = CreateFrame("StatusBar", nil, bar)
	status:SetPoint("TOPLEFT", bar.icon, "TOPRIGHT", 1, 0)
	status:SetPoint("BOTTOMRIGHT", -1, 1)
	status:SetStatusBarTexture(FLAT)
	status:SetMinMaxValues(0, 1)
	bar.status = status

	bar.track = status:CreateTexture(nil, "BACKGROUND")
	bar.track:SetAllPoints()

	bar.shine = status:CreateTexture(nil, "OVERLAY")
	bar.shine:SetPoint("TOPLEFT")
	bar.shine:SetPoint("TOPRIGHT")
	bar.shine:SetHeight(math.floor((BAR_H - 2) / 2))
	bar.shine:SetColorTexture(1, 1, 1, 0.07)

	bar.tick = status:CreateTexture(nil, "OVERLAY")
	bar.tick:SetWidth(2)

	bar.value = status:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	bar.value:SetPoint("RIGHT", -5, 0)
	bar.value:SetJustifyH("RIGHT")

	bar.name = status:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	bar.name:SetPoint("LEFT", 5, 0)
	bar.name:SetPoint("RIGHT", bar.value, "LEFT", -4, 0)
	bar.name:SetJustifyH("LEFT")
	bar.name:SetWordWrap(false)
	return setupBar(bar)
end

function Bars:CreateCleanBar()
	local metrics = STYLES.clean
	local bar = CreateFrame("Button", nil, self.frame)
	bar.style = "clean"
	bar:SetHeight(metrics.row)

	bar.icon = bar:CreateTexture(nil, "ARTWORK")
	bar.icon:SetSize(metrics.icon, metrics.icon)
	bar.icon:SetPoint("LEFT", 0, 0)
	bar.icon:SetTexCoord(0.08, 0.92, 0.08, 0.92)

	-- A thin colour line under the text; no frame, no gloss.
	local status = CreateFrame("StatusBar", nil, bar)
	status:SetPoint("BOTTOMLEFT", metrics.textX, 0)
	status:SetPoint("BOTTOMRIGHT", 0, 0)
	status:SetHeight(metrics.bar)
	status:SetStatusBarTexture(FLAT)
	status:SetMinMaxValues(0, 1)
	bar.status = status
	bar.track = status:CreateTexture(nil, "BACKGROUND")
	bar.track:SetAllPoints()
	bar.tick = status:CreateTexture(nil, "OVERLAY")
	bar.tick:SetWidth(1)

	-- Name ... note value, all sharing one baseline above the line.
	bar.value = bar:CreateFontString(nil, "OVERLAY", "GameFontHighlight")
	bar.value:SetPoint("BOTTOMRIGHT", 0, 8)
	bar.value:SetJustifyH("RIGHT")
	bar.note = bar:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	bar.note:SetPoint("BOTTOMRIGHT", bar.value, "BOTTOMLEFT", -6, 1)
	bar.note:SetJustifyH("RIGHT")
	bar.name = bar:CreateFontString(nil, "OVERLAY", "GameFontHighlight")
	bar.name:SetPoint("BOTTOMLEFT", metrics.textX, 8)
	bar.name:SetPoint("BOTTOMRIGHT", bar.note, "BOTTOMLEFT", -6, -1)
	bar.name:SetJustifyH("LEFT")
	bar.name:SetWordWrap(false)

	-- Selection: a thin colour line at the panel's edge.
	bar.accent = bar:CreateTexture(nil, "ARTWORK")
	bar.accent:SetWidth(2)
	bar.accent:SetPoint("TOPLEFT", bar, "TOPLEFT", -metrics.pad, 3)
	bar.accent:SetPoint("BOTTOMLEFT", bar, "BOTTOMLEFT", -metrics.pad, -3)
	bar.accent:Hide()

	setupBar(bar)
	bar.hl:ClearAllPoints()
	bar.hl:SetPoint("TOPLEFT", bar, "TOPLEFT", -metrics.pad, 4)
	bar.hl:SetPoint("BOTTOMRIGHT", bar, "BOTTOMRIGHT", metrics.pad, -4)
	bar.hl:SetColorTexture(1, 1, 1, 0.04)
	return bar
end

-- A small gold diamond built from 1px rows, so it looks the same everywhere.
local DIAMOND = { 1, 3, 5, 7, 5, 3, 1 }
local function CreateDiamond(parent)
	local diamond = CreateFrame("Frame", nil, parent)
	diamond:SetSize(7, 7)
	for i, width in ipairs(DIAMOND) do
		local row = diamond:CreateTexture(nil, "OVERLAY")
		row:SetSize(width, 1)
		row:SetPoint("TOP", diamond, "TOP", 0, -(i - 1))
		if i < 4 then
			row:SetColorTexture(1.00, 0.90, 0.52, 1)
		elseif i == 4 then
			row:SetColorTexture(0.96, 0.76, 0.30, 1)
		else
			row:SetColorTexture(0.70, 0.50, 0.15, 1)
		end
	end
	return diamond
end

function Bars:CreateClassicBar()
	local metrics = STYLES.classic
	local bar = CreateFrame("Button", nil, self.frame)
	bar.style = "classic"
	bar:SetHeight(metrics.row)

	-- Label row: icon, name ... value.
	bar.icon = bar:CreateTexture(nil, "ARTWORK")
	bar.icon:SetSize(14, 14)
	bar.icon:SetPoint("TOPLEFT", 1, -1)
	bar.icon:SetTexCoord(0.08, 0.92, 0.08, 0.92)
	bar.value = bar:CreateFontString(nil, "OVERLAY", "GameFontHighlight")
	bar.value:SetPoint("RIGHT", bar, "TOPRIGHT", -1, -8)
	bar.value:SetJustifyH("RIGHT")
	bar.name = bar:CreateFontString(nil, "OVERLAY", "GameFontHighlight")
	bar.name:SetPoint("LEFT", bar.icon, "RIGHT", 5, 0)
	bar.name:SetPoint("RIGHT", bar.value, "LEFT", -6, 0)
	bar.name:SetJustifyH("LEFT")
	bar.name:SetWordWrap(false)

	-- The framed bar: gold edge, darker inner edge, glossy fill.
	local frame = CreateFrame("Frame", nil, bar)
	frame:SetPoint("BOTTOMLEFT")
	frame:SetPoint("BOTTOMRIGHT")
	frame:SetHeight(metrics.bar)
	ns.AddBackground(frame, 0.02, 0.03, 0.05, 1)
	ns.AddBorder(frame, GOLD[1], GOLD[2], GOLD[3], 1)
	local inner = CreateFrame("Frame", nil, frame)
	inner:SetPoint("TOPLEFT", 1, -1)
	inner:SetPoint("BOTTOMRIGHT", -1, 1)
	ns.AddBorder(inner, GOLD_DARK[1], GOLD_DARK[2], GOLD_DARK[3], 1)
	bar.frame = frame

	local status = CreateFrame("StatusBar", nil, frame)
	status:SetPoint("TOPLEFT", 2, -2)
	status:SetPoint("BOTTOMRIGHT", -2, 2)
	status:SetStatusBarTexture(GLOSS)
	status:SetMinMaxValues(0, 1)
	bar.status = status
	bar.track = status:CreateTexture(nil, "BACKGROUND")
	bar.track:SetAllPoints()
	bar.shine = status:CreateTexture(nil, "OVERLAY")
	bar.shine:SetPoint("TOPLEFT")
	bar.shine:SetPoint("TOPRIGHT")
	bar.shine:SetHeight(math.floor((metrics.bar - 4) / 2))
	bar.shine:SetColorTexture(1, 1, 1, 0.08)

	-- Segment dividers and the gold line where the next rank can be trained.
	bar.dividers = {}
	for i = 1, SEGMENTS - 1 do
		local divider = status:CreateTexture(nil, "OVERLAY")
		divider:SetWidth(1)
		divider:SetColorTexture(0, 0, 0, 0.5)
		bar.dividers[i] = divider
	end
	bar.tick = status:CreateTexture(nil, "OVERLAY")
	bar.tick:SetWidth(1)

	bar.rank = status:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	bar.rank:SetPoint("CENTER", 0, 0)

	-- Gold marker with diamond ends at your current skill.
	local marker = CreateFrame("Frame", nil, frame)
	marker:SetFrameLevel(status:GetFrameLevel() + 2)
	marker:SetSize(9, metrics.bar + 8)
	marker.line = marker:CreateTexture(nil, "ARTWORK")
	marker.line:SetPoint("TOP", 0, -3)
	marker.line:SetPoint("BOTTOM", 0, 3)
	marker.line:SetWidth(2)
	marker.line:SetColorTexture(1, 0.84, 0.36, 1)
	marker.top = CreateDiamond(marker)
	marker.top:SetPoint("CENTER", marker, "TOP", 0, -3)
	marker.bottom = CreateDiamond(marker)
	marker.bottom:SetPoint("CENTER", marker, "BOTTOM", 0, 3)
	bar.marker = marker
	return setupBar(bar)
end

function Bars:PlaceMarker(bar, value)
	local x = math.floor((bar.fillWidth or 0) * math.max(0, math.min(1, value)) + 0.5)
	bar.marker:ClearAllPoints()
	bar.marker:SetPoint("CENTER", bar.status, "LEFT", x, 0)
end

local function setBarValue(status, value)
	status:SetValue(value)
	if status.bar.marker then
		Bars:PlaceMarker(status.bar, value)
	end
end

function Bars:UpdateBar(bar, prof, hidden)
	local previous = bar.prof
	bar.key, bar.prof = prof.key, prof
	local classic, clean = bar.style == "classic", bar.style == "clean"
	local theme = ns.GetTheme()
	local r, g, b = ns.Color(prof.key)
	local width = self:FillWidth(bar.style)
	bar.fillWidth = width

	bar.icon:SetTexture(prof.icon)
	bar.icon:SetDesaturated(hidden and true or false)
	bar.status:SetStatusBarColor(r, g, b, 1)
	if clean then
		bar.track:SetColorTexture(1, 1, 1, 0.09)
	elseif classic then
		bar.track:SetColorTexture(0.03 + r * 0.07, 0.04 + g * 0.07, 0.06 + b * 0.07, 1)
	else
		bar.track:SetColorTexture(r * theme.track, g * theme.track, b * theme.track, 0.95)
		bar.bg:SetColorTexture(0, 0, 0, theme.panel)
	end
	bar.name:SetText(prof.name)
	bar:SetAlpha(hidden and 0.45 or 1)

	local fraction = prof.max > 0 and math.min(1, prof.rank / prof.max) or 0
	local before = previous and previous.max > 0 and (previous.rank / previous.max) or nil
	if before and math.abs(before - fraction) > 0.0001 then
		ns.Tween(bar.status, 0.45, before, fraction, setBarValue)
		if prof.rank > previous.rank then
			ns.Tween(bar.flash, 0.6, 0.45, 0, ns.SetAlphaTo)
			self:Wake(4)
		end
	else
		ns.StopTween(bar.status)
		setBarValue(bar.status, fraction)
	end

	-- Gold line where the next rank can be trained (the clean style only
	-- shows it until you get there; then the numbers turn gold instead).
	local step = ns.NextStep(prof.rank, prof.max, playerLevel())
	local nextRank = step.nextRank
	if clean then
		if step.state == "skill" and nextRank.skill < prof.max then
			local x = math.floor(width * nextRank.skill / prof.max + 0.5)
			bar.tick:ClearAllPoints()
			bar.tick:SetPoint("TOPLEFT", bar.status, "TOPLEFT", x, 1)
			bar.tick:SetPoint("BOTTOMLEFT", bar.status, "BOTTOMLEFT", x, -1)
			bar.tick:SetColorTexture(1, 0.82, 0, 0.85)
			bar.tick:Show()
		else
			bar.tick:Hide()
		end
	elseif nextRank and nextRank.skill < prof.max then
		local x = math.floor(width * nextRank.skill / prof.max + 0.5)
		local half = classic and 0 or 1
		bar.tick:ClearAllPoints()
		bar.tick:SetPoint("TOPLEFT", bar.status, "TOPLEFT", x - half, 0)
		bar.tick:SetPoint("BOTTOMLEFT", bar.status, "BOTTOMLEFT", x - half, 0)
		bar.tick:SetColorTexture(1, 0.82, 0, step.state == "train" and 1 or (classic and 0.75 or 0.5))
		bar.tick:Show()
	else
		bar.tick:Hide()
	end

	-- Skill gained this session shows as a small green "+5".
	local gain = ns.SessionGain(prof)
	local gained = gain > 0 and ("|cff59ff73+" .. gain .. "|r  ") or ""
	if clean then
		local notes = {}
		if not self.editing then
			if gain > 0 then
				notes[#notes + 1] = "|cff59ff73+" .. gain .. "|r"
			end
			if step.state == "train" then
				notes[#notes + 1] = "|cffffd100train|r"
			end
		end
		bar.note:SetText(table.concat(notes, "  "))
		if self.editing then
			bar.value:SetText(hidden and "|cff8a8a90hidden|r" or "shown")
		elseif step.state == "train" then
			bar.value:SetText("|cffffd100" .. prof.rank .. " / " .. prof.max .. "|r")
		else
			bar.value:SetText(prof.rank .. "|cff8c8c96 / " .. prof.max .. "|r")
		end
	elseif self.editing then
		bar.value:SetText(hidden and "|cff8a8a90hidden|r" or "shown")
	elseif step.state == "train" then
		bar.value:SetText(gained .. "|cffffd100" .. prof.rank .. "/" .. prof.max .. "|r")
	else
		bar.value:SetText(gained .. prof.rank .. "/" .. prof.max)
	end

	if classic then
		if step.state == "train" then
			bar.rank:SetText("|cffffd100Ready for " .. nextRank.name .. "|r")
		elseif step.state == "mastered" then
			bar.rank:SetText("Mastered")
		else
			bar.rank:SetText(step.rank.name)
		end
		for i, divider in ipairs(bar.dividers) do
			local x = math.floor(width * i / SEGMENTS + 0.5)
			divider:ClearAllPoints()
			divider:SetPoint("TOPLEFT", bar.status, "TOPLEFT", x, 0)
			divider:SetPoint("BOTTOMLEFT", bar.status, "BOTTOMLEFT", x, 0)
		end
	end
	self:UpdateSelection(bar)
end

-- Highlight the bar whose map is open.
function Bars:UpdateSelection(only)
	for key, bar in pairs(self.bars) do
		if not only or only == bar then
			local selected = ns.Map:IsOpenFor(key)
			bar.selected = selected
			if bar.style == "clean" then
				bar.accent:SetShown(selected)
				if selected then
					bar.accent:SetColorTexture(ns.Color(key))
				end
			elseif bar.style == "classic" then
				local edge = selected and GOLD_BRIGHT or GOLD
				ns.SetBorderColor(bar.frame, edge[1], edge[2], edge[3], 1)
				if selected then
					local r, g, b = ns.Color(key)
					bar.name:SetTextColor(0.45 + r * 0.55, 0.45 + g * 0.55, 0.45 + b * 0.55)
				else
					bar.name:SetTextColor(1, 1, 1)
				end
			elseif selected then
				ns.SetBorderColor(bar, ns.Color(key))
			else
				ns.SetBorderColor(bar, 0, 0, 0, 0.9)
			end
		end
	end
end

-- Clean: a flat, translucent panel. Classic: slate-blue with a dark edge.
-- Slim: no panel.
function Bars:UpdatePanel(style)
	local frame = self.frame
	frame.bg:SetShown(style == "clean" or style == "classic")
	for _, edge in ipairs(frame.border) do
		edge:SetShown(style == "classic")
	end
	if style == "classic" then
		local bottom, top = ns.PanelColors()
		ns.SetVerticalGradient(frame.bg, bottom, top, 0.93)
	elseif style == "clean" then
		frame.bg:SetColorTexture(0.03, 0.035, 0.045, math.min(0.8, ns.GetTheme().panel + 0.1))
	end
end

function Bars:Refresh()
	if not self.frame then
		return
	end
	local style = self:Style()
	for name, set in pairs(self.byStyle) do
		if name ~= style then
			for _, bar in pairs(set) do
				bar:Hide()
			end
		end
	end
	self.bars = self.byStyle[style]
	local metrics = STYLES[style]

	self.frame:SetWidth(ns.db.width)
	if self.appliedScale ~= ns.db.scale then
		self.frame:SetScale(ns.db.scale)
		self.appliedScale = ns.db.scale
		self:RestorePosition()
	end
	local shown, used = 0, {}
	for _, prof in ipairs(ns.profs or {}) do
		local hidden = ns.char.hidden[prof.key] and true or false
		if self.editing or not hidden then
			local bar = self.bars[prof.key]
			if not bar then
				if style == "clean" then
					bar = self:CreateCleanBar()
				elseif style == "classic" then
					bar = self:CreateClassicBar()
				else
					bar = self:CreateSlimBar()
				end
				self.bars[prof.key] = bar
			end
			local y = -(metrics.pad + shown * (metrics.row + metrics.gap))
			bar:ClearAllPoints()
			bar:SetPoint("TOPLEFT", self.frame, "TOPLEFT", metrics.pad, y)
			bar:SetPoint("TOPRIGHT", self.frame, "TOPRIGHT", -metrics.pad, y)
			self:UpdateBar(bar, prof, hidden)
			bar:Show()
			used[prof.key] = true
			shown = shown + 1
		end
	end
	for key, bar in pairs(self.bars) do
		if not used[key] then
			bar:Hide()
		end
	end
	if shown == 0 then
		self.empty.text:SetText(#(ns.profs or {}) > 0 and "All bars hidden - hover here, then 'edit'" or "No professions yet - visit a trainer")
		self.empty:Show()
		self.frame:SetHeight(BAR_H)
	else
		self.empty:Hide()
		self.frame:SetHeight(2 * metrics.pad + shown * metrics.row + (shown - 1) * metrics.gap)
	end
	self:UpdatePanel(shown > 0 and style or "slim")
	self:UpdateHeader()
	self:ApplyFade()
end

function Bars:SetEditing(on)
	self.editing = on and true or false
	if not self.editing then
		self:MaybeHideHeader()
	end
	self:Refresh()
end

---------------------------------------------------------------------------
-- Frame and position
---------------------------------------------------------------------------

function Bars:Create()
	if self.frame then
		return
	end
	local frame = CreateFrame("Frame", "ProfessionBarsFrame", UIParent)
	frame:SetSize(ns.db.width, BAR_H)
	ns.AddBackground(frame, 0, 0, 0, 0)
	ns.AddBorder(frame, 0, 0, 0, 0.7)
	self.byStyle = { clean = {}, classic = {}, slim = {} }
	self.bars = self.byStyle.clean
	frame:SetFrameStrata("MEDIUM")
	frame:SetClampedToScreen(true)
	frame:SetMovable(true)
	if frame.SetDontSavePosition then
		frame:SetDontSavePosition(true)
	end
	self.frame = frame

	local empty = CreateFrame("Frame", nil, frame)
	empty:SetAllPoints()
	empty:EnableMouse(true)
	ns.AddBackground(empty, 0, 0, 0, 0.45)
	empty.text = empty:CreateFontString(nil, "OVERLAY", "GameFontDisableSmall")
	empty.text:SetPoint("LEFT", 6, 0)
	empty.text:SetPoint("RIGHT", -6, 0)
	empty.text:SetJustifyH("LEFT")
	empty:SetScript("OnEnter", function() self:ShowHeader() end)
	empty:SetScript("OnLeave", function() self:MaybeHideHeader() end)
	empty:Hide()
	self.empty = empty

	self:CreateHeader()
	self:RestorePosition()
end

-- Positions are stored in screen pixels so scale changes don't move the bars.
function Bars:SavePosition()
	local frame = self.frame
	local scale = frame:GetEffectiveScale()
	local left, top = frame:GetLeft(), frame:GetTop()
	if left and top then
		ns.db.point = { left * scale, top * scale }
	end
	self:RestorePosition()
end

function Bars:RestorePosition()
	local frame = self.frame
	frame:ClearAllPoints()
	local point = ns.db.point
	if point then
		local scale = frame:GetEffectiveScale()
		frame:SetPoint("TOPLEFT", UIParent, "BOTTOMLEFT", point[1] / scale, point[2] / scale)
	else
		frame:SetPoint("TOPLEFT", UIParent, "LEFT", 40, 160)
	end
end

function Bars:SetShown(shown)
	if self.frame then
		self.frame:SetShown(shown)
	end
end

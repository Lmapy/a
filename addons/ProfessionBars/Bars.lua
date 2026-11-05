-- The bar stack: one slim, colour-themed bar per profession. Click a bar to
-- open its mind map. Hovering reveals a small header with controls.
local _, ns = ...

local Bars = { bars = {} }
ns.Bars = Bars

local BAR_H, GAP, HEADER_H = 18, 3, 16
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
	header.bg:SetColorTexture(0, 0, 0, ns.GetTheme().panel + 0.1)
	if self.editing then
		header:Show()
	end
end

function Bars:ShowHeader()
	self.hoverToken = (self.hoverToken or 0) + 1
	self.header:Show()
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

function Bars:CreateBar()
	local bar = CreateFrame("Button", nil, self.frame)
	bar:SetHeight(BAR_H)
	bar:RegisterForClicks("LeftButtonUp")
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

	bar.flash = status:CreateTexture(nil, "OVERLAY")
	bar.flash:SetAllPoints()
	bar.flash:SetColorTexture(1, 1, 1, 1)
	bar.flash:SetAlpha(0)

	bar.value = status:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	bar.value:SetPoint("RIGHT", -5, 0)
	bar.value:SetJustifyH("RIGHT")

	bar.name = status:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	bar.name:SetPoint("LEFT", 5, 0)
	bar.name:SetPoint("RIGHT", bar.value, "LEFT", -4, 0)
	bar.name:SetJustifyH("LEFT")
	bar.name:SetWordWrap(false)

	bar.hl = bar:CreateTexture(nil, "HIGHLIGHT")
	bar.hl:SetAllPoints(status)
	bar.hl:SetColorTexture(1, 1, 1, 0.08)

	bar:SetScript("OnEnter", BarOnEnter)
	bar:SetScript("OnLeave", BarOnLeave)
	bar:SetScript("OnClick", BarOnClick)
	return bar
end

local function setBarValue(status, value)
	status:SetValue(value)
end

function Bars:UpdateBar(bar, prof, hidden)
	local previous = bar.prof
	bar.key, bar.prof = prof.key, prof
	local theme = ns.GetTheme()
	local r, g, b = ns.Color(prof.key)

	bar.icon:SetTexture(prof.icon)
	bar.icon:SetDesaturated(hidden and true or false)
	bar.status:SetStatusBarColor(r, g, b, 1)
	bar.track:SetColorTexture(r * theme.track, g * theme.track, b * theme.track, 0.95)
	bar.bg:SetColorTexture(0, 0, 0, theme.panel)
	bar.name:SetText(prof.name)
	bar:SetAlpha(hidden and 0.45 or 1)

	local fraction = prof.max > 0 and math.min(1, prof.rank / prof.max) or 0
	local before = previous and previous.max > 0 and (previous.rank / previous.max) or nil
	if before and math.abs(before - fraction) > 0.0001 then
		ns.Tween(bar.status, 0.45, before, fraction, setBarValue)
		if prof.rank > previous.rank then
			ns.Tween(bar.flash, 0.6, 0.45, 0, ns.SetAlphaTo)
		end
	else
		ns.StopTween(bar.status)
		bar.status:SetValue(fraction)
	end

	-- Gold tick where the next rank can be trained.
	local step = ns.NextStep(prof.rank, prof.max, playerLevel())
	local nextRank = step.nextRank
	if nextRank and nextRank.skill < prof.max then
		local width = ns.db.width - BAR_H - 1
		local x = math.floor(width * nextRank.skill / prof.max + 0.5)
		bar.tick:ClearAllPoints()
		bar.tick:SetPoint("TOPLEFT", bar.status, "TOPLEFT", x - 1, 0)
		bar.tick:SetPoint("BOTTOMLEFT", bar.status, "BOTTOMLEFT", x - 1, 0)
		bar.tick:SetColorTexture(1, 0.82, 0, step.state == "train" and 1 or 0.5)
		bar.tick:Show()
	else
		bar.tick:Hide()
	end

	-- Skill gained this session shows as a small green "+5".
	local gain = ns.SessionGain(prof)
	local gained = gain > 0 and ("|cff59ff73+" .. gain .. "|r  ") or ""
	if self.editing then
		bar.value:SetText(hidden and "|cff8a8a90hidden|r" or "shown")
	elseif step.state == "train" then
		bar.value:SetText(gained .. "|cffffd100" .. prof.rank .. "/" .. prof.max .. "|r")
	else
		bar.value:SetText(gained .. prof.rank .. "/" .. prof.max)
	end
	self:UpdateSelection(bar)
end

-- Highlight the bar whose map is open.
function Bars:UpdateSelection(only)
	for key, bar in pairs(self.bars) do
		if not only or only == bar then
			if ns.Map:IsOpenFor(key) then
				ns.SetBorderColor(bar, ns.Color(key))
			else
				ns.SetBorderColor(bar, 0, 0, 0, 0.9)
			end
		end
	end
end

function Bars:Refresh()
	if not self.frame then
		return
	end
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
				bar = self:CreateBar()
				self.bars[prof.key] = bar
			end
			local y = -shown * (BAR_H + GAP)
			bar:ClearAllPoints()
			bar:SetPoint("TOPLEFT", self.frame, "TOPLEFT", 0, y)
			bar:SetPoint("TOPRIGHT", self.frame, "TOPRIGHT", 0, y)
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
	else
		self.empty:Hide()
	end
	self.frame:SetHeight(shown > 0 and (shown * (BAR_H + GAP) - GAP) or BAR_H)
	self:UpdateHeader()
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

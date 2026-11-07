-- The profession panel. Clicking a bar slides it out beside the bars: one
-- column that reads top to bottom. The header and a 0-300 track show where
-- you are, one line says what to do next, and a few tabs hold the plan,
-- recipes, where to gather and your other characters.
local _, ns = ...

local Detail = { expanded = {} }
ns.Detail = Detail

-- Metrics in UI units.
local W, PAD = 320, 12
local INNER = W - 2 * PAD
local ICON = 22
local TITLE_H, ROW_H, LINE_H, GAP = 14, 16, 13, 10
local GREY = ns.GREY

Detail.metrics = { W = W, PAD = PAD, INNER = INNER }

local function insertLink(link)
	if ChatEdit_InsertLink then
		return ChatEdit_InsertLink(link)
	elseif ChatFrameUtil and ChatFrameUtil.InsertLink then
		return ChatFrameUtil.InsertLink(link)
	end
end

-- Height of a string wrapped at `width`: the client's own measure, or an
-- estimate from its one-line width when that is larger (it can lag a frame).
local function wrappedHeight(fontString, width)
	local natural = fontString.GetUnboundedStringWidth and fontString:GetUnboundedStringWidth() or fontString:GetStringWidth()
	local lines = math.max(1, math.ceil(natural / width))
	return math.max(lines * LINE_H, math.ceil(fontString:GetStringHeight() or 0))
end

local function addLines(lines)
	for _, entry in ipairs(lines or {}) do
		GameTooltip:AddLine(entry[1], entry[2], entry[3], entry[4], true)
	end
end

---------------------------------------------------------------------------
-- Rows, section titles and tabs (pooled)
---------------------------------------------------------------------------

local function RowOnEnter(self)
	local data = self.data
	if not data then
		return
	end
	GameTooltip:SetOwner(self, "ANCHOR_RIGHT")
	local recipe = data.recipe
	if recipe then
		local ok = recipe.link and pcall(GameTooltip.SetHyperlink, GameTooltip, recipe.link)
		if not ok and type(recipe.id) == "number" and GameTooltip.SetSpellByID then
			ok = pcall(GameTooltip.SetSpellByID, GameTooltip, recipe.id)
		end
		if ok then
			addLines(data.tooltip)
			if data.count and data.count > 0 then
				GameTooltip:AddLine("You can make " .. data.count .. " from your bags.", 0.35, 1, 0.45)
			end
			if recipe.link then
				GameTooltip:AddLine("Shift-click to link it in chat.", 0.6, 0.6, 0.65)
			end
			GameTooltip:Show()
			return
		end
	end
	if data.itemID and GameTooltip.SetItemByID and pcall(GameTooltip.SetItemByID, GameTooltip, data.itemID) then
		addLines(data.tooltip)
		GameTooltip:Show()
		return
	end
	if not data.tooltip then
		GameTooltip:Hide()
		return
	end
	GameTooltip:AddLine(data.label, 1, 1, 1)
	addLines(data.tooltip)
	GameTooltip:Show()
end

local function RowOnLeave()
	GameTooltip:Hide()
end

local function RowOnClick(self)
	local data = self.data
	if not data then
		return
	end
	if data.expand then
		ns.SafeCall("showing more", Detail.Expand, Detail, data.expand)
	elseif data.recipe and data.recipe.link and IsShiftKeyDown and IsShiftKeyDown() then
		insertLink(data.recipe.link)
	end
end

local function CreateRow(parent)
	local row = CreateFrame("Button", nil, parent)
	row:RegisterForClicks("LeftButtonUp")
	row.value = row:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	row.value:SetPoint("TOPRIGHT", 0, -2)
	row.value:SetJustifyH("RIGHT")
	row.label = row:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	row.label:SetJustifyH("LEFT")
	row.hl = row:CreateTexture(nil, "HIGHLIGHT")
	row.hl:SetPoint("TOPLEFT", -4, 0)
	row.hl:SetPoint("BOTTOMRIGHT", 4, 0)
	row.hl:SetColorTexture(1, 1, 1, 0.05)
	row:SetScript("OnEnter", RowOnEnter)
	row:SetScript("OnLeave", RowOnLeave)
	row:SetScript("OnClick", RowOnClick)
	return row
end

local function CreateTitle(parent)
	local title = CreateFrame("Frame", nil, parent)
	title:SetSize(INNER, TITLE_H)
	title.text = title:CreateFontString(nil, "OVERLAY", "GameFontDisableSmall")
	title.text:SetPoint("LEFT", 0, 0)
	title.right = title:CreateFontString(nil, "OVERLAY", "GameFontDisableSmall")
	title.right:SetPoint("RIGHT", 0, 0)
	title.right:SetJustifyH("RIGHT")
	return title
end

local function TabOnClick(self)
	ns.SafeCall("switching tabs", Detail.SetTab, Detail, self.key)
end

local function CreateTab(parent)
	local tab = CreateFrame("Button", nil, parent)
	tab:RegisterForClicks("LeftButtonUp")
	tab:SetHeight(18)
	tab.text = tab:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	tab.text:SetPoint("TOP", 0, -2)
	tab.line = tab:CreateTexture(nil, "ARTWORK")
	tab.line:SetPoint("BOTTOMLEFT", 0, 0)
	tab.line:SetPoint("BOTTOMRIGHT", 0, 0)
	tab.line:SetHeight(2)
	tab:SetScript("OnClick", TabOnClick)
	return tab
end

local function resetRegion(object)
	object:ClearAllPoints()
end

---------------------------------------------------------------------------
-- Panel
---------------------------------------------------------------------------

function Detail:Create()
	if self.frame then
		return
	end
	local frame = CreateFrame("Frame", "ProfessionBarsDetail", UIParent)
	frame:SetSize(W, 200)
	frame:SetFrameStrata("HIGH")
	frame:SetToplevel(true)
	frame:SetClampedToScreen(true)
	frame:EnableMouse(true)
	if frame.SetDontSavePosition then
		frame:SetDontSavePosition(true)
	end
	self.frame = frame
	frame:Hide()
	frame:SetScript("OnHide", function() self:OnHidden() end)

	ns.AddBackground(frame, 0.04, 0.045, 0.06, 0.96)
	ns.AddBorder(frame, 1, 1, 1, 0.07)
	frame.accent = frame:CreateTexture(nil, "ARTWORK")
	frame.accent:SetPoint("TOPLEFT", 1, -1)
	frame.accent:SetPoint("TOPRIGHT", -1, -1)
	frame.accent:SetHeight(2)

	-- Header: icon, name and rank ... value, close.
	frame.icon = frame:CreateTexture(nil, "ARTWORK")
	frame.icon:SetSize(ICON, ICON)
	frame.icon:SetPoint("TOPLEFT", PAD, -PAD)
	frame.icon:SetTexCoord(0.08, 0.92, 0.08, 0.92)
	frame.close = ns.CreateCloseButton(frame, 16)
	frame.close:SetPoint("RIGHT", frame, "TOPRIGHT", -8, -(PAD + ICON / 2))
	frame.close:SetScript("OnClick", function() self:Close() end)
	frame.value = frame:CreateFontString(nil, "OVERLAY", "GameFontHighlight")
	frame.value:SetPoint("RIGHT", frame.close, "LEFT", -6, 0)
	frame.value:SetJustifyH("RIGHT")
	frame.name = frame:CreateFontString(nil, "OVERLAY", "GameFontHighlight")
	frame.name:SetPoint("LEFT", frame.icon, "RIGHT", 8, 0)
	frame.rankName = frame:CreateFontString(nil, "OVERLAY", "GameFontDisableSmall")
	frame.rankName:SetPoint("LEFT", frame.name, "RIGHT", 6, 0)

	-- Track: the whole 0-300 range, with the reachable part, your skill,
	-- rank boundaries and the next training point.
	local track = CreateFrame("Button", nil, frame)
	track:SetSize(INNER, 6)
	track.base = track:CreateTexture(nil, "BACKGROUND")
	track.base:SetAllPoints()
	track.base:SetColorTexture(1, 1, 1, 0.06)
	track.avail = track:CreateTexture(nil, "BORDER")
	track.avail:SetPoint("TOPLEFT")
	track.avail:SetPoint("BOTTOMLEFT")
	track.avail:SetColorTexture(1, 1, 1, 0.12)
	track.fill = track:CreateTexture(nil, "ARTWORK")
	track.fill:SetPoint("TOPLEFT")
	track.fill:SetPoint("BOTTOMLEFT")
	track.ticks = {}
	for i = 1, 3 do
		local tick = track:CreateTexture(nil, "OVERLAY")
		tick:SetWidth(1)
		tick:SetColorTexture(1, 1, 1, 0.35)
		track.ticks[i] = tick
	end
	track.goal = track:CreateTexture(nil, "OVERLAY")
	track.goal:SetWidth(2)
	track.goal:SetColorTexture(1, 0.82, 0, 0.95)
	track:SetScript("OnEnter", function(button)
		GameTooltip:SetOwner(button, "ANCHOR_RIGHT")
		addLines(button.lines)
		GameTooltip:Show()
	end)
	track:SetScript("OnLeave", RowOnLeave)
	frame.track = track

	-- What to do next, in one line.
	local nextLine = CreateFrame("Button", nil, frame)
	nextLine:SetSize(INNER, 16)
	nextLine.label = nextLine:CreateFontString(nil, "OVERLAY", "GameFontDisableSmall")
	nextLine.label:SetPoint("LEFT", 0, 0)
	nextLine.label:SetText("NEXT")
	nextLine.text = nextLine:CreateFontString(nil, "OVERLAY", "GameFontHighlight")
	nextLine.text:SetPoint("LEFT", nextLine.label, "RIGHT", 8, 0)
	nextLine.text:SetPoint("RIGHT", 0, 0)
	nextLine.text:SetJustifyH("LEFT")
	nextLine.text:SetWordWrap(false)
	nextLine:SetScript("OnEnter", function(button)
		GameTooltip:SetOwner(button, "ANCHOR_RIGHT")
		GameTooltip:AddLine(button.tip or "", 1, 1, 1, true)
		GameTooltip:Show()
	end)
	nextLine:SetScript("OnLeave", RowOnLeave)
	frame.nextLine = nextLine

	frame.divider = frame:CreateTexture(nil, "ARTWORK")
	frame.divider:SetHeight(1)
	frame.divider:SetColorTexture(1, 1, 1, 0.08)

	frame.footer = frame:CreateFontString(nil, "OVERLAY", "GameFontDisableSmall")
	frame.footer:SetJustifyH("LEFT")
	frame.footer:SetWidth(INNER)

	self.rows = ns.NewPool(function() return CreateRow(frame) end, resetRegion)
	self.titles = ns.NewPool(function() return CreateTitle(frame) end, resetRegion)
	self.tabs = ns.NewPool(function() return CreateTab(frame) end, resetRegion)

	if UISpecialFrames then
		table.insert(UISpecialFrames, "ProfessionBarsDetail")
	end
	self:Place()
end

-- Beside the bars, on the side with more room.
function Detail:Place()
	local frame = self.frame
	frame:ClearAllPoints()
	local bars = ns.Bars.frame
	local left, right = bars and bars:GetLeft(), bars and bars:GetRight()
	if bars and bars:IsShown() and left and right then
		local barScale = bars:GetEffectiveScale()
		local roomRight = UIParent:GetWidth() * UIParent:GetEffectiveScale() - right * barScale
		if roomRight >= left * barScale then
			frame:SetPoint("TOPLEFT", bars, "TOPRIGHT", 8, 0)
		else
			frame:SetPoint("TOPRIGHT", bars, "TOPLEFT", -8, 0)
		end
	else
		frame:SetPoint("CENTER", UIParent, "CENTER", 0, 40)
	end
end

function Detail:DrawTrack(prof, step, r, g, b)
	local track = self.frame.track
	local rules = step.rules
	local top = rules[#rules].cap
	local function x(skill)
		return math.floor(INNER * math.min(skill, top) / top + 0.5)
	end
	track.avail:SetWidth(math.max(1, x(prof.max)))
	track.fill:SetWidth(math.max(1, x(prof.rank)))
	track.fill:SetColorTexture(r, g, b, 1)
	local lines = { { prof.name .. ": " .. prof.rank .. " of " .. prof.max .. " (" .. step.rank.name .. ")", 1, 1, 1 } }
	for i, tick in ipairs(track.ticks) do
		local rank = rules[i]
		tick:ClearAllPoints()
		if rank and i < #rules then
			tick:SetPoint("TOPLEFT", track, "TOPLEFT", x(rank.cap), 2)
			tick:SetPoint("BOTTOMLEFT", track, "BOTTOMLEFT", x(rank.cap), -2)
			tick:Show()
		else
			tick:Hide()
		end
	end
	for _, rank in ipairs(rules) do
		local color = rank.cap <= prof.max and { 0.8, 0.8, 0.8 } or GREY
		lines[#lines + 1] = { rank.name .. ": up to " .. rank.cap, color[1], color[2], color[3] }
	end
	local nextRank = step.nextRank
	track.goal:ClearAllPoints()
	if nextRank and nextRank.skill then
		track.goal:SetPoint("TOPLEFT", track, "TOPLEFT", x(nextRank.skill) - 1, 3)
		track.goal:SetPoint("BOTTOMLEFT", track, "BOTTOMLEFT", x(nextRank.skill) - 1, -3)
		track.goal:Show()
		lines[#lines + 1] = { "Gold mark: where " .. nextRank.name .. " can be learned (" .. nextRank.skill .. ").", 1, 0.82, 0 }
	else
		track.goal:Hide()
	end
	track.lines = lines
end

function Detail:Render()
	local frame = self.frame
	local prof = ns.byKey and ns.byKey[self.current]
	if not frame or not prof then
		return
	end
	local view = ns.BuildDetail(prof)
	local r, g, b = ns.Color(prof.key)
	self.view = view

	-- Look: the classic style keeps its slate panel and gold edge.
	if ns.Bars:Style() == "classic" then
		local bottom, top = ns.PanelColors()
		ns.SetVerticalGradient(frame.bg, bottom, top, 0.96)
		ns.SetBorderColor(frame, ns.GOLD_EDGE[1], ns.GOLD_EDGE[2], ns.GOLD_EDGE[3], 1)
	else
		frame.bg:SetColorTexture(0.04, 0.045, 0.06, 0.96)
		ns.SetBorderColor(frame, 1, 1, 1, 0.07)
	end
	frame.accent:SetColorTexture(r, g, b, 1)

	-- Header.
	frame.icon:SetTexture(prof.icon)
	frame.name:SetText(prof.name)
	frame.rankName:SetText(view.step.rank.name)
	local gain = ns.SessionGain(prof)
	frame.value:SetText((gain > 0 and ("|cff59ff73+" .. gain .. "|r  ") or "") .. prof.rank .. "|cff8c8c96 / " .. prof.max .. "|r")

	local y = PAD + ICON + 10
	frame.track:ClearAllPoints()
	frame.track:SetPoint("TOPLEFT", PAD, -y)
	self:DrawTrack(prof, view.step, r, g, b)
	y = y + 6 + 10

	local nextLine = frame.nextLine
	nextLine:ClearAllPoints()
	nextLine:SetPoint("TOPLEFT", PAD, -y)
	nextLine.text:SetText(view.next.text)
	nextLine.text:SetTextColor(view.next.color[1], view.next.color[2], view.next.color[3])
	nextLine.tip = view.next.tooltip
	y = y + 16 + 8

	-- Tabs: only the ones with something to show.
	self.tabs:ReleaseAll()
	local active = ns.db.tabs[prof.key]
	local activeTab
	for _, tab in ipairs(view.tabs) do
		if tab.key == active then
			activeTab = tab
		end
	end
	activeTab = activeTab or view.tabs[1]
	local x = 0
	for _, tab in ipairs(view.tabs) do
		local button = self.tabs:Acquire()
		button.key = tab.key
		button.text:SetText(tab.label)
		local width = math.floor(button.text:GetStringWidth() + 0.5)
		button:SetWidth(width)
		button:SetPoint("TOPLEFT", PAD + x, -y)
		local isActive = tab == activeTab
		button.text:SetTextColor(isActive and 1 or 0.6, isActive and 1 or 0.6, isActive and 1 or 0.64)
		button.line:SetColorTexture(r, g, b, 1)
		button.line:SetShown(isActive)
		x = x + width + 16
	end
	self.activeTab = activeTab and activeTab.key
	y = y + 18
	frame.divider:ClearAllPoints()
	frame.divider:SetPoint("TOPLEFT", PAD, -y)
	frame.divider:SetPoint("TOPRIGHT", -PAD, -y)
	y = y + 1 + GAP

	-- Sections of the active tab.
	self.rows:ReleaseAll()
	self.titles:ReleaseAll()
	for _, section in ipairs(activeTab and activeTab.sections or {}) do
		local title = self.titles:Acquire()
		title:SetPoint("TOPLEFT", PAD, -y)
		title.text:SetText(section.title:upper())
		title.right:SetText(section.right or "")
		local rc = section.rightColor or GREY
		title.right:SetTextColor(rc[1], rc[2], rc[3])
		y = y + TITLE_H + 2
		local expandKey = prof.key .. ":" .. (activeTab.key or "") .. ":" .. section.key
		local limit = (section.limit and not self.expanded[expandKey]) and section.limit or #section.rows
		local hidden = #section.rows - limit
		local shown = {}
		for i = 1, math.min(#section.rows, hidden > 0 and limit - 1 or limit) do
			shown[#shown + 1] = section.rows[i]
		end
		if hidden > 0 then
			shown[#shown + 1] = { label = "Show all " .. #section.rows, color = GREY, expand = expandKey }
		end
		for _, data in ipairs(shown) do
			local row = self.rows:Acquire()
			row.data = data
			row:SetPoint("TOPLEFT", PAD, -y)
			row:SetWidth(INNER)
			local color = data.color or ns.WHITE
			row.label:SetText(data.label)
			row.label:SetTextColor(color[1], color[2], color[3])
			row.value:SetText(data.value or "")
			local vc = data.valueColor or GREY
			row.value:SetTextColor(vc[1], vc[2], vc[3])
			row.label:ClearAllPoints()
			row.label:SetPoint("TOPLEFT", 0, -2)
			local height = ROW_H
			if data.wrap then
				row.label:SetWidth(INNER)
				row.label:SetWordWrap(true)
				height = wrappedHeight(row.label, INNER) + 4
			else
				row.label:SetPoint("TOPRIGHT", row.value, "TOPLEFT", -8, 0)
				row.label:SetWordWrap(false)
			end
			row:SetHeight(height)
			y = y + height
		end
		y = y + GAP
	end

	if view.footer then
		frame.footer:ClearAllPoints()
		frame.footer:SetPoint("TOPLEFT", PAD, -y)
		frame.footer:SetText(view.footer)
		frame.footer:Show()
		y = y + wrappedHeight(frame.footer, INNER) + 4
	else
		frame.footer:Hide()
	end
	local height = y - GAP + PAD
	frame:SetHeight(height)

	local scale = ns.db.panelScale
	local room = UIParent:GetHeight() - 40
	if height * scale > room then
		scale = room / height
	end
	if math.abs(frame:GetScale() - scale) > 0.001 then
		frame:SetScale(scale)
	end
	self:Place()
end

function Detail:SetTab(key)
	if self.current then
		ns.db.tabs[self.current] = key
		self:Render()
	end
end

function Detail:Expand(key)
	self.expanded[key] = true
	self:Render()
end

function Detail:Open(key)
	if not self.frame or not (ns.byKey and ns.byKey[key]) then
		return false
	end
	local wasShown = self.frame:IsShown()
	self.current = key
	self:Render()
	if not wasShown then
		self.frame:Show()
		ns.Tween(self.frame, 0.15, 0, 1, ns.SetAlphaTo)
	end
	ns.Bars:UpdateSelection()
	return true
end

function Detail:Toggle(key)
	if self:IsOpenFor(key) then
		self:Close()
	else
		self:Open(key)
	end
end

function Detail:Close()
	if self.frame then
		self.frame:Hide()
	end
end

function Detail:OnHidden()
	ns.StopTween(self.frame)
	self.frame:SetAlpha(1)
	self.current = nil
	ns.Bars:UpdateSelection()
end

function Detail:IsOpenFor(key)
	return self.frame ~= nil and self.frame:IsShown() and self.current == key
end

function Detail:Refresh()
	if self.frame and self.frame:IsShown() and self.current then
		self:Render()
	end
end

-- Professions changed (skill-up, learned or dropped one).
function Detail:OnProfessionsChanged()
	if self.current and not (ns.byKey and ns.byKey[self.current]) then
		self:Close()
	else
		self:Refresh()
	end
end

function Detail:RestorePosition()
	if self.frame then
		self:Place()
	end
end

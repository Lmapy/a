-- The mind map panel: the profession sits in the middle, branches fan out
-- to both sides on smooth curves, and clicking a branch unfolds its leaves.
local _, ns = ...

local Map = {}
ns.Map = Map

-- Layout metrics (UI units).
local CW, CH = 180, 64 -- centre node
local BW, BH = 108, 36 -- branch nodes
local LW, LH = 150, 20 -- leaf nodes
local G1, G2 = 26, 16 -- horizontal gaps: centre -> branch, branch -> leaf
local VB, VL = 10, 4 -- vertical gaps: between branches, between leaves
local PAD, TOP, FOOT = 14, 26, 22
local MAX_LEAVES = 6
local SIDE = CW / 2 + G1 + BW + G2 + LW
local WIDTH = 2 * SIDE + 2 * PAD
local MASK = "Interface\\CharacterFrame\\TempPortraitAlphaMask"

local GREY = ns.GREY

local function round(v)
	return math.floor(v + 0.5)
end

---------------------------------------------------------------------------
-- Layout (pure: only sets x/y on the tree)
---------------------------------------------------------------------------

local function visibleChildren(branch)
	local kids = branch.children or {}
	if #kids <= MAX_LEAVES then
		return kids
	end
	local shown, rest = {}, {}
	for i = 1, MAX_LEAVES - 1 do
		shown[i] = kids[i]
	end
	for i = MAX_LEAVES, math.min(#kids, MAX_LEAVES + 17) do
		rest[#rest + 1] = { kids[i].label, 0.8, 0.8, 0.8 }
	end
	shown[MAX_LEAVES] = { label = "+" .. (#kids - MAX_LEAVES + 1) .. " more", color = GREY, title = "Also in " .. branch.label, tooltip = rest }
	return shown
end

local function leafBlockHeight(count)
	return count * LH + (count - 1) * VL
end

local function slotHeight(branch)
	if branch.open and #branch.shown > 0 then
		return math.max(BH, leafBlockHeight(#branch.shown))
	end
	return BH
end

local function columnHeight(column)
	local height = 0
	for i, branch in ipairs(column) do
		height = height + slotHeight(branch) + (i > 1 and VB or 0)
	end
	return height
end

local function placeColumn(column, sign)
	local top = columnHeight(column) / 2
	local branchX = sign * (CW / 2 + G1 + BW / 2)
	local leafX = sign * (CW / 2 + G1 + BW + G2 + LW / 2)
	for _, branch in ipairs(column) do
		local height = slotHeight(branch)
		branch.sign, branch.x, branch.y = sign, branchX, top - height / 2
		if branch.open then
			local y = branch.y + leafBlockHeight(#branch.shown) / 2 - LH / 2
			for _, leaf in ipairs(branch.shown) do
				leaf.x, leaf.y = leafX, y
				y = y - LH - VL
			end
		end
		top = top - height - VB
	end
end

-- Split branches into a right and a left column (keeping them balanced),
-- position everything and return the content height.
function Map.Layout(tree)
	local right, left = {}, {}
	for _, branch in ipairs(tree.branches) do
		branch.shown = branch.shown or visibleChildren(branch)
		if branch.side == "left" then
			left[#left + 1] = branch
		else
			right[#right + 1] = branch
		end
	end
	while #left - #right > 1 do
		table.insert(right, table.remove(left))
	end
	while #right - #left > 1 do
		table.insert(left, table.remove(right))
	end
	placeColumn(right, 1)
	placeColumn(left, -1)
	return math.max(columnHeight(right), columnHeight(left), CH)
end

Map.metrics = { CW = CW, CH = CH, BW = BW, BH = BH, LW = LW, LH = LH, WIDTH = WIDTH, TOP = TOP, FOOT = FOOT, PAD = PAD }

---------------------------------------------------------------------------
-- Nodes
---------------------------------------------------------------------------

local function insertLink(link)
	if ChatEdit_InsertLink then
		return ChatEdit_InsertLink(link)
	elseif ChatFrameUtil and ChatFrameUtil.InsertLink then
		return ChatFrameUtil.InsertLink(link)
	end
end

local function showRecipeTooltip(data)
	local recipe = data.recipe
	local ok = false
	if recipe.link then
		ok = pcall(GameTooltip.SetHyperlink, GameTooltip, recipe.link)
	end
	if not ok and type(recipe.id) == "number" and GameTooltip.SetSpellByID then
		ok = pcall(GameTooltip.SetSpellByID, GameTooltip, recipe.id)
	end
	if not ok then
		return false
	end
	if data.count and data.count > 0 then
		GameTooltip:AddLine("You can make " .. data.count .. " from your bags", 0.35, 1, 0.45)
	end
	if recipe.link then
		GameTooltip:AddLine("Shift-click to link it in chat", 0.6, 0.6, 0.65)
	end
	GameTooltip:Show()
	return true
end

local function NodeOnEnter(self)
	local data = self.data
	if not data then
		return
	end
	GameTooltip:SetOwner(self, (self.sign or 1) > 0 and "ANCHOR_RIGHT" or "ANCHOR_LEFT")
	if data.recipe and showRecipeTooltip(data) then
		return
	end
	GameTooltip:AddLine(data.title or data.label, 1, 1, 1)
	for _, entry in ipairs(data.tooltip or {}) do
		GameTooltip:AddLine(entry[1], entry[2], entry[3], entry[4], true)
	end
	if self.kind == "branch" and data.children and #data.children > 0 then
		GameTooltip:AddLine(data.open and "Click to fold this branch" or "Click to unfold this branch", 0.6, 0.6, 0.65)
	end
	GameTooltip:Show()
end

local function NodeOnLeave()
	GameTooltip:Hide()
end

local function NodeOnClick(self)
	local data = self.data
	if self.kind == "branch" then
		Map:ToggleBranch(data.key)
	elseif data and data.recipe and data.recipe.link and IsShiftKeyDown and IsShiftKeyDown() then
		insertLink(data.recipe.link)
	end
end

local function CreateNode(parent)
	local node = CreateFrame("Button", nil, parent)
	node:RegisterForClicks("LeftButtonUp")
	ns.AddBackground(node, 0, 0, 0, 0.8)
	ns.AddBorder(node, 1, 1, 1, 0.1)
	node.stripe = node:CreateTexture(nil, "ARTWORK")
	node.label = node:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	node.label:SetWordWrap(false)
	node.value = node:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	node.value:SetWordWrap(false)
	node.toggle = node:CreateFontString(nil, "OVERLAY", "GameFontNormal")
	node.hl = node:CreateTexture(nil, "HIGHLIGHT")
	node.hl:SetAllPoints()
	node.hl:SetColorTexture(1, 1, 1, 0.07)
	node:SetScript("OnEnter", NodeOnEnter)
	node:SetScript("OnLeave", NodeOnLeave)
	node:SetScript("OnClick", NodeOnClick)
	return node
end

local function resetNode(node)
	ns.StopTween(node)
	node:ClearAllPoints()
	node:SetAlpha(1)
	node.data = nil
end

local function resetLine(line)
	ns.StopTween(line)
	line:SetAlpha(1)
end

function Map:StyleBranch(node, branch, r, g, b)
	node.kind, node.data, node.sign = "branch", branch, branch.sign
	node:SetSize(BW, BH)
	node.bg:SetColorTexture(r * 0.15, g * 0.15, b * 0.15, 0.96)
	ns.SetBorderColor(node, r, g, b, branch.open and 0.9 or 0.4)
	node.stripe:SetColorTexture(r, g, b, 1)
	node.stripe:ClearAllPoints()
	node.label:ClearAllPoints()
	node.value:ClearAllPoints()
	node.toggle:ClearAllPoints()
	node.stripe:SetWidth(3)
	-- Text hugs the side facing the centre; the +/- sits on the outer side.
	if branch.sign > 0 then
		node.stripe:SetPoint("TOPLEFT", 1, -1)
		node.stripe:SetPoint("BOTTOMLEFT", 1, 1)
		node.label:SetPoint("TOPLEFT", 9, -6)
		node.label:SetPoint("TOPRIGHT", -18, -6)
		node.value:SetPoint("BOTTOMLEFT", 9, 6)
		node.value:SetPoint("BOTTOMRIGHT", -18, 6)
		node.toggle:SetPoint("RIGHT", -6, 0)
		node.label:SetJustifyH("LEFT")
		node.value:SetJustifyH("LEFT")
	else
		node.stripe:SetPoint("TOPRIGHT", -1, -1)
		node.stripe:SetPoint("BOTTOMRIGHT", -1, 1)
		node.label:SetPoint("TOPLEFT", 18, -6)
		node.label:SetPoint("TOPRIGHT", -9, -6)
		node.value:SetPoint("BOTTOMLEFT", 18, 6)
		node.value:SetPoint("BOTTOMRIGHT", -9, 6)
		node.toggle:SetPoint("LEFT", 6, 0)
		node.label:SetJustifyH("RIGHT")
		node.value:SetJustifyH("RIGHT")
	end
	node.label:SetText(branch.label)
	node.label:SetTextColor(1, 1, 1)
	local color = branch.valueColor or GREY
	node.value:SetText(branch.value or "")
	node.value:SetTextColor(color[1], color[2], color[3])
	local hasChildren = branch.children and #branch.children > 0
	node.toggle:SetText(hasChildren and (branch.open and "-" or "+") or "")
	node.toggle:SetTextColor(r, g, b)
end

function Map:StyleLeaf(node, leaf, sign, r, g, b)
	node.kind, node.data, node.sign = "leaf", leaf, sign
	node:SetSize(LW, LH)
	node.bg:SetColorTexture(0.075, 0.075, 0.095, 0.95)
	ns.SetBorderColor(node, 1, 1, 1, 0.07)
	local stripe = leaf.stripe or { r, g, b }
	node.stripe:SetColorTexture(stripe[1], stripe[2], stripe[3], 1)
	node.stripe:ClearAllPoints()
	node.label:ClearAllPoints()
	node.value:ClearAllPoints()
	node.toggle:ClearAllPoints()
	node.stripe:SetWidth(2)
	local leftPad, rightPad = 8, 7
	if sign > 0 then
		node.stripe:SetPoint("TOPLEFT", 1, -1)
		node.stripe:SetPoint("BOTTOMLEFT", 1, 1)
	else
		node.stripe:SetPoint("TOPRIGHT", -1, -1)
		node.stripe:SetPoint("BOTTOMRIGHT", -1, 1)
		leftPad, rightPad = 7, 8
	end
	node.value:SetPoint("RIGHT", -rightPad, 0)
	node.value:SetJustifyH("RIGHT")
	node.label:SetPoint("LEFT", leftPad, 0)
	node.label:SetPoint("RIGHT", node.value, "LEFT", -5, 0)
	node.label:SetJustifyH("LEFT")
	local labelColor = leaf.color or ns.WHITE
	node.label:SetText(leaf.label)
	node.label:SetTextColor(labelColor[1], labelColor[2], labelColor[3])
	local valueColor = leaf.valueColor or GREY
	node.value:SetText(leaf.value or "")
	node.value:SetTextColor(valueColor[1], valueColor[2], valueColor[3])
	node.toggle:SetText("")
end

---------------------------------------------------------------------------
-- Centre node
---------------------------------------------------------------------------

local function roundMask(owner, texture)
	if not owner.CreateMaskTexture or not texture.AddMaskTexture then
		return
	end
	pcall(function()
		local mask = owner:CreateMaskTexture()
		mask:SetTexture(MASK, "CLAMPTOBLACKADDITIVE", "CLAMPTOBLACKADDITIVE")
		mask:SetAllPoints(texture)
		texture:AddMaskTexture(mask)
	end)
end

local RING, ICON = 40, 34
local TEXT_X = 10 + RING + 8

local function CreateCenter(parent)
	local c = CreateFrame("Button", nil, parent)
	c:SetSize(CW, CH)
	c:RegisterForClicks("LeftButtonUp")
	ns.AddBackground(c, 0, 0, 0, 0.9)
	ns.AddBorder(c, 1, 1, 1, 1)
	c.ring = c:CreateTexture(nil, "ARTWORK", nil, 1)
	c.ring:SetSize(RING, RING)
	c.ring:SetPoint("LEFT", 10, 0)
	roundMask(c, c.ring)
	c.icon = c:CreateTexture(nil, "ARTWORK", nil, 2)
	c.icon:SetSize(ICON, ICON)
	c.icon:SetPoint("CENTER", c.ring, "CENTER")
	roundMask(c, c.icon)

	local top = (CH - RING) / 2 + 3
	c.name = c:CreateFontString(nil, "OVERLAY", "GameFontNormal")
	c.name:SetPoint("TOPLEFT", TEXT_X, -top)
	c.name:SetPoint("TOPRIGHT", -8, -top)
	c.name:SetJustifyH("LEFT")
	c.name:SetWordWrap(false)
	c.sub = c:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
	c.sub:SetPoint("TOPLEFT", TEXT_X, -(top + 17))
	c.sub:SetPoint("TOPRIGHT", -8, -(top + 17))
	c.sub:SetJustifyH("LEFT")
	c.sub:SetWordWrap(false)

	local bottom = (CH - RING) / 2 + 3
	c.bar = CreateFrame("StatusBar", nil, c)
	c.bar:SetPoint("BOTTOMLEFT", TEXT_X, bottom)
	c.bar:SetPoint("BOTTOMRIGHT", -10, bottom)
	c.bar:SetHeight(4)
	c.bar:SetStatusBarTexture("Interface\\Buttons\\WHITE8X8")
	c.bar:SetMinMaxValues(0, 1)
	c.barBg = c.bar:CreateTexture(nil, "BACKGROUND")
	c.barBg:SetAllPoints()

	c.hl = c:CreateTexture(nil, "HIGHLIGHT")
	c.hl:SetAllPoints()
	c.hl:SetColorTexture(1, 1, 1, 0.05)
	c:SetScript("OnClick", function() Map:ToggleAll() end)
	c:SetScript("OnEnter", function(self)
		GameTooltip:SetOwner(self, "ANCHOR_TOP")
		GameTooltip:AddLine(self.title or "", 1, 1, 1)
		GameTooltip:AddLine("Click to unfold or fold every branch", 0.6, 0.6, 0.65)
		GameTooltip:Show()
	end)
	c:SetScript("OnLeave", NodeOnLeave)
	return c
end

function Map:DrawCenter(prof, step, r, g, b)
	local c = self.center
	c.title = prof.name
	c:ClearAllPoints()
	c:SetPoint("CENTER", self.frame.canvas, "CENTER", 0, 0)
	c.bg:SetColorTexture(r * 0.22, g * 0.22, b * 0.22, 0.97)
	ns.SetBorderColor(c, r, g, b, 1)
	c.ring:SetColorTexture(r, g, b, 1)
	c.icon:SetTexture(prof.icon)
	c.name:SetText(prof.name)
	c.name:SetTextColor(1, 1, 1)
	c.sub:SetText(ns.ColorCode(r, g, b) .. prof.rank .. "/" .. prof.max .. "|r · " .. step.rank.name)
	c.sub:SetTextColor(0.72, 0.72, 0.78)
	c.bar:SetStatusBarColor(r, g, b, 1)
	c.barBg:SetColorTexture(r * 0.25, g * 0.25, b * 0.25, 1)
	c.bar:SetValue(prof.max > 0 and math.min(1, prof.rank / prof.max) or 0)
end

---------------------------------------------------------------------------
-- Panel
---------------------------------------------------------------------------

function Map:Create()
	if self.frame then
		return
	end
	local frame = CreateFrame("Frame", "ProfessionBarsMap", UIParent)
	frame:SetSize(WIDTH, 300)
	frame:SetFrameStrata("HIGH")
	frame:SetToplevel(true)
	frame:SetClampedToScreen(true)
	frame:SetMovable(true)
	frame:EnableMouse(true)
	frame:RegisterForDrag("LeftButton")
	if frame.SetDontSavePosition then
		frame:SetDontSavePosition(true)
	end
	frame:SetScript("OnDragStart", function(f) f:StartMoving() end)
	frame:SetScript("OnDragStop", function(f)
		f:StopMovingOrSizing()
		self:SavePosition()
	end)
	self.frame = frame
	frame:Hide()
	frame:SetScript("OnHide", function() self:OnHidden() end)

	ns.AddBackground(frame, 0.035, 0.035, 0.05, 0.95)
	ns.AddBorder(frame, 1, 1, 1, 0.2)
	frame.accent = frame:CreateTexture(nil, "ARTWORK")
	frame.accent:SetPoint("TOPLEFT", 1, -1)
	frame.accent:SetPoint("TOPRIGHT", -1, -1)
	frame.accent:SetHeight(2)

	frame.hint = frame:CreateFontString(nil, "OVERLAY", "GameFontDisableSmall")
	frame.hint:SetPoint("TOPLEFT", 10, -9)
	frame.hint:SetText("Click a branch to unfold it  ·  drag to move  ·  Esc closes")

	frame.close = ns.CreateCloseButton(frame, 18)
	frame.close:SetPoint("TOPRIGHT", -5, -4)
	frame.close:SetScript("OnClick", function() self:Close() end)

	frame.canvas = CreateFrame("Frame", nil, frame)
	frame.canvas:SetPoint("TOPLEFT", 0, -TOP)
	frame.canvas:SetPoint("BOTTOMRIGHT", 0, FOOT)

	frame.footer = frame:CreateFontString(nil, "OVERLAY", "GameFontDisableSmall")
	frame.footer:SetPoint("BOTTOMLEFT", 10, 8)
	frame.footer:SetPoint("BOTTOMRIGHT", -10, 8)
	frame.footer:SetJustifyH("LEFT")
	frame.footer:SetWordWrap(false)

	local canvas = frame.canvas
	self.lines = ns.NewPool(function() return canvas:CreateLine(nil, "BACKGROUND") end, resetLine)
	self.nodes = ns.NewPool(function() return CreateNode(canvas) end, resetNode)
	self.center = CreateCenter(canvas)

	if UISpecialFrames then
		table.insert(UISpecialFrames, "ProfessionBarsMap")
	end
	self:RestorePosition()
end

function Map:Render()
	local frame = self.frame
	local prof = ns.byKey and ns.byKey[self.current]
	if not frame or not prof then
		return
	end
	local tree = ns.BuildMindMap(prof)
	local saved = ns.db.expanded[prof.key] or {}
	self.openState = {}
	for _, branch in ipairs(tree.branches) do
		if saved[branch.key] == nil then
			branch.open = branch.defaultOpen and true or false
		else
			branch.open = saved[branch.key]
		end
		self.openState[branch.key] = branch.open
	end
	local contentHeight = Map.Layout(tree)
	contentHeight = contentHeight + contentHeight % 2
	local height = TOP + FOOT + 2 * PAD + contentHeight
	frame:SetSize(WIDTH, height)

	-- Shrink to fit the screen if a lot is unfolded.
	local scale = ns.db.mapScale
	local room = UIParent:GetHeight() - 40
	if height * scale > room then
		scale = room / height
	end
	if math.abs(frame:GetScale() - scale) > 0.001 then
		frame:SetScale(scale)
		self:RestorePosition()
	end

	local r, g, b = ns.Color(prof.key)
	frame.bg:SetColorTexture(0.035, 0.035, 0.05, math.min(0.97, ns.GetTheme().panel + 0.4))
	ns.SetBorderColor(frame, r * 0.8, g * 0.8, b * 0.8, 0.9)
	frame.accent:SetColorTexture(r, g, b, 1)
	frame.footer:SetText(tree.footer or "")
	self:DrawCenter(prof, tree.step, r, g, b)

	self.lines:ReleaseAll()
	self.nodes:ReleaseAll()
	local canvas = frame.canvas
	for _, branch in ipairs(tree.branches) do
		local s = branch.sign
		local bx, by = round(branch.x), round(branch.y)
		ns.DrawCurve(self.lines, canvas, s * CW / 2, 0, bx - s * BW / 2, by, r, g, b, 0.8, 2)
		local node = self.nodes:Acquire()
		self:StyleBranch(node, branch, r, g, b)
		node:SetPoint("CENTER", canvas, "CENTER", bx, by)
		if branch.open then
			local fresh = (self.animate == branch.key) and {} or nil
			for _, leaf in ipairs(branch.shown) do
				local lx, ly = round(leaf.x), round(leaf.y)
				ns.DrawCurve(self.lines, canvas, bx + s * BW / 2, by, lx - s * LW / 2, ly, r, g, b, 0.45, 1.5, fresh)
				local leafNode = self.nodes:Acquire()
				self:StyleLeaf(leafNode, leaf, s, r, g, b)
				leafNode:SetPoint("CENTER", canvas, "CENTER", lx, ly)
				if fresh then
					fresh[#fresh + 1] = leafNode
				end
			end
			if fresh then
				for _, object in ipairs(fresh) do
					ns.Tween(object, 0.2, 0, 1, ns.SetAlphaTo)
				end
			end
		end
	end
	self.animate = nil
	self.tree = tree
end

function Map:ToggleBranch(branchKey)
	local key = self.current
	if not key then
		return
	end
	local saved = ns.db.expanded[key] or {}
	ns.db.expanded[key] = saved
	local open = not (self.openState and self.openState[branchKey])
	saved[branchKey] = open
	if open then
		self.animate = branchKey
	end
	self:Render()
end

function Map:ToggleAll()
	local key = self.current
	if not key or not self.openState then
		return
	end
	local anyOpen = false
	for _, open in pairs(self.openState) do
		anyOpen = anyOpen or open
	end
	local saved = {}
	for branchKey in pairs(self.openState) do
		saved[branchKey] = not anyOpen
	end
	ns.db.expanded[key] = saved
	self:Render()
end

function Map:Open(key)
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

function Map:Toggle(key)
	if self:IsOpenFor(key) then
		self:Close()
	else
		self:Open(key)
	end
end

function Map:Close()
	if self.frame then
		self.frame:Hide()
	end
end

function Map:OnHidden()
	ns.StopTween(self.frame)
	self.frame:SetAlpha(1)
	self.current = nil
	ns.Bars:UpdateSelection()
end

function Map:IsOpenFor(key)
	return self.frame ~= nil and self.frame:IsShown() and self.current == key
end

function Map:Refresh()
	if self.frame and self.frame:IsShown() and self.current then
		self:Render()
	end
end

-- Professions changed (skill-up, learned or dropped one).
function Map:OnProfessionsChanged()
	if self.current and not (ns.byKey and ns.byKey[self.current]) then
		self:Close()
	else
		self:Refresh()
	end
end

function Map:SavePosition()
	local frame = self.frame
	local x, y = frame:GetCenter()
	if x and y then
		local scale = frame:GetEffectiveScale()
		ns.db.mapPoint = { x * scale, y * scale }
	end
	self:RestorePosition()
end

function Map:RestorePosition()
	local frame = self.frame
	frame:ClearAllPoints()
	local point = ns.db.mapPoint
	if point then
		local scale = frame:GetEffectiveScale()
		frame:SetPoint("CENTER", UIParent, "BOTTOMLEFT", point[1] / scale, point[2] / scale)
	else
		frame:SetPoint("CENTER", UIParent, "CENTER", 0, 40)
	end
end

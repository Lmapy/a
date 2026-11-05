-- Headless stand-in for the slice of the WoW UI API our addons use.
--
-- It is deliberately strict where the real client is strict (unknown events
-- error, unknown methods are missing) and it records every frame, anchor,
-- texture, line and string so tests can assert on them. Dump() resolves the
-- anchors to screen rectangles so a layout can be rendered outside the game.
--
-- Usage: local Mock = dofile("tools/wowmock/wow.lua"); local m = Mock.new()
--        m.LoadAddon("addons/ProfessionBars", "ProfessionBars")
local Mock = {}

local POINTS = {
	TOPLEFT = { 0, 1 }, TOP = { 0.5, 1 }, TOPRIGHT = { 1, 1 },
	LEFT = { 0, 0.5 }, CENTER = { 0.5, 0.5 }, RIGHT = { 1, 0.5 },
	BOTTOMLEFT = { 0, 0 }, BOTTOM = { 0.5, 0 }, BOTTOMRIGHT = { 1, 0 },
}
local LAYERS = { BACKGROUND = 1, BORDER = 2, ARTWORK = 3, OVERLAY = 4, HIGHLIGHT = 5 }
local STRATA = { BACKGROUND = 1, LOW = 2, MEDIUM = 3, HIGH = 4, DIALOG = 5, FULLSCREEN = 6, FULLSCREEN_DIALOG = 7, TOOLTIP = 8 }
local FONTS = {
	GameFontNormal = { 12, 1, 0.82, 0 }, GameFontNormalSmall = { 10, 1, 0.82, 0 }, GameFontNormalLarge = { 16, 1, 0.82, 0 },
	GameFontHighlight = { 12, 1, 1, 1 }, GameFontHighlightSmall = { 10, 1, 1, 1 }, GameFontHighlightLarge = { 16, 1, 1, 1 },
	GameFontDisable = { 12, 0.5, 0.5, 0.5 }, GameFontDisableSmall = { 10, 0.5, 0.5, 0.5 },
}
-- Events that exist on a Mainline-based client (LEARNED_SPELL_IN_TAB and the
-- Classic CRAFT_* events deliberately do not).
local KNOWN_EVENTS = {}
for _, name in ipairs({
	"ADDON_LOADED", "PLAYER_LOGIN", "PLAYER_ENTERING_WORLD", "SKILL_LINES_CHANGED", "CHAT_MSG_SKILL",
	"LEARNED_SPELL_IN_SKILL_LINE", "PLAYER_LEVEL_UP", "TRADE_SKILL_SHOW", "TRADE_SKILL_CLOSE",
	"TRADE_SKILL_LIST_UPDATE", "TRADE_SKILL_DATA_SOURCE_CHANGED", "BAG_UPDATE_DELAYED",
	"PLAYER_REGEN_DISABLED", "PLAYER_REGEN_ENABLED",
}) do
	KNOWN_EVENTS[name] = true
end

local function compose(...)
	local out = {}
	for _, source in ipairs({ ... }) do
		for k, v in pairs(source) do
			out[k] = v
		end
	end
	return out
end

local function stripCodes(text)
	return (text:gsub("|c%x%x%x%x%x%x%x%x", ""):gsub("|r", ""))
end

local function utf8len(text)
	local n = 0
	for i = 1, #text do
		local byte = text:byte(i)
		if byte < 0x80 or byte >= 0xC0 then
			n = n + 1
		end
	end
	return n
end

function Mock.new(options)
	options = options or {}
	local m = {
		W = options.width or 1366, H = options.height or 768,
		now = 0, chat = {}, warnings = {}, level = 1, inCombat = false, shift = false,
		bags = {}, mouseOver = nil,
	}
	local env = setmetatable({}, { __index = _G })
	m.env = env
	local all, eventMap, timers = {}, {}, {}
	local seq = 0

	local function warn(message)
		m.warnings[#m.warnings + 1] = message
	end

	-- Shared region behaviour ------------------------------------------------
	local Region = {}
	function Region:GetObjectType() return self.kind end
	function Region:GetParent() return self.parent end
	function Region:GetName() return self.name end
	function Region:SetPoint(point, a, b, c, d)
		local rel, relPoint, x, y
		if a == nil then
			rel, relPoint, x, y = self.parent, point, 0, 0
		elseif type(a) == "number" then
			rel, relPoint, x, y = self.parent, point, a, b
		else
			rel = type(a) == "string" and env[a] or a
			if type(b) == "string" then
				relPoint, x, y = b, c, d
			elseif type(b) == "number" then
				relPoint, x, y = point, b, c
			else
				relPoint, x, y = point, 0, 0
			end
		end
		assert(POINTS[point], "SetPoint: bad point " .. tostring(point))
		assert(POINTS[relPoint], "SetPoint: bad relative point " .. tostring(relPoint))
		assert(type(rel) == "table", "SetPoint: no frame to anchor to")
		assert(rel ~= self, "SetPoint: anchored to itself")
		local entry = { point = point, rel = rel, relPoint = relPoint, x = x or 0, y = y or 0 }
		for i, existing in ipairs(self.points) do
			if existing.point == point then
				self.points[i] = entry
				return
			end
		end
		self.points[#self.points + 1] = entry
	end
	function Region:ClearAllPoints() self.points = {} end
	function Region:SetAllPoints(rel)
		self.points = {}
		rel = rel or self.parent
		self:SetPoint("TOPLEFT", rel, "TOPLEFT", 0, 0)
		self:SetPoint("BOTTOMRIGHT", rel, "BOTTOMRIGHT", 0, 0)
	end
	function Region:GetNumPoints() return #self.points end
	function Region:GetPoint(i)
		local p = self.points[i or 1]
		if p then
			return p.point, p.rel, p.relPoint, p.x, p.y
		end
	end
	function Region:SetSize(w, h) self.w, self.h = w, h end
	function Region:SetWidth(w) self.w = w end
	function Region:SetHeight(h) self.h = h end
	function Region:GetWidth()
		local rect = m.Resolve(self)
		return rect and (rect.r - rect.l) / m.EffectiveScale(self) or (self.w or 0)
	end
	function Region:GetHeight()
		local rect = m.Resolve(self)
		return rect and (rect.t - rect.b) / m.EffectiveScale(self) or (self.h or 0)
	end
	function Region:GetSize() return self:GetWidth(), self:GetHeight() end
	local function edge(key)
		return function(self)
			local rect = m.Resolve(self)
			return rect and rect[key] / m.EffectiveScale(self) or nil
		end
	end
	Region.GetLeft, Region.GetRight, Region.GetTop, Region.GetBottom = edge("l"), edge("r"), edge("t"), edge("b")
	function Region:GetCenter()
		local rect = m.Resolve(self)
		if rect then
			local s = m.EffectiveScale(self)
			return (rect.l + rect.r) / 2 / s, (rect.b + rect.t) / 2 / s
		end
	end
	function Region:Show() m.SetShown(self, true) end
	function Region:Hide() m.SetShown(self, false) end
	function Region:SetShown(value) m.SetShown(self, value and true or false) end
	function Region:IsShown() return self.shown end
	function Region:IsVisible() return m.IsVisible(self) end
	function Region:SetAlpha(alpha) self.alpha = alpha end
	function Region:GetAlpha() return self.alpha end
	function Region:IsMouseOver() return m.mouseOver ~= nil and (m.mouseOver == self or m.IsAncestor(self, m.mouseOver)) end
	function Region:GetEffectiveScale() return m.EffectiveScale(self) end

	local Drawable = {}
	function Drawable:SetDrawLayer(layer, sublevel) self.layer, self.sublevel = layer, sublevel end
	function Drawable:SetVertexColor(r, g, b, a) self.vertex = { r, g, b, a or 1 } end
	function Drawable:SetColorTexture(r, g, b, a)
		assert(type(r) == "number" and type(g) == "number" and type(b) == "number", "SetColorTexture needs numbers")
		self.color, self.texturePath = { r, g, b, a or 1 }, nil
	end

	local TextureOnly = {}
	function TextureOnly:SetTexture(path) self.texturePath, self.color = path, nil end
	function TextureOnly:GetTexture() return self.texturePath end
	function TextureOnly:SetTexCoord() end
	function TextureOnly:SetDesaturated(value) self.desaturated = value end
	function TextureOnly:SetBlendMode() end
	function TextureOnly:AddMaskTexture(mask)
		self.masks = self.masks or {}
		self.masks[#self.masks + 1] = mask
	end

	local LineOnly = {}
	function LineOnly:SetStartPoint(relPoint, rel, x, y) self.from = { relPoint, rel, x or 0, y or 0 } end
	function LineOnly:SetEndPoint(relPoint, rel, x, y) self.to = { relPoint, rel, x or 0, y or 0 } end
	function LineOnly:SetThickness(t) self.thickness = t end

	local FontOnly = {}
	function FontOnly:SetFontObject(font)
		assert(type(font) == "table" and font.size, "SetFontObject: not a font object")
		self.font = font
		self.textColor = { font.r, font.g, font.b, 1 }
	end
	function FontOnly:SetFont(path, size, flags) self.font = { size = size, r = 1, g = 1, b = 1, path = path, flags = flags } return true end
	function FontOnly:GetFont() return "Fonts\\FRIZQT__.TTF", self.font.size end
	function FontOnly:SetText(text) self.text = text ~= nil and tostring(text) or nil end
	function FontOnly:GetText() return self.text end
	function FontOnly:SetFormattedText(fmt, ...) self.text = string.format(fmt, ...) end
	function FontOnly:SetTextColor(r, g, b, a) self.textColor = { r, g, b, a or 1 } end
	function FontOnly:GetTextColor() return unpack(self.textColor) end
	function FontOnly:SetJustifyH(value) assert(value == "LEFT" or value == "RIGHT" or value == "CENTER", "bad JustifyH") self.justifyH = value end
	function FontOnly:SetJustifyV(value) self.justifyV = value end
	function FontOnly:SetWordWrap(value) self.wordWrap = value end
	function FontOnly:SetNonSpaceWrap() end
	function FontOnly:SetMaxLines() end
	function FontOnly:SetShadowOffset() end
	function FontOnly:SetShadowColor() end
	function FontOnly:SetSpacing() end
	function FontOnly:GetStringWidth()
		local text = self.text and stripCodes(self.text) or ""
		return utf8len(text) * self.font.size * 0.52
	end
	function FontOnly:GetStringHeight() return self.font.size + 2 end

	local MaskOnly = {}
	function MaskOnly:SetTexture(path) self.texturePath = path end

	-- Frames -------------------------------------------------------------------
	local FrameOnly = {}
	function FrameOnly:SetScript(name, fn) self.scripts[name] = fn end
	function FrameOnly:GetScript(name) return self.scripts[name] end
	function FrameOnly:HookScript(name, fn)
		local old = self.scripts[name]
		self.scripts[name] = old and function(...) old(...) fn(...) end or fn
	end
	function FrameOnly:RegisterEvent(name)
		if not KNOWN_EVENTS[name] then
			error('Attempt to register unknown event "' .. tostring(name) .. '"', 2)
		end
		eventMap[name] = eventMap[name] or {}
		for _, frame in ipairs(eventMap[name]) do
			if frame == self then
				return
			end
		end
		table.insert(eventMap[name], self)
	end
	function FrameOnly:UnregisterEvent(name)
		for i, frame in ipairs(eventMap[name] or {}) do
			if frame == self then
				table.remove(eventMap[name], i)
				return
			end
		end
	end
	function FrameOnly:IsEventRegistered(name)
		for _, frame in ipairs(eventMap[name] or {}) do
			if frame == self then
				return true
			end
		end
		return false
	end
	function FrameOnly:EnableMouse(value) self.mouse = value end
	function FrameOnly:IsMouseEnabled() return self.mouse end
	function FrameOnly:SetMovable(value) self.movable = value end
	function FrameOnly:IsMovable() return self.movable end
	function FrameOnly:RegisterForDrag() end
	function FrameOnly:StartMoving()
		assert(self.movable, "StartMoving on a frame that is not movable")
		self.moving = true
	end
	function FrameOnly:StopMovingOrSizing()
		-- The client re-anchors a moved frame to its TOPLEFT on screen.
		if self.moving then
			local rect = m.Resolve(self)
			local s = m.EffectiveScale(self)
			self.moving = nil
			self.points = {}
			self:SetPoint("TOPLEFT", m.UIParent, "BOTTOMLEFT", rect.l / s + (m.dragDX or 0), rect.t / s + (m.dragDY or 0))
		end
	end
	function FrameOnly:SetClampedToScreen(value) self.clamped = value end
	function FrameOnly:SetToplevel() end
	function FrameOnly:SetUserPlaced() end
	function FrameOnly:SetDontSavePosition() end
	function FrameOnly:SetFrameStrata(strata) assert(STRATA[strata], "bad strata") self.strata = strata end
	function FrameOnly:GetFrameStrata() return m.Strata(self) end
	function FrameOnly:SetFrameLevel(level) self.level = level end
	function FrameOnly:GetFrameLevel() return m.Level(self) end
	function FrameOnly:SetScale(scale) assert(scale > 0, "scale must be positive") self.scale = scale end
	function FrameOnly:GetScale() return self.scale end
	function FrameOnly:GetChildren() return unpack(self.frameChildren) end
	function FrameOnly:SetHitRectInsets() end
	function FrameOnly:CreateTexture(name, layer, template, sublevel)
		local texture = m.NewObject("Texture", self, compose(Region, Drawable, TextureOnly))
		texture.layer, texture.sublevel = layer or "ARTWORK", sublevel or 0
		if name then env[name] = texture end
		return texture
	end
	function FrameOnly:CreateMaskTexture()
		return m.NewObject("MaskTexture", self, compose(Region, MaskOnly))
	end
	function FrameOnly:CreateLine(name, layer, template, sublevel)
		local line = m.NewObject("Line", self, compose(Region, Drawable, TextureOnly, LineOnly))
		line.layer, line.sublevel, line.thickness = layer or "ARTWORK", sublevel or 0, 1
		return line
	end
	function FrameOnly:CreateFontString(name, layer, template)
		local fs = m.NewObject("FontString", self, compose(Region, FontOnly))
		fs.layer = layer or "ARTWORK"
		fs:SetFontObject(env[template] or env.GameFontNormal)
		if name then env[name] = fs end
		return fs
	end

	local ButtonOnly = {}
	function ButtonOnly:RegisterForClicks() end
	function ButtonOnly:Click(button) m.Click(self, button) end
	function ButtonOnly:Enable() self.disabled = false end
	function ButtonOnly:Disable() self.disabled = true end
	function ButtonOnly:IsEnabled() return not self.disabled end
	function ButtonOnly:SetEnabled(value) self.disabled = not value end

	local BarOnly = {}
	function BarOnly:SetStatusBarTexture(texture) self.barTexture = texture end
	function BarOnly:GetStatusBarTexture() return self.barTexture end
	function BarOnly:SetStatusBarColor(r, g, b, a) self.barColor = { r, g, b, a or 1 } end
	function BarOnly:SetMinMaxValues(low, high) self.minValue, self.maxValue = low, high end
	function BarOnly:GetMinMaxValues() return self.minValue, self.maxValue end
	function BarOnly:SetValue(value)
		assert(type(value) == "number", "SetValue needs a number")
		self.value = math.max(self.minValue, math.min(self.maxValue, value))
	end
	function BarOnly:GetValue() return self.value end
	function BarOnly:SetOrientation() end

	local TooltipOnly = {}
	function TooltipOnly:SetOwner(owner, anchor) self.owner, self.anchor, self.lines = owner, anchor, {} end
	function TooltipOnly:ClearLines() self.lines = {} end
	function TooltipOnly:AddLine(text) assert(text ~= nil, "AddLine(nil)") table.insert(self.lines, tostring(text)) end
	function TooltipOnly:AddDoubleLine(left, right) table.insert(self.lines, tostring(left) .. " | " .. tostring(right)) end
	function TooltipOnly:SetText(text) self.lines = { tostring(text) } end
	function TooltipOnly:NumLines() return #self.lines end
	function TooltipOnly:SetHyperlink(link)
		assert(type(link) == "string" and link:find("|H"), "SetHyperlink: not a link")
		table.insert(self.lines, "[link] " .. link)
	end
	function TooltipOnly:SetSpellByID(id) table.insert(self.lines, "[spell] " .. tostring(id)) end

	local CLASSES = {
		Frame = compose(Region, FrameOnly),
		Button = compose(Region, FrameOnly, ButtonOnly),
		StatusBar = compose(Region, FrameOnly, BarOnly),
		GameTooltip = compose(Region, FrameOnly, TooltipOnly),
	}

	function m.NewObject(kind, parent, methods)
		seq = seq + 1
		local object = setmetatable({ kind = kind, parent = parent, points = {}, shown = true, alpha = 1, id = seq }, { __index = methods })
		all[#all + 1] = object
		return object
	end

	function m.CreateFrame(kind, name, parent, template)
		local methods = CLASSES[kind]
		assert(methods, "CreateFrame: unsupported frame type " .. tostring(kind))
		if template and template ~= "BackdropTemplate" then
			error("CreateFrame: unknown template " .. tostring(template))
		end
		local frame = m.NewObject(kind, parent, methods)
		frame.isFrame, frame.scripts, frame.frameChildren, frame.scale, frame.name = true, {}, {}, 1, name
		if kind == "StatusBar" then
			frame.minValue, frame.maxValue, frame.value = 0, 1, 0
		end
		if parent then
			table.insert(parent.frameChildren, frame)
		end
		if name then
			env[name] = frame
		end
		return frame
	end

	-- Visibility, scale, strata ------------------------------------------------
	function m.IsAncestor(ancestor, object)
		object = object and object.parent
		while object do
			if object == ancestor then
				return true
			end
			object = object.parent
		end
		return false
	end

	function m.IsVisible(object)
		while object do
			if not object.shown then
				return false
			end
			object = object.parent
		end
		return true
	end

	local function fireTree(frame, script)
		if frame.scripts[script] then
			frame.scripts[script](frame)
		end
		for _, child in ipairs(frame.frameChildren) do
			if child.shown then
				fireTree(child, script)
			end
		end
	end

	function m.SetShown(object, shown)
		local before = m.IsVisible(object)
		object.shown = shown
		if object.isFrame and before ~= m.IsVisible(object) then
			fireTree(object, shown and "OnShow" or "OnHide")
		end
	end

	function m.EffectiveScale(object)
		local scale = 1
		while object do
			if object.isFrame then
				scale = scale * object.scale
			end
			object = object.parent
		end
		return scale
	end

	function m.Strata(frame)
		while frame do
			if frame.strata then
				return frame.strata
			end
			frame = frame.parent
		end
		return "MEDIUM"
	end

	function m.Level(frame)
		if frame.level then
			return frame.level
		end
		return frame.parent and (m.Level(frame.parent) + 1) or 0
	end

	-- Anchor resolution --------------------------------------------------------
	local function measure(object, axis)
		local size = axis == "x" and object.w or object.h
		if size then
			return size
		end
		if object.kind == "FontString" then
			return axis == "x" and object:GetStringWidth() or object:GetStringHeight()
		end
		return 0
	end

	-- Screen rectangle {l, r, b, t} of a region, or nil when not anchored.
	function m.Resolve(object, cache)
		if object == m.UIParent then
			return { l = 0, r = m.W, b = 0, t = m.H }
		end
		cache = cache or {}
		if cache[object] ~= nil then
			return cache[object] or nil
		end
		cache[object] = false
		if #object.points == 0 then
			return nil
		end
		local scale = m.EffectiveScale(object)
		local xs, ys = {}, {}
		for _, p in ipairs(object.points) do
			local rel = m.Resolve(p.rel, cache)
			if not rel then
				return nil
			end
			local relAnchor, ownAnchor = POINTS[p.relPoint], POINTS[p.point]
			local ax = rel.l + (rel.r - rel.l) * relAnchor[1] + p.x * scale
			local ay = rel.b + (rel.t - rel.b) * relAnchor[2] + p.y * scale
			local hk = ownAnchor[1] == 0 and "l" or (ownAnchor[1] == 1 and "r" or "c")
			local vk = ownAnchor[2] == 0 and "b" or (ownAnchor[2] == 1 and "t" or "c")
			if xs[hk] and math.abs(xs[hk] - ax) > 0.01 then
				warn(object.kind .. " has two conflicting horizontal anchors")
			end
			if ys[vk] and math.abs(ys[vk] - ay) > 0.01 then
				warn(object.kind .. " has two conflicting vertical anchors")
			end
			xs[hk], ys[vk] = ax, ay
		end
		if xs.c and (xs.l or xs.r) then
			warn(object.kind .. " is over-constrained horizontally (edge + centre anchors)")
		end
		if ys.c and (ys.t or ys.b) then
			warn(object.kind .. " is over-constrained vertically (edge + centre anchors)")
		end
		local w, h = measure(object, "x") * scale, measure(object, "y") * scale
		local l, r = xs.l, xs.r
		if l and not r then
			r = l + w
		elseif r and not l then
			l = r - w
		elseif not l then
			l, r = xs.c - w / 2, xs.c + w / 2
		end
		local b, t = ys.b, ys.t
		if b and not t then
			t = b + h
		elseif t and not b then
			b = t - h
		elseif not b then
			b, t = ys.c - h / 2, ys.c + h / 2
		end
		-- Clamped frames are nudged back onto the screen, like the client does.
		if object.clamped then
			local dx = (l < 0 and -l) or (r > m.W and m.W - r) or 0
			local dy = (b < 0 and -b) or (t > m.H and m.H - t) or 0
			l, r, b, t = l + dx, r + dx, b + dy, t + dy
		end
		local rect = { l = l, r = r, b = b, t = t }
		cache[object] = rect
		return rect
	end

	local function linePoint(line, spec, cache)
		local rel = m.Resolve(spec[2], cache)
		if not rel then
			return nil
		end
		local anchor = POINTS[spec[1]]
		local scale = m.EffectiveScale(line)
		return rel.l + (rel.r - rel.l) * anchor[1] + spec[3] * scale, rel.b + (rel.t - rel.b) * anchor[2] + spec[4] * scale
	end

	local function opacity(object)
		local alpha = 1
		while object do
			alpha = alpha * object.alpha
			object = object.parent
		end
		return alpha
	end

	-- Every visible drawable as a flat, back-to-front list.
	function m.Dump()
		local ops, cache = {}, {}
		local function base(object, frame, layer, sublevel)
			return {
				strata = STRATA[m.Strata(frame)], level = m.Level(frame), layer = LAYERS[layer] or 3,
				sub = sublevel or 0, seq = object.id, alpha = opacity(object),
			}
		end
		for _, object in ipairs(all) do
			local frame = object.parent
			local hovered = object.layer == "HIGHLIGHT" and m.mouseOver ~= nil and (m.mouseOver == frame)
			local drawable = not object.isFrame and object.kind ~= "MaskTexture" and (object.layer ~= "HIGHLIGHT" or hovered)
			if drawable and m.IsVisible(object) then
				local op = base(object, frame, object.layer, object.sublevel)
				if object.kind == "Texture" then
					local rect = m.Resolve(object, cache)
					if rect and (object.color or object.texturePath) then
						op.type, op.rect, op.color, op.path = "tex", rect, object.color, object.texturePath
						op.vertex, op.round, op.desaturated = object.vertex, object.masks ~= nil, object.desaturated
						ops[#ops + 1] = op
					end
				elseif object.kind == "FontString" then
					local rect = m.Resolve(object, cache)
					if rect and object.text and object.text ~= "" then
						op.type, op.rect, op.text = "text", rect, object.text
						op.size = object.font.size * m.EffectiveScale(object)
						op.color, op.justifyH, op.wrap = object.textColor, object.justifyH or "CENTER", object.wordWrap ~= false
						op.fixedWidth = (#object.points > 1 and (object.points[1].point ~= object.points[2].point)) or object.w ~= nil
						local hx = POINTS[object.points[1].point][1]
						op.anchorH = hx == 0 and "LEFT" or (hx == 1 and "RIGHT" or "CENTER")
						ops[#ops + 1] = op
					end
				elseif object.kind == "Line" and object.from and object.to and object.color then
					local x1, y1 = linePoint(object, object.from, cache)
					local x2, y2 = linePoint(object, object.to, cache)
					if x1 and x2 then
						op.type, op.x1, op.y1, op.x2, op.y2 = "line", x1, y1, x2, y2
						op.thickness, op.color = object.thickness * m.EffectiveScale(object), object.color
						ops[#ops + 1] = op
					end
				end
			end
			if object.kind == "StatusBar" and object.barColor and m.IsVisible(object) then
				local rect = m.Resolve(object, cache)
				if rect then
					local span = object.maxValue - object.minValue
					local fraction = span > 0 and (object.value - object.minValue) / span or 0
					local op = base(object, object, "ARTWORK", -1)
					op.type, op.color = "tex", object.barColor
					op.rect = { l = rect.l, r = rect.l + (rect.r - rect.l) * fraction, b = rect.b, t = rect.t }
					ops[#ops + 1] = op
				end
			end
		end
		table.sort(ops, function(a, b)
			if a.strata ~= b.strata then return a.strata < b.strata end
			if a.level ~= b.level then return a.level < b.level end
			if a.layer ~= b.layer then return a.layer < b.layer end
			if a.sub ~= b.sub then return a.sub < b.sub end
			return a.seq < b.seq
		end)
		return ops
	end

	-- Driving the client -------------------------------------------------------
	function m.Fire(event, ...)
		local list = {}
		for i, frame in ipairs(eventMap[event] or {}) do
			list[i] = frame
		end
		for _, frame in ipairs(list) do
			if frame.scripts.OnEvent then
				frame.scripts.OnEvent(frame, event, ...)
			end
		end
	end

	function m.RunScript(frame, name, ...)
		local script = frame.scripts[name]
		if script then
			return script(frame, ...)
		end
	end

	function m.Click(frame, button)
		return m.RunScript(frame, "OnClick", button or "LeftButton", false)
	end

	function m.Hover(frame)
		if m.mouseOver and m.mouseOver ~= frame then
			m.Leave()
		end
		m.mouseOver = frame
		m.RunScript(frame, "OnEnter", false)
	end

	function m.Leave()
		local frame = m.mouseOver
		m.mouseOver = nil
		if frame then
			m.RunScript(frame, "OnLeave", false)
		end
	end

	function m.Advance(seconds)
		local step = 1 / 60
		local elapsed = 0
		while elapsed < seconds - 1e-9 do
			local dt = math.min(step, seconds - elapsed)
			m.now = m.now + dt
			elapsed = elapsed + dt
			local fired = true
			while fired do
				fired = false
				table.sort(timers, function(a, b) if a.at ~= b.at then return a.at < b.at end return a.seq < b.seq end)
				if timers[1] and timers[1].at <= m.now + 1e-9 then
					local timer = table.remove(timers, 1)
					timer.fn()
					fired = true
				end
			end
			for _, object in ipairs(all) do
				if object.isFrame and object.scripts.OnUpdate and m.IsVisible(object) then
					object.scripts.OnUpdate(object, dt)
				end
			end
		end
	end

	-- Every frame/region of a kind that is currently visible (test helper).
	function m.Visible(kind, filter)
		local out = {}
		for _, object in ipairs(all) do
			if object.kind == kind and m.IsVisible(object) and (not filter or filter(object)) then
				out[#out + 1] = object
			end
		end
		return out
	end

	-- Globals ------------------------------------------------------------------
	for name, spec in pairs(FONTS) do
		env[name] = { size = spec[1], r = spec[2], g = spec[3], b = spec[4], name = name }
	end
	m.UIParent = m.CreateFrame("Frame", "UIParent")
	m.UIParent.w, m.UIParent.h = m.W, m.H
	env.WorldFrame = m.UIParent
	env.GameTooltip = m.CreateFrame("GameTooltip", "GameTooltip", m.UIParent)
	env.GameTooltip.lines = {}
	env.GameTooltip:SetFrameStrata("TOOLTIP")
	env.GameTooltip:Hide()
	env.CreateFrame = m.CreateFrame
	env.DEFAULT_CHAT_FRAME = { AddMessage = function(_, text) m.chat[#m.chat + 1] = text end }
	env.print = function(...) m.chat[#m.chat + 1] = table.concat({ ... }, " ") end
	env.C_Timer = {
		After = function(delay, fn)
			assert(type(fn) == "function", "C_Timer.After needs a function")
			seq = seq + 1
			timers[#timers + 1] = { at = m.now + delay, fn = fn, seq = seq }
		end,
	}
	env.GetTime = function() return m.now end
	env.time = function() return 1793750400 + math.floor(m.now) end
	env.UnitLevel = function() return m.level end
	env.InCombatLockdown = function() return m.inCombat end
	env.IsShiftKeyDown = function() return m.shift end
	env.GetBuildInfo = function() return "1.60.1", "99999", "Nov 1 2026", 16001 end
	env.SlashCmdList = {}
	env.UISpecialFrames = {}
	env.Enum = {
		TradeskillRelativeDifficulty = { Optimal = 0, Medium = 1, Easy = 2, Trivial = 3 },
		CraftingReagentType = { Modifying = 0, Basic = 1, Finishing = 2 },
	}
	env.C_Item = { GetItemCount = function(itemID) return m.bags[itemID] or 0 end }
	env.ChatEdit_InsertLink = function(link) m.insertedLink = link return true end

	-- Load an addon folder in TOC order, passing (name, namespace) like the client.
	function m.LoadAddon(folder, name)
		local ns = {}
		local toc = assert(io.open(folder .. "/" .. name .. ".toc"))
		for line in toc:lines() do
			line = line:gsub("\r", "")
			if line ~= "" and not line:match("^#") then
				local chunk = assert(loadfile(folder .. "/" .. line))
				setfenv(chunk, env)
				chunk(name, ns)
			end
		end
		toc:close()
		return ns
	end

	-- The chat line that contains `needle`, if any.
	function m.ChatHas(needle)
		for _, line in ipairs(m.chat) do
			if line:find(needle, 1, true) then
				return line
			end
		end
	end

	return m
end

-- Tiny JSON encoder for Dump() output.
function Mock.ToJSON(value)
	local kind = type(value)
	if kind == "nil" then
		return "null"
	elseif kind == "boolean" then
		return value and "true" or "false"
	elseif kind == "number" then
		if value ~= value or value == math.huge or value == -math.huge then
			return "null"
		end
		return string.format("%.3f", value):gsub("%.?0+$", "")
	elseif kind == "string" then
		return '"' .. value:gsub('[%c"\\]', function(c)
			return string.format("\\u%04x", c:byte())
		end) .. '"'
	elseif kind == "table" then
		if #value > 0 or next(value) == nil then
			local parts = {}
			for i, item in ipairs(value) do
				parts[i] = Mock.ToJSON(item)
			end
			return "[" .. table.concat(parts, ",") .. "]"
		end
		local parts = {}
		for k, v in pairs(value) do
			if type(v) ~= "table" or k ~= "rel" then
				parts[#parts + 1] = Mock.ToJSON(tostring(k)) .. ":" .. Mock.ToJSON(v)
			end
		end
		return "{" .. table.concat(parts, ",") .. "}"
	end
	return '"' .. tostring(value) .. '"'
end

return Mock

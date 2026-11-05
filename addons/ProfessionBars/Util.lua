-- Small shared helpers: safe API values, printing, throttling, tweens,
-- object pools and flat "pixel" widgets.
local _, ns = ...

-- Modern clients can hand out "secret" values (mostly in combat). Treat
-- them as missing instead of erroring on them.
function ns.Plain(value)
	if value ~= nil and issecretvalue then
		local ok, secret = pcall(issecretvalue, value)
		if ok and secret then
			return nil
		end
	end
	return value
end

function ns.PlainNumber(value)
	value = ns.Plain(value)
	if type(value) == "number" then
		return value
	elseif type(value) == "string" then
		return tonumber(value)
	end
	return nil
end

function ns.PlainString(value)
	value = ns.Plain(value)
	if type(value) == "string" and value ~= "" then
		return value
	end
	return nil
end

function ns.Print(message)
	local text = "|cff66ccffProfession Bars|r: " .. tostring(message)
	if DEFAULT_CHAT_FRAME then
		DEFAULT_CHAT_FRAME:AddMessage(text)
	else
		print(text)
	end
end

-- Run `fn`, containing any error: it is reported once per label (and kept
-- for /pb debug) instead of breaking the UI or repeating on every event.
local reported = {}
function ns.SafeCall(label, fn, ...)
	local ok, err = pcall(fn, ...)
	if not ok then
		ns.lastError = label .. ": " .. tostring(err)
		if not reported[label] then
			reported[label] = true
			ns.Print("something failed while " .. label .. "; everything else keeps working. /pb debug shows the error.")
		end
	end
	return ok
end

-- Run `fn` once, `delay` seconds after the first request; extra requests
-- in that window are merged into the same run.
local pending = {}
function ns.Throttle(key, delay, fn)
	if pending[key] then
		return
	end
	pending[key] = true
	C_Timer.After(delay, function()
		pending[key] = nil
		ns.SafeCall("updating " .. key, fn)
	end)
end

-- Tweens: ns.Tween(object, seconds, from, to, setter[, onDone]).
-- One tween per object; starting a new one replaces the old.
local tweens = {}
local driver = CreateFrame("Frame")
driver:Hide()
driver:SetScript("OnUpdate", function(self, elapsed)
	local done
	for object, tween in pairs(tweens) do
		tween.t = tween.t + elapsed
		local progress = math.min(1, tween.t / tween.duration)
		local eased = 1 - (1 - progress) * (1 - progress)
		tween.set(object, tween.from + (tween.to - tween.from) * eased)
		if progress >= 1 then
			tweens[object] = nil
			if tween.onDone then
				done = done or {}
				done[#done + 1] = tween.onDone
			end
		end
	end
	if done then
		for _, callback in ipairs(done) do
			callback()
		end
	end
	if next(tweens) == nil then
		self:Hide()
	end
end)

function ns.Tween(object, duration, from, to, set, onDone)
	tweens[object] = { t = 0, duration = math.max(duration, 0.01), from = from, to = to, set = set, onDone = onDone }
	set(object, from)
	driver:Show()
end

function ns.StopTween(object, finish)
	local tween = tweens[object]
	if tween then
		tweens[object] = nil
		if finish then
			tween.set(object, tween.to)
		end
	end
end

function ns.SetAlphaTo(object, value)
	object:SetAlpha(value)
end

-- Minimal object pool.
function ns.NewPool(create, reset)
	local pool = { free = {}, active = {} }
	function pool:Acquire()
		local object = table.remove(self.free)
		if not object then
			object = create()
		end
		self.active[#self.active + 1] = object
		object:Show()
		return object
	end
	function pool:ReleaseAll()
		for i = #self.active, 1, -1 do
			local object = self.active[i]
			self.active[i] = nil
			object:Hide()
			if reset then
				reset(object)
			end
			self.free[#self.free + 1] = object
		end
	end
	return pool
end

-- Flat background and 1px border made from plain textures (no backdrop
-- templates, so it looks the same on every client).
function ns.AddBackground(frame, r, g, b, a)
	local texture = frame:CreateTexture(nil, "BACKGROUND")
	texture:SetAllPoints()
	texture:SetColorTexture(r, g, b, a)
	frame.bg = texture
	return texture
end

function ns.AddBorder(frame, r, g, b, a)
	local edges = {}
	for i = 1, 4 do
		edges[i] = frame:CreateTexture(nil, "BORDER")
	end
	edges[1]:SetPoint("TOPLEFT")
	edges[1]:SetPoint("TOPRIGHT")
	edges[1]:SetHeight(1)
	edges[2]:SetPoint("BOTTOMLEFT")
	edges[2]:SetPoint("BOTTOMRIGHT")
	edges[2]:SetHeight(1)
	edges[3]:SetPoint("TOPLEFT")
	edges[3]:SetPoint("BOTTOMLEFT")
	edges[3]:SetWidth(1)
	edges[4]:SetPoint("TOPRIGHT")
	edges[4]:SetPoint("BOTTOMRIGHT")
	edges[4]:SetWidth(1)
	frame.border = edges
	ns.SetBorderColor(frame, r, g, b, a)
end

function ns.SetBorderColor(frame, r, g, b, a)
	for _, edge in ipairs(frame.border) do
		edge:SetColorTexture(r, g, b, a or 1)
	end
end

-- A small flat "x" button drawn with two lines.
function ns.CreateCloseButton(parent, size)
	local button = CreateFrame("Button", nil, parent)
	button:SetSize(size, size)
	local inset = math.floor(size * 0.28)
	button.lines = {}
	for i = 1, 2 do
		local line = button:CreateLine(nil, "OVERLAY")
		line:SetThickness(1.5)
		button.lines[i] = line
	end
	button.lines[1]:SetStartPoint("TOPLEFT", button, inset, -inset)
	button.lines[1]:SetEndPoint("BOTTOMRIGHT", button, -inset, inset)
	button.lines[2]:SetStartPoint("TOPRIGHT", button, -inset, -inset)
	button.lines[2]:SetEndPoint("BOTTOMLEFT", button, inset, inset)
	local function paint(r, g, b, a)
		for _, line in ipairs(button.lines) do
			line:SetColorTexture(r, g, b, a)
		end
	end
	paint(0.6, 0.6, 0.65, 1)
	button:SetScript("OnEnter", function() paint(1, 1, 1, 1) end)
	button:SetScript("OnLeave", function() paint(0.6, 0.6, 0.65, 1) end)
	return button
end

-- Smooth horizontal S-curve between two points given as offsets from the
-- centre of `anchor`, drawn with pooled Line objects. Optionally collects
-- the segments into `out` (used to fade new branches in).
local SEGMENTS = 14
function ns.DrawCurve(pool, anchor, x1, y1, x2, y2, r, g, b, a, thickness, out)
	local bend = (x2 - x1) * 0.5
	local cx1, cx2 = x1 + bend, x2 - bend
	local px, py = x1, y1
	for i = 1, SEGMENTS do
		local t = i / SEGMENTS
		local u = 1 - t
		local x = u * u * u * x1 + 3 * u * u * t * cx1 + 3 * u * t * t * cx2 + t * t * t * x2
		local y = u * u * u * y1 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y2
		local line = pool:Acquire()
		line:SetThickness(thickness)
		line:SetColorTexture(r, g, b, a)
		line:SetStartPoint("CENTER", anchor, px, py)
		line:SetEndPoint("CENTER", anchor, x, y)
		if out then
			out[#out + 1] = line
		end
		px, py = x, y
	end
end

-- Vertical gradient (bottom colour to top colour) on a texture. Falls back to
-- the average colour on clients without ColorMixin gradients.
function ns.SetVerticalGradient(texture, bottom, top, alpha)
	texture:SetColorTexture(1, 1, 1, 1)
	if texture.SetGradient and CreateColor then
		local ok = pcall(texture.SetGradient, texture, "VERTICAL",
			CreateColor(bottom[1], bottom[2], bottom[3], alpha), CreateColor(top[1], top[2], top[3], alpha))
		if ok then
			return
		end
	end
	texture:SetColorTexture((bottom[1] + top[1]) / 2, (bottom[2] + top[2]) / 2, (bottom[3] + top[3]) / 2, alpha)
end

-- ============================================================================
--  VCP Fly — TOOLS-скрипт для керування дроном через USB-VCP + GVAR
-- ============================================================================
--  Шлях: SD-карта → SCRIPTS/TOOLS/vcpfly.lua
--
--  Читає VCP дані від Python, записує в Global Variables (GV1–GV4),
--  які мікшер використовує як джерело для каналів.
--
--  НАЛАШТУВАННЯ В МІКШЕРІ (MDL → Mixes):
--    CH1 (Throttle): Source = GV1, Weight = 100%
--    CH2 (Roll):     Source = GV2, Weight = 100%
--    CH3 (Pitch):    Source = GV3, Weight = 100%
--    CH4 (Yaw):      Source = GV4, Weight = 100%
--
--  ВАЖЛИВО: GV діапазон в EdgeTX = -1024..1024
--  Протокол з Python вже у цьому діапазоні.
--
--  EXIT = вихід + скидання GV в 0 (безпечно)
-- ============================================================================

local HEADER_1 = 0x55
local HEADER_2 = 0xAA
local FRAME_SIZE = 11
local WIRE_OFFSET = 1024
local FAILSAFE_MS = 200

-- Телеметрія → Python
local TELEM_H1 = 0xAA
local TELEM_H2 = 0x55
local TELEM_OFFSET = 1800   -- зсув: градуси*10 + 1800 → uint16
local telemCounter = 0
local TELEM_EVERY = 1       -- відправка кожен цикл (мінімальна затримка)
local telemPitch = 0
local telemRoll = 0

-- Буфер
local buf = {}
local bufLen = 0

-- Канали
local vThr = 0
local vRol = 0
local vPit = 0
local vYaw = 0

-- Стан
local framesOK = 0
local lastFrameTime = 0
local armed = false  -- true = дані передаються в мікшер

local function init()
  buf = {}
  bufLen = 0
  framesOK = 0
  lastFrameTime = 0
  armed = false
  vThr = 0
  vRol = 0
  vPit = 0
  vYaw = 0
  -- Скинути GV (AETR: GV1=Rol, GV2=Pit, GV3=Thr, GV4=Yaw)
  model.setGlobalVariable(0, 0, 0)     -- GV1 = Roll = центр
  model.setGlobalVariable(1, 0, 0)     -- GV2 = Pitch = центр
  model.setGlobalVariable(2, 0, -1024) -- GV3 = Throttle = МІНІМУМ
  model.setGlobalVariable(3, 0, 0)     -- GV4 = Yaw = центр
end

local function setChannels(thr, rol, pit, yaw)
  -- Clamp до -1024..1024
  if thr < -1024 then thr = -1024 elseif thr > 1024 then thr = 1024 end
  if rol < -1024 then rol = -1024 elseif rol > 1024 then rol = 1024 end
  if pit < -1024 then pit = -1024 elseif pit > 1024 then pit = 1024 end
  if yaw < -1024 then yaw = -1024 elseif yaw > 1024 then yaw = 1024 end

  vThr = thr
  vRol = rol
  vPit = pit
  vYaw = yaw

  -- Записуємо в Global Variables (AETR: GV1=Rol, GV2=Pit, GV3=Thr, GV4=Yaw)
  model.setGlobalVariable(0, 0, rol)  -- GV1 = Roll
  model.setGlobalVariable(1, 0, pit)  -- GV2 = Pitch
  model.setGlobalVariable(2, 0, thr)  -- GV3 = Throttle
  model.setGlobalVariable(3, 0, yaw)  -- GV4 = Yaw
end

local function sendTelemetry()
  -- Читаємо телеметрію від дрона (CRSF/ELRS)
  -- Назви сенсорів: "Ptch" і "Roll" (EdgeTX стандарт для CRSF Attitude)
  -- Якщо у вас інші назви — перевірте MODEL → Telemetry → Sensors
  local pitchDeg = getValue("Ptch") or 0
  local rollDeg = getValue("Roll") or 0

  telemPitch = pitchDeg
  telemRoll = rollDeg

  -- Кодуємо: градуси * 10 + offset → uint16 (0..3600)
  local pv = math.floor(pitchDeg * 10 + 0.5) + TELEM_OFFSET
  local rv = math.floor(rollDeg * 10 + 0.5) + TELEM_OFFSET
  if pv < 0 then pv = 0 elseif pv > 3600 then pv = 3600 end
  if rv < 0 then rv = 0 elseif rv > 3600 then rv = 3600 end

  local ph = math.floor(pv / 256)
  local pl = pv % 256
  local rh = math.floor(rv / 256)
  local rl = rv % 256

  local xor = bit32.bxor(ph, pl, rh, rl)
  serialWrite(string.char(TELEM_H1, TELEM_H2, ph, pl, rh, rl, xor))
end

local function run(event)
  -- EXIT = безпечний вихід (throttle мінімум)
  if event == EVT_EXIT_BREAK then
    setChannels(-1024, 0, 0, 0)
    return 1
  end

  -- ENTER = toggle armed/disarmed
  if event == EVT_ENTER_BREAK then
    armed = not armed
    if not armed then
      setChannels(-1024, 0, 0, 0)
    end
  end

  -- Читаємо VCP
  local data = serialRead()

  if data and #data > 0 then
    for i = 1, #data do
      bufLen = bufLen + 1
      buf[bufLen] = string.byte(data, i)
    end

    while bufLen >= FRAME_SIZE do
      if buf[1] == HEADER_1 and buf[2] == HEADER_2 then
        local xor = 0
        for i = 3, 10 do
          xor = bit32.bxor(xor, buf[i])
        end

        if xor == buf[11] then
          local thr = (buf[3] * 256 + buf[4]) - WIRE_OFFSET
          local rol = (buf[5] * 256 + buf[6]) - WIRE_OFFSET
          local pit = (buf[7] * 256 + buf[8]) - WIRE_OFFSET
          local yaw = (buf[9] * 256 + buf[10]) - WIRE_OFFSET

          framesOK = framesOK + 1
          lastFrameTime = getTime()

          if armed then
            setChannels(thr, rol, pit, yaw)
          end
        end

        local newBuf = {}
        for i = FRAME_SIZE + 1, bufLen do
          newBuf[i - FRAME_SIZE] = buf[i]
        end
        buf = newBuf
        bufLen = bufLen - FRAME_SIZE
      else
        local newBuf = {}
        for i = 2, bufLen do
          newBuf[i - 1] = buf[i]
        end
        buf = newBuf
        bufLen = bufLen - 1
      end
    end
  end

  -- Failsafe: немає даних >200ms → нейтраль
  local now = getTime()
  if armed and lastFrameTime > 0 and (now - lastFrameTime) > 20 then
    -- getTime() = 10ms тіки, 20 тіків = 200ms
    setChannels(-1024, 0, 0, 0)
  end

  -- Телеметрія → Python (кожні TELEM_EVERY циклів)
  telemCounter = telemCounter + 1
  if telemCounter >= TELEM_EVERY then
    telemCounter = 0
    sendTelemetry()
  end

  -- Екран 128x64
  lcd.clear()

  if armed then
    lcd.drawText(1, 0, "** VCP FLY [ARMED] **", BOLD + BLINK)
  else
    lcd.drawText(1, 0, "VCP FLY [SAFE]", BOLD)
  end

  lcd.drawText(1, 11, "T:" .. tostring(vThr), 0)
  lcd.drawText(65, 11, "R:" .. tostring(vRol), 0)
  lcd.drawText(1, 21, "P:" .. tostring(vPit), 0)
  lcd.drawText(65, 21, "Y:" .. tostring(vYaw), 0)

  lcd.drawText(1, 33, "Frames:" .. tostring(framesOK), SMLSIZE)
  lcd.drawText(1, 42, "P:" .. string.format("%.1f", telemPitch)
                    .. " R:" .. string.format("%.1f", telemRoll), SMLSIZE)

  if armed then
    lcd.drawText(1, 51, "ENTER=disarm  EXIT=quit", SMLSIZE)
  else
    lcd.drawText(1, 51, "ENTER=ARM     EXIT=quit", SMLSIZE)
  end

  lcd.drawText(1, 59, "GV1-4 -> CH1-4 in Mixes", SMLSIZE + INVERS)

  return 0
end

return { init=init, run=run }

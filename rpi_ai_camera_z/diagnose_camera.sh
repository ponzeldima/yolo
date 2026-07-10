#!/bin/bash
# Діагностичний скрипт для перевірки стану камери на Raspberry Pi

echo "╔════════════════════════════════════════════╗"
echo "║        Camera Diagnostic Script            ║"
echo "╚════════════════════════════════════════════╝"
echo

# 1. Перевір V4L2 пристрої
echo "1️⃣ V4L2 пристрої (OpenCV):"
if command -v v4l2-ctl &> /dev/null; then
    echo "   v4l2-ctl знайдений:"
    v4l2-ctl --list-devices || echo "   (помилка при виконанні)"
else
    echo "   ⚠️ v4l2-ctl не встановлений"
    echo "   Встанови: sudo apt-get install v4l-utils"
fi
echo

# 2. Перевір /dev/video*
echo "2️⃣ /dev/video* пристрої:"
ls -la /dev/video* 2>/dev/null || echo "   (немає пристроїв)"
echo

# 3. Перевір libcamera
echo "3️⃣ libcamera:"
if command -v libcamera-hello &> /dev/null; then
    echo "   ✓ libcamera встановлена"
    echo "   Спробуй: libcamera-hello --list-cameras"
else
    echo "   ⚠️ libcamera не встановлена"
    echo "   Встанови: sudo apt-get install -y libcamera-tools"
fi
echo

# 4. Перевір picamera2
echo "4️⃣ Python picamera2:"
python3 << 'EOF'
try:
    from picamera2 import Picamera2
    print("   ✓ picamera2 встановлена")
    try:
        cameras = Picamera2.global_camera_info()
        for info in cameras:
            print(f"   Камера: {info['Model']}")
    except Exception as e:
        print(f"   ⚠️ Помилка при отриманні інформації: {e}")
except ImportError:
    print("   ❌ picamera2 не встановлена")
    print("   Встанови: sudo apt-get install -y python3-picamera2 python3-libcamera")
EOF
echo

# 5. Перевір дозволів
echo "5️⃣ Дозволи доступу:"
echo "   Користувач: $(whoami)"
if [ -e /dev/video0 ]; then
    ls -l /dev/video0
fi
if groups | grep -q "video"; then
    echo "   ✓ Користувач у групі 'video'"
else
    echo "   ⚠️ Користувач НЕ у групі 'video'"
    echo "   Додай: sudo usermod -a -G video $(whoami)"
fi
echo

echo "📝 Рекомендації:"
echo "   - Якщо камера не визначена, переконайся що вона підключена до cam/disp1"
echo "   - Якщо помилка дозволів, додай користувача до групи video:"
echo "     sudo usermod -a -G video \$USER"
echo "   - Потім перезавантажся або заново логініся"
echo "   - Якщо все з V4L2, спробуй libcamera версію:"
echo "     python3 ai_camera_stream_libcamera.py"
echo

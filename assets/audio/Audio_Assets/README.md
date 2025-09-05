# Audio Assets

This folder contains audio files used by the ARCIS weapon detection system for threat alerts and notifications.

## 📁 Contents

### Alert Sounds
- `danger_alert.mp3` - **Primary danger alert sound (37KB)**
  - High-priority threat notification
  - Used for CRITICAL and HIGH threat levels
  - Designed for military field operations
  - Clear, attention-grabbing audio signal

## 🎯 Audio Specifications

### danger_alert.mp3
- **Format**: MP3
- **Size**: 37KB
- **Duration**: ~2-3 seconds
- **Quality**: High clarity for field conditions
- **Volume**: Optimized for tactical environments
- **Purpose**: Immediate threat notification

## 🚀 Usage

### In Original ARCIS System
```python
# Automatically triggered for high-threat detections
# Configurable in train_weapon_detection.py
enable_sound = True  # Enable/disable audio alerts
```

### In Redis System
```python
# Used across distributed system
# Raspberry Pi clients can play local alerts
# Configurable per deployment
```

### Manual Testing
```python
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load and play alert
sound = pygame.mixer.Sound('Audio_Assets/danger_alert.mp3')
sound.play()
```

## 🔧 Alert Trigger Conditions

### Automatic Triggers
- **CRITICAL Threats**: Tanks, aircraft, heavy weapons
- **HIGH Threats**: Guns, rifles, confirmed weapons
- **Confidence > 0.8**: High-confidence detections
- **Distance < 50m**: Close-range threats

### Configurable Settings
```python
# In ARCIS detection scripts
ALERT_COOLDOWN = 3  # seconds between alerts
MIN_CONFIDENCE = 0.8  # minimum confidence for alert
DANGEROUS_CLASSES = ['weapon', 'gun', 'rifle', 'military_tank']
```

## 📊 Audio Integration

### System Integration Points
1. **Main Detection Loop**: Real-time threat alerts
2. **Background Threading**: Non-blocking audio playback
3. **Cooldown Management**: Prevents audio spam
4. **Fallback System**: System beep if audio fails

### Multi-device Support
- **Jetson Nano**: Local speaker output
- **Raspberry Pi**: Field alert speakers
- **Desktop/Server**: Standard audio output
- **Headless Systems**: Optional audio disable

## 🔄 Customization

### Adding New Alert Sounds
```bash
# Add new audio file
cp new_alert.mp3 Audio_Assets/

# Update script references
# Modify audio file paths in detection scripts
```

### Audio Format Support
- **Primary**: MP3 (universal compatibility)
- **Alternative**: WAV (uncompressed, larger files)
- **Fallback**: System beep (no file required)

### Volume and Quality Settings
```python
# Pygame mixer settings
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Volume control (0.0 to 1.0)
sound.set_volume(0.8)
```

## ⚙️ Technical Requirements

### Software Dependencies
- **pygame**: Audio playback library
- **Python 3.8+**: Core runtime
- **Audio drivers**: System audio support

### Hardware Requirements
- **Audio output**: Speakers, headphones, or audio jack
- **Audio codec**: MP3 decoder support
- **Memory**: Minimal (37KB for danger_alert.mp3)

## 🛠️ Troubleshooting

### Common Issues

1. **No Audio Output**
   ```bash
   # Check audio system
   python -c "import pygame; pygame.mixer.init(); print('Audio OK')"
   
   # Test system audio
   # Linux: aplay test.wav
   # Windows: Check volume mixer
   ```

2. **Audio File Not Found**
   ```bash
   # Verify file exists
   ls -la Audio_Assets/danger_alert.mp3
   
   # Check file permissions
   chmod 644 Audio_Assets/danger_alert.mp3
   ```

3. **Pygame Import Error**
   ```bash
   # Install pygame
   pip install pygame
   
   # Verify installation
   python -c "import pygame; print(pygame.version.ver)"
   ```

### Performance Issues
```python
# Optimize audio loading
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.init()

# Preload sounds for faster playback
danger_sound = pygame.mixer.Sound('Audio_Assets/danger_alert.mp3')
```

## 🔊 Audio Best Practices

### Field Deployment
1. **Volume Testing**: Test audio levels in deployment environment
2. **Background Noise**: Consider ambient noise levels
3. **Speaker Placement**: Position for optimal coverage
4. **Power Management**: Consider battery usage for portable systems

### System Integration
1. **Threading**: Use background threads for audio playback
2. **Error Handling**: Graceful fallback to system beep
3. **Cooldown Timers**: Prevent audio alert spam
4. **User Control**: Allow audio enable/disable

## 📈 Future Enhancements

### Planned Features
- **Multiple Alert Types**: Different sounds for different threat levels
- **Voice Alerts**: Spoken threat descriptions
- **Directional Audio**: Spatial audio for threat bearing
- **Volume Auto-adjustment**: Adaptive volume based on environment

### Custom Alert Creation
```bash
# Guidelines for new alert sounds:
# - Duration: 1-3 seconds
# - Format: MP3 or WAV
# - Quality: Clear and attention-grabbing
# - Size: Keep under 100KB for efficiency
```

## 📚 Related Documentation

- **Original ARCIS**: `../Original_ARCIS_System/README.md` - Audio integration in main system
- **Redis System**: `../ARCIS_Redis_System/README.md` - Distributed audio alerts
- **Utilities**: `../Utilities/README.md` - Audio testing tools

---

**Note**: Audio alerts are a critical safety feature of the ARCIS system. Ensure proper testing and configuration for your deployment environment. 
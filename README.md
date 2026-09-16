# 🚗 Car Parking Detection using Computer Vision

A **Computer Vision** project built with **Python** that automatically detects
parking slot occupancy in real time. The system identifies which parking spaces
are **free** and which are **occupied**, counts the total number of available
slots, and displays the result visually on the video feed.

## ✨ Features
- 🎥 Real-time parking slot detection from video / webcam / image
- 🟢 Highlights **free** slots in green
- 🔴 Highlights **occupied** slots in red
- 🔢 Displays live **count of free slots**
- 🧠 Built using Computer Vision techniques (OpenCV)
- ⚙️ Customizable parking slot regions (ROI) via JSON or interactive selector

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** OpenCV, NumPy
- **Domain:** Computer Vision / Image Processing

## 🧠 How It Works
1. Load the video feed or image of the parking area.
2. Define parking slot regions (ROI) in `slots.json` or create them interactively.
3. Preprocess each slot (grayscale → blur → threshold).
4. Analyze white-pixel density in each slot to determine occupancy.
5. Draw bounding boxes: **green = free**, **red = occupied**.
6. Display the total **free slot count** on the screen.

## 📦 Installation
```bash
git clone https://github.com/yourusername/car-parking-detection.git
cd car-parking-detection
pip install -r requirements.txt
```

## ▶️ Usage
Run detection on webcam:
```bash
python parking_detection.py --source 0 --slots slots.json
```

Run detection on video/image:
```bash
python parking_detection.py --source /path/to/parking.mp4 --slots slots.json
python parking_detection.py --source /path/to/parking.jpg --slots slots.json
```

Create slot ROIs interactively from first frame:
```bash
python parking_detection.py --source /path/to/parking.mp4 --slots slots.json --select-slots
```

### Controls
- `q`: quit
- `s`: save slots while selecting

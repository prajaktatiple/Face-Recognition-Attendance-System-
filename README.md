# Face Recognition Attendance System — Task 3

A Python desktop application that registers face samples from a webcam, trains an LBPH face-recognition model, and records attendance to a CSV file.

## Features
- Register a person and capture face samples from a webcam
- Train an OpenCV LBPH recognition model
- Recognize registered people and mark them Present
- Prevent duplicate attendance entries for the same person on the same date
- Save attendance in `attendance/attendance.csv`
- Dark-mode Tkinter dashboard

## Requirements
- Python 3.9 or newer
- A working webcam
- Windows, macOS, or Linux with GUI support

## Setup (VS Code)
1. Extract this folder.
2. Open the `Face_Recognition_Attendance_System` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment (recommended):
   - Windows: `py -m venv .venv`
   - Activate: `.venv\Scripts\activate`
5. Install packages: `python -m pip install -r requirements.txt`
6. Run: `python app.py`

If `python` does not work on Windows, try `py app.py`.

## How to use
1. Click **Register Face**, type a person's name, and look toward the camera. Keep good lighting and move your head slightly.
2. Register each person separately.
3. Click **Train Recognition Model**.
4. Click **Start Attendance Camera**. A recognized person is marked Present.
5. Press `Q` in the camera window to stop.
6. Open `attendance/attendance.csv` to see the records.

## Important
- Register and train the model again whenever you add new people.
- Face recognition can be affected by lighting, camera quality, pose, and similar-looking faces. Verify attendance records when accuracy matters.
- Get consent before capturing or storing anyone's face data. Store the dataset securely and delete it when no longer needed.

"""
Task 3: Face Recognition Attendance System
A desktop attendance app using OpenCV LBPH face recognition and Tkinter.

Install:
    pip install -r requirements.txt

Run:
    python app.py

Notes:
- Use good lighting and keep one face in the camera frame during registration.
- Register each person before training the recognizer.
- Attendance is saved to attendance/attendance.csv.
"""
import csv
import os
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
MODEL_PATH = os.path.join(TRAINER_DIR, "face_model.yml")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.csv")
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")
FACE_SIZE = (200, 200)
SAMPLES_TO_CAPTURE = 30

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def get_recognizer():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is missing. Install opencv-contrib-python "
            "using: pip install opencv-contrib-python"
        )
    return cv2.face.LBPHFaceRecognizer_create()


def capture_person(name):
    safe_name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in (" ", "_", "-")).strip()
    if not safe_name:
        raise ValueError("Please enter a valid name.")
    person_dir = os.path.join(DATASET_DIR, safe_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permission or connection.")

    count = 0
    last_capture = 0
    try:
        while count < SAMPLES_TO_CAPTURE:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (70, 220, 130), 2)
                cv2.putText(frame, f"Samples: {count}/{SAMPLES_TO_CAPTURE}", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 220, 130), 2)
                now = time.time()
                if now - last_capture > 0.18:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, FACE_SIZE)
                    count += 1
                    cv2.imwrite(os.path.join(person_dir, f"{count:03d}.jpg"), face)
                    last_capture = now
                break
            cv2.imshow("Register Face - press Q to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return count, safe_name


def train_model():
    faces, labels = [], []
    label_to_name = {}
    person_dirs = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    label_id = 0
    for person in person_dirs:
        person_path = os.path.join(DATASET_DIR, person)
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not image_files:
            continue
        label_to_name[label_id] = person
        for file_name in image_files:
            img = cv2.imread(os.path.join(person_path, file_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(cv2.resize(img, FACE_SIZE))
                labels.append(label_id)
        label_id += 1

    if not faces:
        raise RuntimeError("No registered face images found. Click Register Face first.")

    recognizer = get_recognizer()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "name"])
        for label_id, name in label_to_name.items():
            writer.writerow([label_id, name])
    return len(faces), len(label_to_name)


def load_labels():
    labels = {}
    if not os.path.exists(LABELS_PATH):
        return labels
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[int(row["label_id"])] = row["name"]
    return labels


def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Name", "Date", "Time", "Status"])


def mark_attendance(name):
    ensure_attendance_file()
    today = datetime.now().strftime("%Y-%m-%d")
    with open(ATTENDANCE_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if any(row["Name"] == name and row["Date"] == today for row in rows):
        return False
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([name, today, datetime.now().strftime("%H:%M:%S"), "Present"])
    return True


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("900x620")
        self.root.configure(bg="#101827")
        self.root.minsize(820, 580)

        header = tk.Frame(root, bg="#172338", padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="FACE RECOGNITION", font=("Segoe UI", 20, "bold"),
                 fg="#f4f7fb", bg="#172338").pack(anchor="w")
        tk.Label(header, text="ATTENDANCE SYSTEM  •  TASK 3", font=("Segoe UI", 10, "bold"),
                 fg="#45d6b5", bg="#172338").pack(anchor="w", pady=(4, 0))

        body = tk.Frame(root, bg="#101827", padx=24, pady=22)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg="#172338", padx=20, pady=20)
        left.pack(side="left", fill="both", expand=True, padx=(0, 16))
        right = tk.Frame(body, bg="#172338", padx=20, pady=20, width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(left, text="Attendance Dashboard", font=("Segoe UI", 16, "bold"),
                 fg="#f4f7fb", bg="#172338").pack(anchor="w")
        tk.Label(left, text="Register faces, train the model, and mark attendance.",
                 font=("Segoe UI", 10), fg="#aab8cc", bg="#172338").pack(anchor="w", pady=(8, 18))

        self.status = tk.StringVar(value="System ready. Register a face to get started.")
        status_box = tk.Frame(left, bg="#202f46", padx=14, pady=14)
        status_box.pack(fill="x", pady=(0, 18))
        tk.Label(status_box, text="SYSTEM STATUS", font=("Segoe UI", 9, "bold"),
                 fg="#45d6b5", bg="#202f46").pack(anchor="w")
        tk.Label(status_box, textvariable=self.status, wraplength=440, justify="left",
                 font=("Segoe UI", 10), fg="#f4f7fb", bg="#202f46").pack(anchor="w", pady=(6, 0))

        self.make_button(left, "1   Register Face", self.register, "#176b87").pack(fill="x", pady=6)
        self.make_button(left, "2   Train Recognition Model", self.train, "#245d9c").pack(fill="x", pady=6)
        self.make_button(left, "3   Start Attendance Camera", self.attendance, "#16795d").pack(fill="x", pady=6)
        self.make_button(left, "Open Attendance CSV", self.open_attendance, "#39485f").pack(fill="x", pady=6)

        tk.Label(right, text="QUICK GUIDE", font=("Segoe UI", 12, "bold"),
                 fg="#f4f7fb", bg="#172338").pack(anchor="w", pady=(0, 12))
        for n, text in [
            ("01", "Register each person using the webcam."),
            ("02", "Train the model after registering faces."),
            ("03", "Start the camera to recognize people."),
            ("04", "Attendance is saved once per person each day.")
        ]:
            item = tk.Frame(right, bg="#202f46", padx=10, pady=10)
            item.pack(fill="x", pady=5)
            tk.Label(item, text=n, font=("Segoe UI", 10, "bold"), fg="#45d6b5",
                     bg="#202f46").pack(anchor="w")
            tk.Label(item, text=text, font=("Segoe UI", 9), fg="#d8e1ed",
                     bg="#202f46", wraplength=205, justify="left").pack(anchor="w", pady=(4, 0))
        tk.Label(root, text="Built with Python • OpenCV • Tkinter • CSV",
                 font=("Segoe UI", 9), fg="#71839b", bg="#101827").pack(pady=(0, 12))

        ensure_attendance_file()

    @staticmethod
    def make_button(parent, text, command, color):
        return tk.Button(parent, text=text, command=command, font=("Segoe UI", 11, "bold"),
                         fg="white", bg=color, activebackground=color, activeforeground="white",
                         relief="flat", bd=0, padx=12, pady=12, cursor="hand2")

    def register(self):
        name = simpledialog.askstring("Register Face", "Enter the person's full name:", parent=self.root)
        if not name:
            return
        try:
            count, safe_name = capture_person(name)
            self.status.set(f"Registered {safe_name}: captured {count} face samples.")
            messagebox.showinfo("Registration Complete", f"{safe_name}: {count} images captured.")
        except Exception as exc:
            messagebox.showerror("Registration Error", str(exc))
            self.status.set("Registration failed. Check the webcam and try again.")

    def train(self):
        try:
            count, people = train_model()
            self.status.set(f"Model trained successfully with {count} images from {people} person(s).")
            messagebox.showinfo("Training Complete", f"Trained with {count} images from {people} person(s).")
        except Exception as exc:
            messagebox.showerror("Training Error", str(exc))
            self.status.set("Training failed. Register at least one face first.")

    def attendance(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            messagebox.showwarning("Model Required", "Please register faces and train the model first.")
            return
        try:
            recognizer = get_recognizer()
            recognizer.read(MODEL_PATH)
            labels = load_labels()
        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam. Check camera permission or connection.")
            return

        marked = set()
        self.status.set("Attendance camera running. Press Q in the camera window to stop.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
                for (x, y, w, h) in faces:
                    crop = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
                    label_id, confidence = recognizer.predict(crop)
                    name = labels.get(label_id, "Unknown")
                    # LBPH confidence is a distance; lower values indicate a closer match.
                    if confidence < 75 and name != "Unknown":
                        newly_marked = mark_attendance(name)
                        if newly_marked:
                            marked.add(name)
                        display = f"{name} - Present"
                        color = (70, 220, 130)
                    else:
                        display = "Unknown"
                        color = (70, 100, 240)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, display, (x, max(25, y-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, color, 2)
                cv2.putText(frame, "Press Q to stop", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2)
                cv2.imshow("Attendance Camera - press Q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        names = ", ".join(sorted(marked)) if marked else "No new attendance marked"
        self.status.set(f"Camera stopped. {names}.")
        messagebox.showinfo("Attendance Session", f"Session finished.\n{names}\n\nFile: attendance/attendance.csv")

    def open_attendance(self):
        ensure_attendance_file()
        try:
            os.startfile(ATTENDANCE_FILE)
        except AttributeError:
            messagebox.showinfo("Attendance File", f"Attendance file location:\n{ATTENDANCE_FILE}")
        except Exception as exc:
            messagebox.showerror("Open File Error", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()

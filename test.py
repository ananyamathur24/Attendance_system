import customtkinter as ctk
from tkinter import messagebox
import os
import cv2
import numpy as np
from PIL import Image
import csv
from datetime import datetime
from tkinter import filedialog
import pandas as pd
#from fpdf import FPDF



ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

USERNAME = "admin"
PASSWORD = "admin123"

class LoginApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Attendance System - Login")
        self.geometry("400x300")
        self.resizable(False, False)

        self.title_label = ctk.CTkLabel(self, text="Login", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=20)

        self.username_entry = ctk.CTkEntry(self, placeholder_text="Username")
        self.username_entry.pack(pady=10)

        self.password_entry = ctk.CTkEntry(self, placeholder_text="Password", show="*")
        self.password_entry.pack(pady=10)

        self.login_button = ctk.CTkButton(self, text="Login", command=self.login)
        self.login_button.pack(pady=20)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if username == USERNAME and password == PASSWORD:
            messagebox.showinfo("Login Successful", "Welcome!")
            self.destroy()
            DashboardApp()
        else:
            messagebox.showerror("Login Failed", "Invalid Credentials")

class DashboardApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Attendance System - Dashboard")
        self.geometry("600x400")
        self.resizable(False, False)

        self.title_label = ctk.CTkLabel(self, text="Attendance System", font=ctk.CTkFont(size=28, weight="bold"))
        self.title_label.pack(pady=20)

        self.register_button = ctk.CTkButton(self, text="Register Student", width=200, command=self.open_register_window)
        self.register_button.pack(pady=10)

        self.attendance_button = ctk.CTkButton(self, text="Take Attendance", width=200, command=self.open_attendance_window)
        self.attendance_button.pack(pady=10)

        self.report_button = ctk.CTkButton(self, text="View Reports", width=200, command=self.view_reports)
        self.report_button.pack(pady=10)

        self.exit_button = ctk.CTkButton(self, text="Exit", width=200, command=self.destroy)
        self.exit_button.pack(pady=10)

        self.mainloop()


    def open_register_window(self):
        RegisterStudentWindow()

    def open_attendance_window(self):
        TakeAttendanceWindow()

    def view_reports(self):
        ReportWindow()

class TakeAttendanceWindow(ctk.CTkToplevel):
    def __init__(self):
        super().__init__()
        
        self.title("Select Class for Attendance")
        self.geometry("300x200")
        self.resizable(False, False)

        self.label = ctk.CTkLabel(self, text="Select Class", font=ctk.CTkFont(size=20, weight="bold"))
        self.label.pack(pady=20)

        self.class_var = ctk.StringVar()
        self.class_dropdown = ctk.CTkOptionMenu(self, values=self.get_class_list(), variable=self.class_var)
        self.class_dropdown.pack(pady=10)

        self.start_button = ctk.CTkButton(self, text="Start Attendance", command=self.take_attendance)
        self.start_button.pack(pady=20)

    def get_class_list(self):
        data_dir = "data"
        if not os.path.exists(data_dir):
            return []
        return [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    def map_enrollment_to_name(self):
        mapping = {}
        data_dir = "data"
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            for student_folder in os.listdir(class_path):
                if '_' in student_folder:
                    enrollment_no, name = student_folder.split('_', 1)
                    mapping[int(enrollment_no)] = name
        return mapping

    def take_attendance(self):
        class_name = self.class_var.get()
        if not class_name:
            messagebox.showerror("Error", "Please select a class")
            return

        self.destroy()  # Close class selection window

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer/trainer.yml")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)

        recognized_ids = {}
        student_names = self.map_enrollment_to_name()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                confidence_percent = round(100 - confidence)

                if confidence < 70:
                    if id_ not in recognized_ids:
                        recognized_ids[id_] = {
                            'id': id_,
                            'name': student_names.get(id_, "Unknown"),
                            'confidence': confidence_percent
                        }
                    text = f"ID: {id_} ({confidence_percent}%)"
                    color = (0, 255, 0)
                else:
                    text = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Attendance - Press Q to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if recognized_ids:
            save_attendance(class_name, recognized_ids.values())
            messagebox.showinfo("Attendance", f"Attendance saved for: {', '.join(str(i) for i in recognized_ids.keys())}")
        else:
            messagebox.showwarning("Attendance", "No known faces were recognized.")

class RegisterStudentWindow(ctk.CTkToplevel):
    def __init__(self):
        super().__init__()
        self.title("Register Student")
        self.geometry("400x400")
        self.resizable(False, False)

        self.title_label = ctk.CTkLabel(self, text="Register Student", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=20)

        self.class_entry = ctk.CTkEntry(self, placeholder_text="Class Name")
        self.class_entry.pack(pady=10)

        self.enrollment_entry = ctk.CTkEntry(self, placeholder_text="Enrollment No")
        self.enrollment_entry.pack(pady=10)

        self.name_entry = ctk.CTkEntry(self, placeholder_text="Student Name")
        self.name_entry.pack(pady=10)

        self.capture_button = ctk.CTkButton(self, text="Start Capture", command=self.capture_faces)
        self.capture_button.pack(pady=20)

    def capture_faces(self):
        class_name = self.class_entry.get()
        enrollment = self.enrollment_entry.get()
        name = self.name_entry.get()

        if not class_name or not enrollment or not name:
            messagebox.showerror("Error", "Please fill all fields")
            return

        save_path = os.path.join('data', class_name, f"{enrollment}_{name}")
        os.makedirs(save_path, exist_ok=True)

        self.withdraw()

        cap = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        img_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                img_count += 1
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                img_name = os.path.join(save_path, f"{img_count}.jpg")
                cv2.imwrite(img_name, face_img)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            cv2.imshow("Capturing Faces (Press 'q' to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or img_count >= 20:
                break

        cap.release()
        cv2.destroyAllWindows()

        if img_count >= 20:
            messagebox.showinfo("Success", f"Captured {img_count} images successfully!")
        else:
            messagebox.showwarning("Incomplete", "Not enough images captured. Try again.")

        train_model()
        self.destroy()

def save_attendance(class_name, attendance_data):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    folder_path = os.path.join("attendance", class_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{date_str}.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Enrollment No", "Name", "Confidence (%)", "Time"])
        for entry in attendance_data:
            writer.writerow([entry['id'], entry['name'], entry['confidence'], time_str])

def train_model(data_dir='data', model_save_path='trainer/trainer.yml'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = []
    ids = []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for student_folder in os.listdir(class_path):
            student_path = os.path.join(class_path, student_folder)
            enrollment_id = int(student_folder.split('_')[0])
            for img_file in os.listdir(student_path):
                img_path = os.path.join(student_path, img_file)
                pil_img = Image.open(img_path).convert('L')
                img_np = np.array(pil_img, 'uint8')
                detected_faces = detector.detectMultiScale(img_np)
                for (x, y, w, h) in detected_faces:
                    faces.append(img_np[y:y+h, x:x+w])
                    ids.append(enrollment_id)

    if not faces:
        print("No faces found for training.")
        return

    recognizer.train(faces, np.array(ids))
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    recognizer.save(model_save_path)
    print(f"Model trained and saved to {model_save_path}")

class ReportWindow(ctk.CTkToplevel):
    instance = None

    def get_class_list(self):
        data_dir = "attendance"
        if not os.path.exists(data_dir):
            return []
        return [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    def __init__(self):
        if ReportWindow.instance is not None:
            ReportWindow.instance.lift()
            return
        super().__init__()
        ReportWindow.instance = self
        self.title("Reports")
        self.geometry("650x600")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.mode_var = ctk.StringVar(value="Student")
        self.mode_selector = ctk.CTkSegmentedButton(self, values=["Student", "Class"], variable=self.mode_var, command=self.toggle_input)
        self.mode_selector.pack(pady=10)

        self.enroll_entry = ctk.CTkEntry(self, placeholder_text="Enter Enrollment Number")
        self.enroll_entry.pack(pady=10)
        self.class_var = ctk.StringVar()
        self.class_dropdown = ctk.CTkOptionMenu(self, values=self.get_class_list(), variable=self.class_var)
        self.class_dropdown.pack(pady=10)

        self.search_button = ctk.CTkButton(self, text="Search", command=self.search_reports)
        self.search_button.pack(pady=10)

        self.result_text = ctk.CTkTextbox(self, width=600, height=350)
        self.result_text.pack(pady=10)

        self.download_csv = ctk.CTkButton(self, text="Download CSV", command=self.download_as_csv)
        self.download_csv.pack(pady=5)

        self.download_pdf = ctk.CTkButton(self, text="Download PDF", command=self.download_as_pdf)
        self.download_pdf.pack(pady=5)

        self.attendance_records = []

    def on_close(self):
        ReportWindow.instance = None
        self.destroy()

    def toggle_input(self, choice=None):
        if self.mode_var.get() == "Student":
            self.enroll_entry.configure(state="normal")
            self.class_dropdown.configure(state="disabled")
        else:
            self.enroll_entry.configure(state="disabled")
            self.class_dropdown.configure(state="normal")

    def search_reports(self):
        self.result_text.delete("1.0", "end")
        self.attendance_records = []

        mode = self.mode_var.get()

        total_classes = 0
        attended_classes = 0
        class_summary = {}

        if mode == "Student":
            enroll_no = self.enroll_entry.get().strip()
            if not enroll_no.isdigit():
                messagebox.showerror("Error", "Please enter a valid enrollment number.")
                return
            enroll_no = int(enroll_no)
        else:
            class_name_filter = self.class_var.get().strip().lower()

        for class_folder in os.listdir("attendance"):
            class_path = os.path.join("attendance", class_folder)
            for file in os.listdir(class_path):
                if file.endswith(".csv"):
                    total_classes += 1
                    date = file.replace(".csv", "")
                    df = pd.read_csv(os.path.join(class_path, file))

                    if mode == "Student":
                        student_rows = df[df["Enrollment No"] == enroll_no]
                        if not student_rows.empty:
                            attended_classes += 1
                            for _, row in student_rows.iterrows():
                                self.attendance_records.append({
                                    "Class": class_folder,
                                    "Date": date,
                                    "Name": row["Name"],
                                    "Confidence": row["Confidence (%)"],
                                    "Time": row["Time"]
                                })
                    elif mode == "Class" and class_name_filter in class_folder.lower():
                        for _, row in df.iterrows():
                            entry = {
                                "Class": class_folder,
                                "Date": date,
                                "Enrollment": row["Enrollment No"],
                                "Name": row["Name"],
                                "Confidence": row["Confidence (%)"],
                                "Time": row["Time"]
                            }
                            self.attendance_records.append(entry)
                            if class_folder not in class_summary:
                                class_summary[class_folder] = 0
                            class_summary[class_folder] += 1
                        for _, row in df.iterrows():
                            entry = {
                                "Class": class_folder,
                                "Date": date,
                                "Enrollment": row["Enrollment No"],
                                "Name": row["Name"],
                                "Confidence": row["Confidence (%)"],
                                "Time": row["Time"]
                            }
                            self.attendance_records.append(entry)
                            if class_folder not in class_summary:
                                class_summary[class_folder] = 0
                            class_summary[class_folder] += 1

        if not self.attendance_records:
            self.result_text.insert("1.0", "No records found.\n")
        else:
            if mode == "Student":
                for record in self.attendance_records:
                    self.result_text.insert("end", f"{record['Date']} - {record['Class']} - {record['Name']} ({record['Confidence']}%) at {record['Time']}\n")
                attendance_percentage = (attended_classes / total_classes * 100) if total_classes > 0 else 0
                self.result_text.insert("end", f"\nAttendance Percentage: {attendance_percentage:.2f}%\n")
            else:
                for record in self.attendance_records:
                    self.result_text.insert("end", f"{record['Date']} - {record['Class']} - {record['Name']} ({record['Enrollment']}) at {record['Time']} ({record['Confidence']}%)\n")
                self.result_text.insert("end", "\nSummary:\n")
                for class_name, count in class_summary.items():
                    self.result_text.insert("end", f"{class_name}: {count} entries\n")

    def download_as_csv(self):
        if not self.attendance_records:
            messagebox.showerror("Error", "No records to download.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            df = pd.DataFrame(self.attendance_records)
            df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Report saved as {path}")

    def download_as_pdf(self):
        if not self.attendance_records:
            messagebox.showerror("Error", "No records to download.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if path:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Attendance Report", ln=True, align="C")
            pdf.ln(10)
            for record in self.attendance_records:
                line = ", ".join([f"{key}: {value}" for key, value in record.items()])
                pdf.multi_cell(0, 10, txt=line)
            pdf.output(path)
            messagebox.showinfo("Saved", f"PDF saved as {path}")

if __name__ == "__main__":
    app = LoginApp()
    app.mainloop()


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import json
from flask import Flask, render_template, request, jsonify
import webbrowser
import csv
from crop3 import OMRMarkerDetector
from samplejee import process_omr_sheet
from datetime import datetime

class OMRProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OMR Sheet Processor")
        self.root.geometry("1200x800")

        # Instance variables
        self.image_files = []
        self.current_index = 0
        self.answer_key = None
        self.processing = False
        self.detector = OMRMarkerDetector()
        self.current_csv_file = self.generate_csv_filename()

        # Start Flask server in a separate thread
        self.flask_thread = threading.Thread(target=self.run_flask_server)
        self.flask_thread.daemon = True
        self.flask_thread.start()

        self.setup_gui()
        
    def generate_csv_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"omr_results_{timestamp}.csv"

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Buttons
        ttk.Button(control_frame, text="Upload OMR Sheet Folder", 
                  command=self.upload_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Input Answer Key", 
                  command=self.open_answer_key_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Answer Key", 
                  command=self.load_answer_key).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Process OMR Sheet", 
                  command=self.process_current_sheet).pack(side=tk.LEFT, padx=5)

        # Navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(nav_frame, text="Previous", 
                  command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", 
                  command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Cancel", 
                  command=self.cancel_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Reset", 
                  command=self.reset_gui).pack(side=tk.LEFT, padx=5)

        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=(5, 0))

    def run_flask_server(self):
        app = Flask(__name__)

        @app.route('/')
        def index():
            return render_template('answer_key.html')

        @app.route('/submit_answer_key', methods=['POST'])
        def submit_answer_key():
            data = request.get_json()
            with open('answer_key.json', 'w') as f:
                json.dump(data, f)
            return jsonify({"status": "success"})

        app.run(port=5000)

    def open_answer_key_page(self):
        webbrowser.open('http://localhost:5000')

    def load_answer_key(self):
        try:
            with open('answer_key.json', 'r') as f:
                self.answer_key = json.load(f)
            messagebox.showinfo("Success", "Answer key loaded successfully!")
        except:
            messagebox.showerror("Error", "Failed to load answer key!")

    def upload_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if self.image_files:
                self.current_index = 0
                self.display_current_image()
            else:
                messagebox.showwarning("Warning", "No image files found in the selected folder!")

    def display_current_image(self):
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]
        image = Image.open(image_path)
        
        # Resize image to fit the window 
        display_size = (800, 600)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def process_current_sheet(self):
        if not self.image_files or not self.answer_key:
            messagebox.showerror("Error", "Please load both images and answer key!")
            return

        self.processing = True
        threading.Thread(target=self._process_sheet).start()

    def _process_sheet(self):
        try:
            image_path = self.image_files[self.current_index]
            
            # Update status
            self.status_var.set("Detecting markers...")
            self.progress_var.set(25)

            # Detect markers and correct perspective
            result_image, markers = self.detector.detect_markers(image_path)
            if len(markers) != 4:
                raise Exception("Failed to detect all markers")

            corrected = self.detector.correct_perspective(cv2.imread(image_path), markers)
            if corrected is None:
                raise Exception("Failed to correct perspective")

            # Save corrected image
            temp_path = "temp_corrected.jpg"
            cv2.imwrite(temp_path, corrected)

            # Update status
            self.status_var.set("Processing OMR sheet...")
            self.progress_var.set(50)

            # Process OMR sheet
            results, _ = process_omr_sheet(temp_path, "template.json")

            # Generate results
            self.status_var.set("Generating results...")
            self.progress_var.set(75)

            self.generate_results(results)

            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

            self.status_var.set("Processing complete!")
            self.progress_var.set(100)

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Processing failed!")
            self.progress_var.set(0)

        finally:
            self.processing = False

    def generate_results(self, results):
        # Extract answers from results
        answers = []
        for section in ['physics', 'chemistry', 'mathematics']:
            section_answers = results.get(section, [])
            # Convert numerical answers to letters or special cases
            for ans in section_answers:
                if ans == -1:
                    answers.append("No Answer")
                elif ans == "Multiple":
                    answers.append("Multiple")
                elif isinstance(ans, int):
                    answers.append(chr(65 + ans))  # Convert 0,1,2,3 to A,B,C,D
                else:
                    answers.append(ans)

        # Compare with answer key and calculate score
        correct_answers = 0
        for i, (student_ans, correct_ans) in enumerate(zip(answers, self.answer_key['answers'])):
            if student_ans not in ["No Answer", "Multiple"] and student_ans == correct_ans:
                correct_answers += 1

        # Calculate score
        total_questions = 90
        score = (correct_answers / total_questions) * 100

        # Prepare results dictionary
        results_dict = {
            'Roll Number': results['roll_number'],
            'Score': f"{score:.2f}%",
            'Correct Answers': correct_answers,
            'Total Questions': total_questions
        }

        # Add individual question answers
        for i, ans in enumerate(answers, 1):
            if i <= 30:
                results_dict[f'P{i}'] = ans
            elif i <= 60:
                results_dict[f'C{i-30}'] = ans
            else:
                results_dict[f'M{i-60}'] = ans

        # Use the current_csv_file instead of hardcoded filename
        file_exists = os.path.isfile(self.current_csv_file)
        
        with open(self.current_csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Roll Number', 'Score', 'Correct Answers', 'Total Questions'] + 
                [f'P{i}' for i in range(1, 31)] +
                [f'C{i}' for i in range(1, 31)] +
                [f'M{i}' for i in range(1, 31)]
            )
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(results_dict)

        messagebox.showinfo("Results", 
                          f"Roll Number: {results['roll_number']}\n"
                          f"Score: {score:.2f}%\n"
                          f"Correct Answers: {correct_answers}/{total_questions}")

    def previous_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_current_image()

    def cancel_processing(self):
        if self.processing:
            self.processing = False
            self.status_var.set("Processing cancelled")
            self.progress_var.set(0)

    def reset_gui(self):
        self.image_files = []
        self.current_index = 0
        self.answer_key = None
        self.processing = False
        self.status_var.set("Ready")
        self.progress_var.set(0)
        self.image_label.configure(image='')
        self.image_label.image = None
        # Generate new CSV filename on reset
        self.current_csv_file = self.generate_csv_filename()
def main():
    root = tk.Tk()
    app = OMRProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
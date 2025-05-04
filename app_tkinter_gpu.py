import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import cv2
import numpy as np
import time
import speech_recognition as sr
import threading
import queue
import ollama
from PIL import Image, ImageTk
import io
import base64
import pyttsx3
import os
import tempfile
from gtts import gTTS
import pygame
import torch
import concurrent.futures

class WebcamQAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Q&A App with Voice")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize state variables
        self.cap = None
        self.is_capturing = False
        self.frames = []
        self.captions = []
        self.combined_captions = ""
        self.current_frame_index = 0
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda:0")
        else:
            print("CUDA is not available, using CPU")
            self.device = torch.device("cpu")
        
        # Initialize thread pool for parallel processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Initialize TTS engine
        self.init_tts_engine()
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Create the UI
        self.create_ui()
        
        # Start webcam preview
        self.start_webcam()
        
        # Message queue for threaded operations
        self.message_queue = queue.Queue()
        self.process_messages()
        
    def init_tts_engine(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 0.9)
            self.tts_available = True
        except Exception as e:
            print(f"pyttsx3 initialization failed: {str(e)}")
            self.tts_engine = None
            self.tts_available = False
    
    def create_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create paned window (resizable split)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame (webcam and controls)
        left_frame = ttk.Frame(paned_window, width=500)
        paned_window.add(left_frame, weight=1)
        
        # Right frame (Q&A)
        right_frame = ttk.Frame(paned_window, width=500)
        paned_window.add(right_frame, weight=1)
        
        # Webcam preview label
        self.webcam_label = ttk.Label(left_frame)
        self.webcam_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(left_frame, text="Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Capture mode radio buttons
        self.capture_mode = tk.StringVar(value="photo")
        ttk.Label(controls_frame, text="Capture mode:").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(controls_frame, text="Photo", variable=self.capture_mode, value="photo").grid(row=0, column=1)
        ttk.Radiobutton(controls_frame, text="Video (10 sec)", variable=self.capture_mode, value="video").grid(row=0, column=2)
        
        # TTS options
        self.use_gtts = tk.BooleanVar(value=True)
        ttk.Label(controls_frame, text="Voice:").grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(controls_frame, text="Use Google TTS (online)", variable=self.use_gtts).grid(row=1, column=1, columnspan=2, sticky=tk.W)
        
        # CUDA status indicator
        cuda_status = "CUDA enabled" if self.cuda_available else "CUDA not available"
        cuda_color = "green" if self.cuda_available else "red"
        ttk.Label(controls_frame, text="GPU Status:").grid(row=2, column=0, sticky=tk.W)
        cuda_label = ttk.Label(controls_frame, text=cuda_status, foreground=cuda_color)
        cuda_label.grid(row=2, column=1, columnspan=2, sticky=tk.W)
        
        # Capture button
        self.capture_btn = ttk.Button(controls_frame, text="Start Capture", command=self.start_capture)
        self.capture_btn.grid(row=3, column=0, columnspan=3, pady=10)
    
        # Status label
        self.status_label = ttk.Label(left_frame, text="Ready. Press 'Start Capture' to begin.", font=("Arial", 10, "italic"))
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Right side - Q&A
        qa_title = ttk.Label(right_frame, text="Ask Questions", font=("Arial", 14, "bold"))
        qa_title.pack(anchor=tk.W, padx=5, pady=5)
        
        # Frame display
        self.frame_display_frame = ttk.LabelFrame(right_frame, text="Captured Media", padding=10)
        self.frame_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame navigation
        self.frame_nav_frame = ttk.Frame(self.frame_display_frame)
        self.frame_nav_frame.pack(fill=tk.X, pady=5)
        
        self.prev_frame_btn = ttk.Button(self.frame_nav_frame, text="< Previous", command=self.prev_frame, state=tk.DISABLED)
        self.prev_frame_btn.pack(side=tk.LEFT, padx=5)
        
        self.frame_label = ttk.Label(self.frame_nav_frame, text="No frames captured")
        self.frame_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.next_frame_btn = ttk.Button(self.frame_nav_frame, text="Next >", command=self.next_frame, state=tk.DISABLED)
        self.next_frame_btn.pack(side=tk.RIGHT, padx=5)
        
        # Frame image
        self.frame_image_label = ttk.Label(self.frame_display_frame)
        self.frame_image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Caption display
        self.caption_text = scrolledtext.ScrolledText(self.frame_display_frame, height=4, wrap=tk.WORD)
        self.caption_text.pack(fill=tk.X, padx=5, pady=5)
        self.caption_text.insert(tk.END, "Captions will appear here after capture")
        self.caption_text.config(state=tk.DISABLED)
        
        #voice command button
        self.voice_command_frame = ttk.Frame(self.frame_display_frame)
        self.voice_command_frame.pack(pady=10)

        self.ask_voice_btn = ttk.Button(self.voice_command_frame, text="Ask by Voice", command=self.ask_by_voice)
        self.ask_voice_btn.pack()
        
        # Question frame
        question_frame = ttk.LabelFrame(right_frame, text="Ask a Question", padding=10)
        question_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.question_entry = ttk.Entry(question_frame, font=("Arial", 12))
        self.question_entry.pack(fill=tk.X, padx=5, pady=5)
        self.question_entry.bind("<Return>", lambda e: self.ask_question())
        
        self.ask_btn = ttk.Button(question_frame, text="Ask", command=self.ask_question)
        self.ask_btn.pack(pady=5)
        
        # Answer display
        answer_frame = ttk.LabelFrame(right_frame, text="Answer", padding=10)
        answer_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.answer_text = scrolledtext.ScrolledText(answer_frame, wrap=tk.WORD)
        self.answer_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.answer_text.insert(tk.END, "Answers will appear here")
        self.answer_text.config(state=tk.DISABLED)
    
    def start_webcam(self):
        """Initialize webcam"""
        try:
            # Use CUDA-accelerated OpenCV backend if available
            if self.cuda_available:
                # Set OpenCV backend to use CUDA
                cv2.setUseOptimized(True)
                self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
            else:
                self.cap = cv2.VideoCapture(0)
                
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Use higher FPS if CUDA is available
            if self.cuda_available:
                self.cap.set(cv2.CAP_PROP_FPS, 60)
                
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")
    
    def update_preview(self):
        """Update webcam preview"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Process frame with CUDA if available
            if self.cuda_available:
                # Convert to torch tensor for CUDA processing
                frame_tensor = torch.from_numpy(frame).to(self.device).float()
                # Apply any CUDA operations here (like basic color correction)
                frame_tensor = frame_tensor / 255.0  # Normalize
                frame_tensor = torch.clamp(frame_tensor * 1.1, 0, 1)  # Simple enhancement
                # Convert back to numpy
                frame = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # Convert to RGB and then to ImageTk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.webcam_label.configure(image=img)
            self.webcam_label.image = img  # Keep a reference
        
        # Schedule the next update
        if not self.is_capturing:
            # Higher FPS if CUDA available
            update_interval = 16 if self.cuda_available else 30  # ~60 FPS with CUDA, ~33 FPS without
            self.root.after(update_interval, self.update_preview)
    
    def start_capture(self):
        """Start capturing photo or video"""
        if self.is_capturing:
            return
            
        self.is_capturing = True
        self.capture_btn.config(state=tk.DISABLED)
        
        mode = self.capture_mode.get()
        threading.Thread(target=self.capture_thread, args=(mode,), daemon=True).start()
    
    def gpu_process_frame(self, frame):
        """Process a frame using GPU if available"""
        if not self.cuda_available:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Convert to torch tensor and move to GPU
        frame_tensor = torch.from_numpy(frame).to(self.device)
        
        # Apply CUDA-accelerated processing
        # Convert color space (BGR to RGB)
        b, g, r = frame_tensor[:, :, 0], frame_tensor[:, :, 1], frame_tensor[:, :, 2]
        frame_rgb_tensor = torch.stack([r, g, b], dim=2)
        
        # Move back to CPU and convert to numpy
        return frame_rgb_tensor.cpu().numpy()
    
    def capture_thread(self, mode):
        """Threaded function to handle capture"""
        self.frames = []
        self.captions = []
        self.message_queue.put(("status", "Starting capture..."))
        
        # Countdown
        for i in range(3, 0, -1):
            self.message_queue.put(("status", f"Starting in {i}..."))
            time.sleep(1)
        
        if mode == "photo":
            # Capture photo
            self.message_queue.put(("status", "Taking photo..."))
            ret, frame = self.cap.read()
            if ret:
                # Process with GPU if available
                frame_rgb = self.gpu_process_frame(frame)
                self.frames.append(frame_rgb)
                self.message_queue.put(("status", "Photo captured!"))
            else:
                self.message_queue.put(("error", "Failed to capture photo"))
        else:
            # Capture video
            self.message_queue.put(("status", "Recording 10 second video..."))
            start_time = time.time()
            duration = 5 # seconds
            num_frames = 6 # 6 frames for 5 seconds
            frame_interval = duration / num_frames
            next_capture_time = start_time
            
            while time.time() - start_time < duration:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                elapsed = time.time() - start_time
                self.message_queue.put(("status", f"Recording: {elapsed:.1f}s / {duration}s"))
                
                # Capture frames at regular intervals
                if time.time() >= next_capture_time:
                    # Process with GPU if available
                    frame_rgb = self.gpu_process_frame(frame)
                    self.frames.append(frame_rgb)
                    next_capture_time = start_time + len(self.frames) * frame_interval
                
                # Shorter sleep time for smoother recording
                time.sleep(0.01)
        
        # Process frames
        if self.frames:
            self.message_queue.put(("status", f"Processing {len(self.frames)} frames..."))
            
            # Use thread pool for parallel caption generation
            future_captions = []
            for i, frame in enumerate(self.frames):
                # Convert to PIL Image
                pil_image = Image.fromarray(frame)
                
                self.message_queue.put(("status", f"Analyzing frame {i+1}/{len(self.frames)}..."))
                
                # Submit to thread pool
                future = self.thread_pool.submit(self.get_image_caption, pil_image)
                future_captions.append(future)
            
            # Collect results as they complete
            for i, future in enumerate(future_captions):
                try:
                    caption = future.result()
                    self.captions.append(caption)
                except Exception as e:
                    self.message_queue.put(("error", f"Error analyzing frame {i+1}: {str(e)}"))
                    self.captions.append("Analysis failed")
            
            # Combine captions
            if len(self.captions) == 1:
                self.combined_captions = f"Image description: {self.captions[0]}"
            else:
                captions_text = "\n".join([f"Frame {i+1}: {caption}" for i, caption in enumerate(self.captions)])
                self.combined_captions = captions_text
            
            # Update the UI with results
            self.message_queue.put(("frames_ready", None))
            self.message_queue.put(("status", "Analysis complete! You can now ask questions."))
            self.speak_text("Analysis complete. I'm ready for your questions.")
        else:
            self.message_queue.put(("status", "Capture failed. No frames obtained."))
        
        # Re-enable capture button
        self.message_queue.put(("enable_capture", None))
        self.is_capturing = False
        
        # Resume preview
        self.message_queue.put(("resume_preview", None))
    
    def process_messages(self):
        """Process messages from the queue"""
        try:
            while not self.message_queue.empty():
                message, data = self.message_queue.get_nowait()
                
                if message == "status":
                    self.status_label.config(text=data)
                elif message == "error":
                    messagebox.showerror("Error", data)
                elif message == "frames_ready":
                    self.update_frame_display()
                elif message == "enable_capture":
                    self.capture_btn.config(state=tk.NORMAL)
                elif message == "resume_preview":
                    self.update_preview()
                elif message == "speak":
                    self.speak_text(data)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_messages)
    
    def update_frame_display(self):
        """Update the frame display with captured frames"""
        if not self.frames:
            return
            
        self.current_frame_index = 0
        self.display_current_frame()
        
        # Update navigation buttons
        if len(self.frames) > 1:
            self.next_frame_btn.config(state=tk.NORMAL)
        else:
            self.next_frame_btn.config(state=tk.DISABLED)
        self.prev_frame_btn.config(state=tk.DISABLED)
    
    def display_current_frame(self):
        """Display the current frame"""
        if not self.frames or self.current_frame_index >= len(self.frames):
            return
            
        # Convert to ImageTk
        frame = self.frames[self.current_frame_index]
        img = Image.fromarray(frame)
        
        # Resize while maintaining aspect ratio
        width, height = img.size
        max_size = 400
        scale = min(max_size/width, max_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Use GPU-accelerated resizing if available
        if self.cuda_available:
            # Convert to tensor for faster resizing
            img_tensor = torch.from_numpy(np.array(img)).to(self.device)
            # CUDA-accelerated resize would go here (using custom CUDA kernels)
            # For now, move back to CPU and use PIL
            img_tensor = img_tensor.cpu()
            img = Image.fromarray(img_tensor.numpy())
            img = img.resize((new_width, new_height), Image.LANCZOS)
        else:
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update image label
        self.frame_image_label.configure(image=img_tk)
        self.frame_image_label.image = img_tk  # Keep a reference
        
        # Update frame label
        self.frame_label.config(text=f"Frame {self.current_frame_index + 1} of {len(self.frames)}")
        
        # Update caption
        if self.captions and self.current_frame_index < len(self.captions):
            self.caption_text.config(state=tk.NORMAL)
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(tk.END, self.captions[self.current_frame_index])
            self.caption_text.config(state=tk.DISABLED)
    
    def next_frame(self):
        """Show next frame"""
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.display_current_frame()
            
            # Update navigation buttons
            self.prev_frame_btn.config(state=tk.NORMAL)
            if self.current_frame_index >= len(self.frames) - 1:
                self.next_frame_btn.config(state=tk.DISABLED)
    
    def prev_frame(self):
        """Show previous frame"""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.display_current_frame()
            
            # Update navigation buttons
            self.next_frame_btn.config(state=tk.NORMAL)
            if self.current_frame_index <= 0:
                self.prev_frame_btn.config(state=tk.DISABLED)
    
    def ask_by_voice(self):
        """Capture voice input and use it as a question"""
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        try:
            # Update status
            self.status_label.config(text="Listening for your question...")
        
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)

            # Recognize speech
            question = recognizer.recognize_google(audio)
        
            # Update the question entry box with recognized text
            self.question_entry.delete(0, tk.END)
            self.question_entry.insert(0, question)

            # Set a status update for feedback
            self.status_label.config(text="Question ready. Click 'Ask' to submit.")

        except sr.WaitTimeoutError:
            self.status_label.config(text="Listening timed out. Please try again.")
        except sr.UnknownValueError:
            self.status_label.config(text="Could not understand audio. Please try again.")
        except sr.RequestError as e:
            self.status_label.config(text=f"STT Error: {e}")

    def ask_question(self):
        """Process a question about the captured media"""
        question = self.question_entry.get().strip()
        if not question or not self.combined_captions:
            return
            
        self.status_label.config(text="Thinking...")
        self.ask_btn.config(state=tk.DISABLED)
        
        # Clear current answer
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, "Thinking...")
        self.answer_text.config(state=tk.DISABLED)
        
        # Process in separate thread
        threading.Thread(target=self.answer_thread, args=(question,), daemon=True).start()
    
    def answer_thread(self, question):
        """Thread for answering questions"""
        try:
            answer = self.answer_question(self.combined_captions, question)
            
            # Update UI
            self.root.after(0, lambda: self.display_answer(answer))
            
            # Speak the answer
            self.message_queue.put(("speak", answer))
        except Exception as e:
            error_msg = f"Error answering question: {str(e)}"
            self.root.after(0, lambda: self.display_answer(error_msg))
            
        # Re-enable ask button
        self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.status_label.config(text="Ready for questions"))
    
    def display_answer(self, answer):
        """Display the answer in the UI"""
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, answer)
        self.answer_text.config(state=tk.DISABLED)
    
    def get_image_caption(self, image):
        """Use Ollama to generate a caption for an image with CUDA optimization"""
        # Resize image for better model compatibility
        image = image.resize((320, 320))

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode image as base64
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Generate caption using llava model
        # Add n_gpu flag to utilize CUDA if available
        model_params = {"n_gpu": -1} if self.cuda_available else {}
        
        response = ollama.generate(
            model="llava:7b-v1.5-q4_0",
            prompt="""Describe this image in extreme detail. Focus on:
1. People - number, gender, age range, facial features, hair color/style, if they're wearing glasses or watches
2. Clothing - colors, types, styles
3. Objects - all visible items, their colors and positions
4. Background - setting, colors, lighting
5. Actions - what people/objects appear to be doing

Be extremely specific about colors, positions, and small details that might be important for answering questions.""",
            images=[base64_image],
            stream=False,
            options=model_params
        )
        return response['response'].strip()
    
    def answer_question(self, captions, question):
        """Use Ollama to answer a question based on image captions with CUDA optimization"""
        prompt = f"""You are an AI assistant that answers questions about images and videos.
Based on the following descriptions, please answer this question:

{captions}

Question: {question}

Answer the question based only on what can be seen in the descriptions.
If you can't answer the question based on the given information, say so clearly.
Keep your answer concise for text-to-speech readability."""

        # Add n_gpu flag to utilize CUDA if available
        model_params = {"n_gpu": -1} if self.cuda_available else {}
        
        response = ollama.generate(
            model="mistral:7b-instruct-v0.2-q4_0",
            prompt=prompt,
            stream=False,
            options=model_params
        )
        return response['response'].strip()
    
    def speak_text(self, text):
        """Speak the given text using either pyttsx3 or gTTS"""
        if not text:
            return
            
        # Start in separate thread to avoid blocking UI
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
    
    def _speak_thread(self, text):
        """Thread function for speaking text"""
        # Truncate very long text for speech
        max_chars_for_speech = 1000
        if len(text) > max_chars_for_speech:
            text = text[:max_chars_for_speech] + "... (text truncated for speech)"
        
        try:
            # Choose TTS method
            if self.use_gtts.get():
                # Use Google TTS
                tts = gTTS(text=text, lang='en', slow=False)
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    tts.save(fp.name)
                    temp_audio_file = fp.name
                
                # Play audio
                pygame.mixer.music.load(temp_audio_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Clean up the temporary file
                if os.path.exists(temp_audio_file):
                    os.unlink(temp_audio_file)
            elif self.tts_available:
                # Use pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                print("No TTS method available")
        except Exception as e:
            print(f"TTS Error: {str(e)}")
    
    def on_closing(self):
        """Clean up on window close"""
        if self.cap is not None:
            self.cap.release()
        
        # Shutdown thread pool
        self.thread_pool.shutdown()
        
        self.root.destroy()

        # Clean up any left-over temporary files (just in case)
        temp_files = [f for f in os.listdir(tempfile.gettempdir()) if f.endswith('.mp3')]
        for file in temp_files:
            try:
                os.unlink(os.path.join(tempfile.gettempdir(), file))
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")

def main():
    root = tk.Tk()
    app = WebcamQAApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
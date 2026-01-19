import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from PIL import Image, ImageTk
import torch
from diffusers import ZImagePipeline
import numpy as np
import cv2
from datetime import datetime
import json

class ZImageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Z-Image Generator")
        self.root.geometry("1200x800")
        
        # Variables
        self.pipeline = None
        self.current_image = None
        self.output_dir = "outputs"
        self.device_status = tk.StringVar(value="Device: Unknown")
        self.dark_mode = tk.BooleanVar(value=True)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Add rounded rectangle method to Canvas
        def create_rounded_rect(canvas, x1, y1, x2, y2, radius=5, **kwargs):
            """Create a rounded rectangle on canvas"""
            points = []
            # Top edge
            points.extend([x1 + radius, y1, x2 - radius, y1])
            # Top-right corner
            points.extend([x2, y1, x2, y1 + radius])
            # Right edge
            points.extend([x2, y2 - radius, x2, y2])
            # Bottom-right corner
            points.extend([x2 - radius, y2, x2 - radius, y2])
            # Bottom edge
            points.extend([x1 + radius, y2, x1 + radius, y2])
            # Bottom-left corner
            points.extend([x1, y2, x1, y2 - radius])
            # Left edge
            points.extend([x1, y1 + radius, x1, y1])
            # Top-left corner
            points.extend([x1 + radius, y1, x1 + radius, y1])
            
            return canvas.create_polygon(points, smooth=True, **kwargs)
        
        # Monkey patch the method to canvas
        tk.Canvas.create_rounded_rect = create_rounded_rect
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Theme toggle
        theme_frame = ttk.LabelFrame(control_frame, text="Theme", padding="5")
        theme_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create custom toggle button frame
        toggle_frame = ttk.Frame(theme_frame)
        toggle_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(toggle_frame, text="Light:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Custom toggle button - simple clean design
        self.toggle_canvas = tk.Canvas(toggle_frame, width=40, height=20, bg="#cccccc", highlightthickness=0, bd=0)
        self.toggle_canvas.pack(side=tk.LEFT, padx=5)
        
        # Draw simple background
        self.toggle_canvas.create_rectangle(2, 2, 38, 18, fill="#cccccc", outline="")
        
        # Draw toggle circle
        self.toggle_circle = self.toggle_canvas.create_oval(3, 3, 17, 17, fill="white", outline="")
        self.toggle_canvas.bind("<Button-1>", self.toggle_theme_click)
        
        ttk.Label(toggle_frame, text="Dark").pack(side=tk.LEFT, padx=(5, 0))
        
        # Initialize toggle position
        self.update_toggle_visual()
        
        # Prompt input
        prompt_frame = ttk.LabelFrame(control_frame, text="Prompt", padding="5")
        prompt_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=4, width=40)
        self.prompt_text.pack(fill=tk.X)
        
        # Image settings
        image_frame = ttk.LabelFrame(control_frame, text="Image Settings", padding="5")
        image_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Get current screen size for default
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Resolution dropdown with more options based on screen sizes
        ttk.Label(image_frame, text="Resolution:").grid(row=0, column=0, sticky=tk.W, pady=2)
        resolution_options = [
            # Square formats
            "256x256",
            "512x512",
            "768x768", 
            "1024x1024",
            "1536x1536",
            "2048x2048",
            
            # Standard (4:3)
            "640x480",
            "800x600",
            "1024x768",
            "1280x960",
            "1600x1200",
            
            # Wide (16:9)
            "854x480",
            "1280x720",
            "1366x768",
            "1600x900",
            "1920x1080",
            "2560x1440",
            "3840x2160",
            
            # Ultra-wide (21:9)
            "2560x1080",
            "3440x1440",
            "5120x2160",
            
            # Mobile formats
            "1080x1920",
            "750x1334",
            "828x1792",
            "1242x2688"
        ]
        
        # Set default to current screen resolution or closest match
        default_resolution = f"{screen_width}x{screen_height}"
        if default_resolution not in resolution_options:
            # Find closest standard resolution
            if screen_width >= 3840:
                default_resolution = "3840x2160"
            elif screen_width >= 2560:
                default_resolution = "2560x1440"
            elif screen_width >= 1920:
                default_resolution = "1920x1080"
            elif screen_width >= 1366:
                default_resolution = "1366x768"
            else:
                default_resolution = "1280x720"
        
        self.resolution_var = tk.StringVar(value=default_resolution)
        self.resolution_combo = ttk.Combobox(image_frame, textvariable=self.resolution_var, values=resolution_options, width=20, state="readonly")
        self.resolution_combo.grid(row=0, column=1, pady=2, sticky=tk.W)
        
        # Steps
        ttk.Label(image_frame, text="Steps:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.steps_var = tk.StringVar(value="4")
        ttk.Entry(image_frame, textvariable=self.steps_var, width=10).grid(row=1, column=1, pady=2, sticky=tk.W)
        
        # Guidance
        ttk.Label(image_frame, text="Guidance:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.guidance_var = tk.StringVar(value="0.0")
        ttk.Entry(image_frame, textvariable=self.guidance_var, width=10).grid(row=2, column=1, pady=2, sticky=tk.W)
        
        # Seed
        ttk.Label(image_frame, text="Seed:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(image_frame, textvariable=self.seed_var, width=10).grid(row=3, column=1, pady=2, sticky=tk.W)
        
        # Random seed button
        ttk.Button(image_frame, text="Random", command=self.random_seed, width=8).grid(row=3, column=2, pady=2, padx=(10, 0), sticky=tk.W)
        
        # Model settings
        model_frame = ttk.LabelFrame(control_frame, text="Model", padding="5")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_var = tk.StringVar(value="Tongyi-MAI/Z-Image-Turbo")
        models = [
            "Tongyi-MAI/Z-Image-Turbo"
        ]
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=models, width=35)
        self.model_combo.pack(fill=tk.X)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.load_button = ttk.Button(button_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(fill=tk.X, pady=2)
        
        # Create a frame for generate and save buttons to be side by side
        generate_save_frame = ttk.Frame(button_frame)
        generate_save_frame.pack(fill=tk.X, pady=2)
        
        self.generate_button = ttk.Button(generate_save_frame, text="Generate", command=self.generate, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.save_button = ttk.Button(generate_save_frame, text="Save", command=self.save_output, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Batch processing
        batch_frame = ttk.LabelFrame(control_frame, text="Batch Processing", padding="5")
        batch_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(batch_frame, text="Load Prompts from File", command=self.load_prompts_file).pack(fill=tk.X, pady=2)
        ttk.Button(batch_frame, text="Batch Generate", command=self.batch_generate).pack(fill=tk.X, pady=2)
        
        # Device status
        device_frame = ttk.LabelFrame(control_frame, text="Device Status", padding="5")
        device_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.device_label = ttk.Label(device_frame, textvariable=self.device_status, font=("Arial", 10, "bold"))
        self.device_label.pack(fill=tk.X)
        
        # Update device status on startup
        self.update_device_status()
        
        # Right panel - Output
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # Canvas for image display
        self.canvas = tk.Canvas(output_frame, bg="white", width=600, height=600)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Apply initial theme after all widgets are created
        self.toggle_theme()
        
    def toggle_theme_click(self, event):
        """Handle toggle button click"""
        self.dark_mode.set(not self.dark_mode.get())
        self.toggle_theme()
        self.update_toggle_visual()
    
    def update_toggle_visual(self):
        """Update visual appearance of toggle button"""
        if self.dark_mode.get():
            # Dark mode - toggle to the right
            bg_color = "#4a4a4a"
            circle_x = 23  # Move circle to right
        else:
            # Light mode - toggle to the left
            bg_color = "#cccccc"
            circle_x = 3   # Circle on left side
        
        # Update background rectangle
        self.toggle_canvas.itemconfig(self.toggle_canvas.find_all()[0], fill=bg_color)
        
        # Update circle position
        self.toggle_canvas.coords(self.toggle_circle, circle_x, 3, circle_x + 14, 17)
    
    def toggle_theme(self):
        """Toggle between dark and light theme for entire window"""
        if self.dark_mode.get():
            # Dark theme with better contrast
            style = ttk.Style()
            style.theme_use('default')
            
            # Configure dark theme colors with better contrast
            bg_color = "#1e1e1e"  # Darker background
            fg_color = "#ffffff"  # Pure white text
            disabled_fg = "#888888"  # Gray for disabled text
            select_bg = "#404040"  # Medium dark for buttons
            frame_bg = "#2d2d2d"  # Slightly lighter than main bg
            
            # Configure styles with better contrast
            style.configure('.', background=bg_color, foreground=fg_color)
            style.configure('TLabel', background=frame_bg, foreground=fg_color, font=('Arial', 9))
            style.configure('TButton', background=select_bg, foreground=fg_color, font=('Arial', 9, 'bold'))
            style.configure('TLabelFrame', background=frame_bg, foreground=fg_color, font=('Arial', 10, 'bold'))
            style.configure('TFrame', background=frame_bg)
            style.configure('TEntry', fieldbackground=bg_color, foreground=fg_color, font=('Arial', 9))
            style.configure('TCombobox', fieldbackground=bg_color, background=select_bg, foreground=fg_color, font=('Arial', 9))
            
            # Configure dropdown list colors
            style.configure('TCombobox.Listbox', background=bg_color, foreground=fg_color, selectbackground=select_bg, selectforeground=fg_color)
            style.configure('Combobox.Popdown', background=bg_color, foreground=fg_color)
            style.configure('Treeview', background=bg_color, foreground=fg_color, fieldbackground=bg_color)
            
            # Configure disabled button style
            style.map('TButton', 
                     background=[('disabled', '#2a2a2a')],
                     foreground=[('disabled', disabled_fg)])
            
            # Configure root
            self.root.configure(bg=bg_color)
            
        else:
            # Light theme with better contrast
            style = ttk.Style()
            style.theme_use('default')
            
            # Configure light theme colors with better contrast
            bg_color = "#ffffff"  # Pure white background
            fg_color = "#000000"  # Pure black text
            disabled_fg = "#666666"  # Gray for disabled text
            select_bg = "#e0e0e0"  # Light gray for buttons
            frame_bg = "#f5f5f5"  # Very light gray for frames
            
            # Configure light theme colors with better contrast
            style.configure('.', background=bg_color, foreground=fg_color)
            style.configure('TLabel', background=frame_bg, foreground=fg_color, font=('Arial', 9))
            style.configure('TButton', background=select_bg, foreground=fg_color, font=('Arial', 9, 'bold'))
            style.configure('TLabelFrame', background=frame_bg, foreground=fg_color, font=('Arial', 10, 'bold'))
            style.configure('TFrame', background=frame_bg)
            style.configure('TEntry', fieldbackground=bg_color, foreground=fg_color, font=('Arial', 9))
            style.configure('TCombobox', fieldbackground=bg_color, background=select_bg, foreground=fg_color, font=('Arial', 9))
            
            # Configure dropdown list colors
            style.configure('TCombobox.Listbox', background=bg_color, foreground=fg_color, selectbackground=select_bg, selectforeground=fg_color)
            style.configure('Combobox.Popdown', background=bg_color, foreground=fg_color)
            style.configure('Treeview', background=bg_color, foreground=fg_color, fieldbackground=bg_color)
            
            # Configure disabled button style
            style.map('TButton', 
                     background=[('disabled', '#f0f0f0')],
                     foreground=[('disabled', disabled_fg)])
            
            # Configure root
            self.root.configure(bg=bg_color)
        
        # Apply to specific widgets directly
        self.apply_direct_theme_changes()
        
        # Apply theme to combobox directly
        if hasattr(self, 'resolution_combo'):
            # Keep font color consistent regardless of theme
            self.resolution_combo.configure(foreground="#000000")
            
            if self.dark_mode.get():
                self.resolution_combo.configure(background="#404040")
                # Force fieldbackground through style
                style = ttk.Style()
                style.configure('Custom.TCombobox', fieldbackground="#1e1e1e", background="#404040", foreground="#000000")
                self.resolution_combo.configure(style='Custom.TCombobox')
            else:
                self.resolution_combo.configure(background="#e0e0e0")
                # Force fieldbackground through style
                style = ttk.Style()
                style.configure('Custom.TCombobox', fieldbackground="#ffffff", background="#e0e0e0", foreground="#000000")
                self.resolution_combo.configure(style='Custom.TCombobox')
            
            # Bind events to fix dropdown colors
            self.resolution_combo.bind('<ButtonPress-1>', self.fix_combobox_dropdown)
            self.resolution_combo.bind('<KeyPress-Down>', self.fix_combobox_dropdown)
    
    def apply_direct_theme_changes(self):
        """Apply theme changes directly to specific widgets"""
        if self.dark_mode.get():
            # Dark theme colors with better contrast
            bg_color = "#1e1e1e"
            fg_color = "#ffffff"
            canvas_bg = "#2d2d2d"
        else:
            # Light theme colors with better contrast
            bg_color = "#ffffff"
            fg_color = "#000000"
            canvas_bg = "#f5f5f5"
        
        # Apply to specific widgets directly
        try:
            # Canvas (image display)
            self.canvas.configure(bg=canvas_bg)
            
            # Text widget (prompt input)
            self.prompt_text.configure(bg=canvas_bg, fg=fg_color, insertbackground=fg_color, font=('Arial', 10))
            
            # Status bar
            if hasattr(self, 'status_var'):
                # Status bar is a label, will be styled by ttk
                pass
                
        except Exception as e:
            print(f"Direct theme error: {e}")
            pass
    
    def fix_combobox_dropdown(self, event):
        """Fix combobox dropdown colors when opened"""
        try:
            # Schedule the fix to run after the dropdown is displayed
            self.root.after(100, self._apply_dropdown_colors)
        except Exception as e:
            print(f"Dropdown fix error: {e}")
    
    def _apply_dropdown_colors(self):
        """Apply colors to the dropdown listbox"""
        try:
            # Find the listbox widget in the combobox
            combobox = self.resolution_combo
            listbox = None
            
            # Try to find the listbox child
            for child in combobox.winfo_children():
                if str(child).endswith('listbox'):
                    listbox = child
                    break
            
            if listbox:
                # Keep font color consistent regardless of theme
                listbox.configure(fg="#000000")
                
                if self.dark_mode.get():
                    listbox.configure(bg="#1e1e1e", selectbackground="#404040")
                else:
                    listbox.configure(bg="#ffffff", selectbackground="#e0e0e0")
        except Exception as e:
            print(f"Listbox color error: {e}")
    
    def random_seed(self):
        import random
        self.seed_var.set(str(random.randint(0, 2**32 - 1)))
    
    def update_device_status(self):
        """Update the device status display - attempt CUDA first"""
        try:
            if torch.cuda.is_available():
                # Test CUDA by attempting to get device info
                device_name = torch.cuda.get_device_name(0)
                self.device_status.set(f"Device: CUDA ({device_name})")
            else:
                raise RuntimeError("CUDA not available")
        except Exception:
            self.device_status.set("Device: CPU (CUDA unavailable)")
        
    def load_model(self):
        def load_in_thread():
            try:
                self.status_var.set("Loading model...")
                self.progress.start()
                self.load_button.config(state=tk.DISABLED)
                
                model_name = self.model_var.get()
                
                # Load Z-Image pipeline
                self.pipeline = ZImagePipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                )
                
                # Attempt CUDA first, fallback to CPU if it fails
                try:
                    if torch.cuda.is_available():
                        self.status_var.set("Attempting to load model on CUDA...")
                        self.pipeline = self.pipeline.to("cuda")
                        device_name = torch.cuda.get_device_name(0)
                        self.device_status.set(f"Device: CUDA ({device_name})")
                        self.status_var.set(f"Model loaded on GPU: {device_name}")
                    else:
                        raise RuntimeError("CUDA not available")
                except Exception as cuda_error:
                    # Fallback to CPU
                    self.status_var.set("CUDA failed, falling back to CPU...")
                    self.device_status.set("Device: CPU (CUDA fallback)")
                    self.status_var.set("Model loaded on CPU (CUDA failed)")
                
                self.generate_button.config(state=tk.NORMAL)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.status_var.set("Model loading failed")
            finally:
                self.progress.stop()
                self.load_button.config(state=tk.NORMAL)
        
        threading.Thread(target=load_in_thread, daemon=True).start()
        
    def adjust_dimensions_for_model(self, width, height):
        """Adjust dimensions to be divisible by 16 while preserving aspect ratio"""
        # Round down to nearest multiple of 16
        adjusted_width = (width // 16) * 16
        adjusted_height = (height // 16) * 16
        
        # Ensure minimum dimensions
        if adjusted_width < 256:
            adjusted_width = 256
        if adjusted_height < 256:
            adjusted_height = 256
        
        return adjusted_width, adjusted_height
    
    def generate(self):
        if not self.pipeline:
            messagebox.showwarning("Warning", "Please load a model first")
            return
            
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt")
            return
            
        def generate_in_thread():
            try:
                self.status_var.set("Generating...")
                self.progress.start()
                self.generate_button.config(state=tk.DISABLED)
                
                # Get parameters and adjust for model requirements
                resolution = self.resolution_var.get()
                width, height = map(int, resolution.split('x'))
                
                # Adjust dimensions to be divisible by 16
                adjusted_width, adjusted_height = self.adjust_dimensions_for_model(width, height)
                
                # Show if dimensions were adjusted
                if (width, height) != (adjusted_width, adjusted_height):
                    self.status_var.set(f"Adjusted dimensions: {width}x{height} → {adjusted_width}x{adjusted_height}")
                
                steps = int(self.steps_var.get())
                guidance = float(self.guidance_var.get())
                
                seed = self.seed_var.get()
                if seed:
                    generator = torch.Generator().manual_seed(int(seed))
                else:
                    generator = None
                
                # Generate image
                self.status_var.set("Generating image...")
                result = self.pipeline(
                    prompt=prompt,
                    width=adjusted_width,
                    height=adjusted_height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator
                )
                
                # Debug: Check if result has images
                if hasattr(result, 'images') and len(result.images) > 0:
                    self.current_image = result.images[0]
                    self.status_var.set(f"Image generated: {self.current_image.size}")
                    self.display_image(self.current_image)
                    self.status_var.set("Image generated successfully")
                else:
                    self.status_var.set("Error: No image in result")
                    messagebox.showerror("Error", "Generation completed but no image was produced")
                
                self.save_button.config(state=tk.NORMAL)
                
            except Exception as e:
                messagebox.showerror("Error", f"Generation failed: {str(e)}")
                self.status_var.set("Generation failed")
            finally:
                self.progress.stop()
                self.generate_button.config(state=tk.NORMAL)
        
        threading.Thread(target=generate_in_thread, daemon=True).start()
        
    def display_image(self, image):
        if image is None:
            self.status_var.set("Error: No image to display")
            return
            
        try:
            # Resize image to fit canvas
            canvas_width = 600
            canvas_height = 600
            
            # Calculate aspect ratio
            img_width, img_height = image.size
            aspect_ratio = img_width / img_height
            
            if aspect_ratio > 1:
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            
            image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image_resized)
            
            # Display on canvas
            self.canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
            
            # Keep reference
            self.canvas.image = photo
            self.status_var.set(f"Image displayed: {img_width}x{img_height}")
            
        except Exception as e:
            self.status_var.set(f"Error displaying image: {str(e)}")
            messagebox.showerror("Display Error", f"Failed to display image: {str(e)}")
    
    def save_output(self):
        if not self.current_image:
            messagebox.showwarning("Warning", "No image to save")
            return
            
        try:
            # Generate default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            prompt_short = prompt[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
            default_filename = f"zimage_{timestamp}_{prompt_short}.png"
            
            # Open file dialog to select save location
            filepath = filedialog.asksaveasfilename(
                title="Save Image",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                initialfile=default_filename
            )
            
            if not filepath:
                return  # User cancelled
            
            # Save image to selected location
            self.current_image.save(filepath)
            self.status_var.set(f"Image saved: {os.path.basename(filepath)}")
            messagebox.showinfo("Success", f"Image saved to:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def load_prompts_file(self):
        try:
            filename = filedialog.askopenfilename(
                title="Load Prompts File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if not filename:
                return
                
            with open(filename, 'r', encoding='utf-8') as f:
                prompts = f.read()
                
            # Clear and set new prompts
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", prompts)
            
            self.status_var.set(f"Loaded {len(prompts.splitlines())} prompts from {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load prompts: {str(e)}")
                
    def batch_generate(self):
        prompts_text = self.prompt_text.get("1.0", tk.END).strip()
        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
        
        if not prompts:
            messagebox.showwarning("Warning", "No prompts found")
            return
            
        if not self.pipeline:
            messagebox.showwarning("Warning", "Please load a model first")
            return
            
        def batch_generate_in_thread():
            try:
                self.status_var.set("Batch generating...")
                self.progress.start()
                self.generate_button.config(state=tk.DISABLED)
                
                # Get parameters and adjust for model requirements
                resolution = self.resolution_var.get()
                width, height = map(int, resolution.split('x'))
                
                # Adjust dimensions to be divisible by 16
                adjusted_width, adjusted_height = self.adjust_dimensions_for_model(width, height)
                
                steps = int(self.steps_var.get())
                guidance = float(self.guidance_var.get())
                
                for i, prompt in enumerate(prompts):
                    self.status_var.set(f"Generating {i+1}/{len(prompts)}: {prompt[:50]}...")
                    
                    result = self.pipeline(
                        prompt=prompt,
                        width=adjusted_width,
                        height=adjusted_height,
                        num_inference_steps=steps,
                        guidance_scale=guidance
                    )
                    
                    # Save each image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.output_dir, f"batch_{timestamp}_{i+1:03d}.png")
                    result.images[0].save(filename)
                
                self.status_var.set(f"Batch complete: {len(prompts)} images saved")
                
            except Exception as e:
                messagebox.showerror("Error", f"Batch generation failed: {str(e)}")
                self.status_var.set("Batch generation failed")
            finally:
                self.progress.stop()
                self.generate_button.config(state=tk.NORMAL)
        
        threading.Thread(target=batch_generate_in_thread, daemon=True).start()

def main():
    root = tk.Tk()
    app = ZImageGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

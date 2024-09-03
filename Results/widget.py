import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import muSpectre
from muFFT import Stencils2D

# Define the model
def build_decoder(latent_dim, property_shape, output_shape):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    alpha = tf.keras.Input(shape=property_shape)
    x = layers.Concatenate()([latent_inputs, alpha])
    x = layers.Dense(8 * 8 * 64, activation='relu')(x)
    x = layers.Reshape((8, 8, 64))(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu', strides=1, padding='same')(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(32, 3, activation='relu', strides=1, padding='same')(x)

    x = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    outputs = layers.Cropping2D(cropping=((3, 0), (3, 0)))(x)

    decoder = tf.keras.Model([latent_inputs, alpha], outputs, name='decoder')
    return decoder

# Load the trained model
latent_dim = 2  # Adjust based on your model's latent dimension
property_shape = (1,)  # Adjust based on your model's alpha shape
output_shape = (29, 29, 1)  # Adjust based on your model's output shape
decoder = build_decoder(latent_dim, property_shape, output_shape)
decoder.load_weights('/Users/User/Documents/Inverse Design of Metamaterials/Results/image_generation_weights.h5')

# Function to generate image from alpha
def generate_image(alpha, num_images=1):
    images = []
    for _ in range(num_images):
        latent_input = np.random.normal(size=(1, latent_dim))
        alpha_input = np.array([[alpha]])
        image = decoder.predict([latent_input, alpha_input])[0]
        thresholded_image = np.round(image)
        images.append(thresholded_image)
    return images

# Function to predict alpha from a given image
def index_to_voigt(C):
    n = np.zeros((4,4))
    n = C
    C_voigt =  np.zeros((3,3))
    C_voigt[0,0] = n[0, 0]
    C_voigt[0,1] = n[0, 3]
    C_voigt[1,0] = n[3, 0]
    C_voigt[1,1] = n[3, 3]
    C_voigt[2,2] = n[2, 2]
    return C_voigt

def predict_alpha(image):
    nb_grid_pts = image.shape
    domain_lens = [1, 1]
    gradient = Stencils2D.linear_finite_elements
    weights = [1,1]
    cell = muSpectre.cell.CellData.make(nb_grid_pts, domain_lens)
    cell.nb_quad_pts = 2
    Mat = muSpectre.material.MaterialLinearElastic1_2d
    young_soft = 0
    young_hard = 1
    poisson = .33
    soft = Mat.make(cell,"soft", young_soft, poisson)
    hard = Mat.make(cell,"hard", young_hard, poisson)
    
    for i, pixel in cell.pixels.enumerate():
        if image[tuple(pixel)] == 0:
            hard.add_pixel(i)
        else:
            soft.add_pixel(i)
    cg_tol = 2e-8
    equi_tol = 0.001
    maxiter = 500  # for linear cell solver
    verbose_krylov = muSpectre.Verbosity.Silent

    krylov_solver = muSpectre.solvers.KrylovSolverCG(cg_tol, maxiter, verbose_krylov)
    newton_tol = 2e-8

    verbose_newton= muSpectre.Verbosity.Full
    control = muSpectre.solvers.MeanControl.strain_control
    newton_solver = muSpectre.solvers.SolverNewtonCG(cell,
                                                    krylov_solver,
                                                    verbose_newton,
                                                    newton_tol,
                                                    equi_tol,
                                                    maxiter,
                                                    gradient,
                                                    weights,
                                                    control)
    
    newton_solver.formulation = muSpectre.Formulation.small_strain
 
    newton_solver.initialise_cell()
    newton_solver.evaluate_stress_tangent()
    # compute effective stiffness tangent 
    C_eff = newton_solver.compute_effective_stiffness()
    C_voigt = index_to_voigt(C_eff)
    alpha = C_voigt[0,0]/C_voigt[1,1]
    return alpha

# Create the GUI
class ImageGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Metamaterial Microstructure Generator")

        self.label_alpha = tk.Label(master, text="Enter alpha value:")
        self.label_alpha.pack()

        self.entry_alpha = tk.Entry(master)
        self.entry_alpha.pack()

        self.label_num_images = tk.Label(master, text="Enter number of geometries to generate (1-10):")
        self.label_num_images.pack()

        self.entry_num_images = tk.Entry(master)
        self.entry_num_images.pack()

        self.generate_button = tk.Button(master, text="Generate Geometries", command=self.generate_images)
        self.generate_button.pack()

        self.predict_button = tk.Button(master, text="Predict Alphas", command=self.predict_alphas)
        self.predict_button.pack()

        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack()

        self.images = []  # Initialize images attribute
        self.alpha_labels = []  # Initialize alpha labels attribute

    def generate_images(self):
        alpha_str = self.entry_alpha.get()
        num_images_str = self.entry_num_images.get()

        try:
            self.alpha = float(alpha_str)
            self.num_images = int(num_images_str)
            if self.num_images < 1 or self.num_images > 10:
                raise ValueError("Number of images must be between 1 and 10")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        self.images = generate_image(self.alpha, self.num_images)

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        self.alpha_labels = []

        for idx, image in enumerate(self.images):
            image_display = (image * 255).astype(np.uint8)
            image_display = Image.fromarray(image_display.squeeze(), 'L')
            image_display = image_display.resize((output_shape[1]*5, output_shape[0]*5), Image.NEAREST)
            image_display = ImageTk.PhotoImage(image_display)

            canvas = tk.Canvas(self.canvas_frame, width=output_shape[1]*5, height=output_shape[0]*5)
            canvas.grid(row=0, column=idx, padx=10, pady=10)
            canvas.create_image(0, 0, anchor=tk.NW, image=image_display)
            canvas.image = image_display

            label = tk.Label(self.canvas_frame, text="Predicted Alpha: N/A")
            label.grid(row=1, column=idx, padx=10, pady=5)
            self.alpha_labels.append(label)

    def predict_alphas(self):
        if not self.images:
            messagebox.showerror("Error", "No images generated yet")
            return

        for idx, image in enumerate(self.images):
            predicted_alpha = predict_alpha(image.squeeze())
            self.alpha_labels[idx].config(text=f"Predicted Alpha: {predicted_alpha:.2f}")

root = tk.Tk()
app = ImageGeneratorApp(root)
root.mainloop()


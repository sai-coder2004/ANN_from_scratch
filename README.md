# ANN_from_scratch
📄 Project Documentation: Manual Neural Network in NumPy — Clean & Improved Version
🧠 Project Title
"From Scratch to Stability: Building a Neural Network in Pure NumPy"

📌 Objective
To build and train a simple 2-layer neural network using NumPy to predict salary (lpa) based on features like cgpa and profile_score. The project focuses on understanding forward propagation, backpropagation, and implementing stabilization techniques to prevent common pitfalls.

🧊 Dataset
python
Copy
Edit
df = pd.DataFrame([
    [8, 8, 4],
    [7, 9, 5],
    [6, 10, 6],
    [5, 12, 7]
], columns=['cgpa', 'profile_score', 'lpa'])
Inputs: CGPA, Profile Score

Output: LPA (Lakhs Per Annum)

Size: 4 samples

Challenge: Small dataset — requires careful handling to avoid overfitting and instability

🧱 Model Architecture
Layer	Neurons	Activation
Input Layer	2	None
Hidden Layer	2	ReLU
Output Layer	1	Linear

🧮 Implementation Steps
Parameter Initialization

Random small weights, zero biases

Shape based on architecture: [2,2,1]

Input Normalization

Used z-score normalization to bring input features to standard scale

Forward Propagation

Hidden layer: ReLU(W1^T x + b1)

Output layer: W2^T x + b2

Loss Function

Mean Squared Error (MSE)

Backpropagation

Manual gradient computation

Includes derivative of ReLU for hidden layer

Training Loop

Stochastic updates using a small learning rate

Periodic reporting of average loss

Stabilization Techniques

Gradient clipping to avoid exploding values

ReLU to avoid unbounded linear activations

Learning rate tuning for slow, steady learning

✅ Highlights of the Final Model
✔️ Stable training across 1000+ epochs

✔️ Gradients and outputs remain within bounds

✔️ Accurate prediction with minimal data

✔️ Demonstrates core understanding of backpropagation and model tuning

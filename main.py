import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import uvicorn
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Step 1: Generate Synthetic Dataset
def generate_synthetic_data(num_samples=1000, board_size=(10, 10)):
    data, labels = [], []
    
    for _ in range(num_samples):
        components = np.random.randint(0, board_size[0], size=(5, 2))  # 5 components (x, y)
        label = np.sort(components, axis=0)  # Simplified target layout
        data.append(components.flatten())
        labels.append(label.flatten())
    
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

# Step 2: Define Neural Network Model
class LayoutNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LayoutNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Step 3: Train Model
def train_model():
    data, labels = generate_synthetic_data()
    train_x = torch.tensor(data)
    train_y = torch.tensor(labels)
    
    model = LayoutNet(10, 10)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    return model

model = train_model()

# Step 4: Deploy Model with FastAPI
app = FastAPI()

class BoardInput(BaseModel):
    board_input: List[float]

@app.post("/predict")
def predict_layout(input_data: BoardInput):
    input_tensor = torch.tensor([input_data.board_input], dtype=torch.float32)
    output = model(input_tensor).detach().numpy().reshape(5, 2)  # Reshape back to (5,2)
    return {"optimized_layout": output.tolist()}

# Step 5: Visualization Function
def visualize_layout(original, optimized):
    plt.scatter(original[:, 0], original[:, 1], color='red', label='Original')
    plt.scatter(optimized[:, 0], optimized[:, 1], color='blue', label='Optimized')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8002)


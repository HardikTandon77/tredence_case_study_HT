import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Part 1: The Prunable Linear Layer

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gate scores
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases using Kaiming Uniform (standard PyTorch linear init)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate_scores to a positive value (e.g., 3.0) 
        # so sigmoid(3.0) ~ 0.95, meaning all weights start mostly "active"
        nn.init.constant_(self.gate_scores, 3.0)

    def forward(self, x):
        # 1. Transform gate_scores to gates between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # 2. Calculate pruned weights
        pruned_weights = self.weight * gates
        
        # 3. Perform standard linear operation
        return F.linear(x, pruned_weights, self.bias)


# Defining the Neural Network

class PrunableMLP(nn.Module):
    def __init__(self):
        super(PrunableMLP, self).__init__()
        # Flattened CIFAR-10 image: 3 channels * 32 * 32 = 3072
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Part 2 & 3: Sparsity Loss, Training, and Evaluation

def calculate_sparsity_loss(model):
    """Calculates the L1 norm of all gate values in the network."""
    l1_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            l1_loss += torch.sum(gates)
    return l1_loss

def calculate_sparsity_level(model, threshold=1e-2):
    """Calculates the percentage of gates below the threshold."""
    total_weights = 0
    pruned_weights = 0
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                pruned_weights += torch.sum(gates < threshold).item()
                total_weights += gates.numel()
                
    return (pruned_weights / total_weights) * 100.0

def train_and_evaluate(lmbda, epochs=5, device='cpu'):
    # Data loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = PrunableMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Training with Lambda: {lmbda} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Classification Loss
            cls_loss = criterion(outputs, targets)
            
            # Sparsity Loss
            sparsity_loss = calculate_sparsity_loss(model)
            
            # Total Loss formulation
            loss = cls_loss + (lmbda * sparsity_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    sparsity = calculate_sparsity_level(model)
    
    print(f"Test Accuracy: {accuracy:.2f}% | Sparsity Level: {sparsity:.2f}%")
    return model, accuracy, sparsity

def plot_gate_distribution(model):
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates)
    
    plt.figure(figsize=(8, 5))
    plt.hist(all_gates, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Final Gate Values")
    plt.xlabel("Gate Value (Sigmoid Output)")
    plt.ylabel("Frequency")
    plt.yscale('log') # Log scale
    plt.grid(axis='y', alpha=0.75)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Three different values of lambda to compare
    lambdas_to_test = [0.0, 0.0001, 0.001]
    results = []
    
    best_model = None
    best_lambda = None
    
    for lmbda in lambdas_to_test:
        model, acc, sparsity = train_and_evaluate(lmbda, epochs=10, device=device)
        results.append((lmbda, acc, sparsity))
        
        # Save a moderately regularized model to plot (where lambda = 0.0001)
        if lmbda == 0.0001:
            best_model = model
            
    
    print("Final Result\n")
    
    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity Level (%)':<20}")
    print("\n")
    for lmbda, acc, sparsity in results:
        print(f"{lmbda:<10} | {acc:<20.2f} | {sparsity:<20.2f}")
        
    if best_model is not None:
        plot_gate_distribution(best_model)
import torch

# NN is a simple implementation of a feedforward neural network. 
# The input, of size 50, takes in the summed energy values at the corresponding depth layers in the HGCAL detector.
# The two hidden layers are of size 40 and 30 and uses the leaky rectified linear unit as the activation function.
# The single value output is the prodiction of the total energy of the electron as it comes into the detector. 
# The Final softplus function is used as an alternative to ReLU to avoid negative outputs

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.input_size = 50
        self.hidden_size1 = 40
        self.hidden_size2 = 30
        self.output_size = 1
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size1, self.hidden_size2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size2, self.output_size),
            torch.nn.Softplus())

    def forward(self, data):
        return self.network(data)

# The network trains using stochastic gradient descent and MARE (Mean absolute relative error) as its loss function.
# This is to avoid overflow on large training sets

def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()

    return loss, output
    
def MARE(x,y):
    return (sum(abs(x-y)/y))/len(y)
    
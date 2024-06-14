import torch
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import numpy as np

class GAT_4_layer(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads, drop_out, learning_rate, weighting_method):
        super().__init__()
        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.weighting_method = weighting_method
        self.gat1 = GATv2Conv(dim_in, dim_h*8, heads = heads)
        self.gat2 = GATv2Conv(dim_h*8*heads, dim_h*4, heads = heads)
        self.gat3 = GATv2Conv(dim_h*4*heads, dim_h * 2, heads = heads)
        self.gat4 = GATv2Conv(dim_h*2*heads, dim_h, heads = 1)
        self.output = Linear(dim_h, dim_out)
    
    def forward(self, x, edge_index):
        h = F.dropout(x, p=self.drop_out, training = self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.drop_out, training = self.training)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.drop_out, training = self.training)
        h = self.gat3(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.drop_out, training = self.training)
        h = self.gat4(h, edge_index)
        h = F.elu(h)
        embedding = h
        out = self.output(h)
        #out = F.log_softmax(out, dim=1)
        return out, embedding # both the learned class and the embedding are returned.

    
    def fit(self, data, epochs):
        # Compute weights
        y = data.y.detach()
        class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y.numpy())
        class_weight = torch.tensor(class_weight, dtype=torch.float32)
        #print(f"The class weighting is {class_weight}")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
        
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out, _ = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = self.__get_accuracy__(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            # Every now and then print out progress reports
            if epoch % 100 == 0:
                test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
                test_acc = self.__get_accuracy__(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
                f1 = self.__get_f1_score__(data, out)
                print(f"Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Test Loss {test_loss: .2f} | Test Acc: {test_acc*100:.2f}% | f1 {f1: .3f}")
        return f1
    
    def get_name(self):
        return "4 layer GAT"

    def __get_f1_score__(self, data, out) -> float:
        if self.weighting_method == 'Macro':
            f1 = f1_score(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], average='macro')
        else:
            f1 = f1_score(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], average='weighted')
        return f1
    
    def __get_accuracy__(self, y_pred, y_true) -> float:
        return torch.sum(y_pred==y_true) / len(y_true)

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out, _ = self(data.x, data.edge_index)
        acc = self.__get_accuracy__(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc
    
class GAT_3_layer(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads, drop_out, learning_rate, weighting_method):
        super().__init__()
        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.weighting_method = weighting_method
        self.gat1 = GATv2Conv(dim_in, dim_h*4, heads = heads)
        self.gat2 = GATv2Conv(dim_h*4*heads, dim_h*2, heads = heads)
        self.gat3 = GATv2Conv(dim_h*2*heads, dim_h, heads = 1)
        self.output = Linear(dim_h, dim_out)
    
    def forward(self, x, edge_index):
        h = F.dropout(x, p=self.drop_out, training = self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.drop_out, training = self.training)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.drop_out, training = self.training)
        h = self.gat3(h, edge_index)
        h = F.elu(h)
        embedding = h
        out = self.output(h)
        #out = F.log_softmax(out, dim=1)
        return out, embedding # both the learned class and the embedding are returned.
    
    def fit(self, data, epochs):
        # Compute weights
        y = data.y.detach()
        class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y.numpy())
        class_weight = torch.tensor(class_weight, dtype=torch.float32)
        #print(f"The class weighting is {class_weight}")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
        
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out, _ = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = self.__get_accuracy__(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            # Every now and then print out progress reports
            if epoch % 100 == 0:
                test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
                test_acc = self.__get_accuracy__(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
                f1 = self.__get_f1_score__(data, out)
                print(f"Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Test Loss {test_loss: .2f} | Test Acc: {test_acc*100:.2f}% | f1 {f1: .3f}")
        return f1
    
    def get_name(self):
        return "3 layer GAT"

    def __get_f1_score__(self, data, out) -> float:
        if self.weighting_method == 'Macro':
            f1 = f1_score(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], average='macro')
        else:
            f1 = f1_score(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], average='weighted')
        return f1
    
    def __get_accuracy__(self, y_pred, y_true) -> float:
        return torch.sum(y_pred==y_true) / len(y_true)

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out, _ = self(data.x, data.edge_index)
        acc = self.__get_accuracy__(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

class GAT_2_layer(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads, drop_out, learning_rate, weighting_method):
        super().__init__()
        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.weighting_method = weighting_method
        self.gat1 = GATv2Conv(dim_in, dim_h*4, heads = heads)
        self.gat2 = GATv2Conv(dim_h*4*heads, dim_h, heads = 1)
        self.output = Linear(dim_h, dim_out)
    
    def forward(self, x, edge_index):
        h = F.dropout(x, p=self.drop_out, training = self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.drop_out, training = self.training)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        embedding = h
        out = self.output(h)
        #out = F.log_softmax(out, dim=1)
        return out, embedding # both the learned class and the embedding are returned.    
    def fit(self, data, epochs):
        # Compute weights
        y = data.y.detach()
        class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y.numpy())
        class_weight = torch.tensor(class_weight, dtype=torch.float32)
        #print(f"The class weighting is {class_weight}")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
        
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out, _ = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = self.__get_accuracy__(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            # Every now and then print out progress reports
            if epoch % 100 == 0:
                test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
                test_acc = self.__get_accuracy__(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
                f1 = self.__get_f1_score__(data, out)
                print(f"Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Test Loss {test_loss: .2f} | Test Acc: {test_acc*100:.2f}% | f1 {f1: .3f}")
        return f1
    
    def get_name(self):
        return "2 layer GAT"

    def __get_f1_score__(self, data, out) -> float:
        if self.weighting_method == 'Macro':
            f1 = f1_score(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], average='macro')
        else:
            f1 = f1_score(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], average='weighted')
        return f1
    
    def __get_accuracy__(self, y_pred, y_true) -> float:
        return torch.sum(y_pred==y_true) / len(y_true)

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out, _ = self(data.x, data.edge_index)
        acc = self.__get_accuracy__(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

class GAT_1_layer(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads, drop_out, learning_rate, weighting_method):
        super().__init__()
        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.weighting_method = weighting_method
        self.gat1 = GATv2Conv(dim_in, dim_h, heads = heads)
        self.output = Linear(dim_h * heads, dim_out)
    
    def forward(self, x, edge_index):
        h = F.dropout(x, p=self.drop_out, training = self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        embedding = h
        out = self.output(h)
        #out = F.log_softmax(out, dim=1)
        return out, embedding # both the learned class and the embedding are returned.
    
    def fit(self, data, epochs):
        # Compute weights
        y = data.y.detach()
        class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y.numpy())
        class_weight = torch.tensor(class_weight, dtype=torch.float32)
        #print(f"The class weighting is {class_weight}")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
        
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out, _ = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = self.__get_accuracy__(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            # Every now and then print out progress reports
            if epoch % 100 == 0:
                test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
                test_acc = self.__get_accuracy__(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
                f1 = self.__get_f1_score__(data, out)
                print(f"Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Test Loss {test_loss: .2f} | Test Acc: {test_acc*100:.2f}% | f1 {f1: .3f}")
        return f1
    
    def get_name(self):
        return "1 layer GAT"

    def __get_f1_score__(self, data, out) -> float:
        if self.weighting_method == 'Macro':
            f1 = f1_score(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], average='macro')
        else:
            f1 = f1_score(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], average='weighted')
        return f1
    
    def __get_accuracy__(self, y_pred, y_true) -> float:
        return torch.sum(y_pred==y_true) / len(y_true)

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out, _ = self(data.x, data.edge_index)
        acc = self.__get_accuracy__(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

import numpy as np
import pyvqnet.nn as nn
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import Adam
from pyvqnet.nn.loss import CrossEntropyLoss
import pyqpanda as pq

# --- 1. 量子线路构建函数 ---
def qml_circuit(input_data, weights, qlist, machine):
    """
    构建量子编码与变分线路
    input_data: 经典网络降维后输出的 8 维角度张量
    weights: Ansatz 变分线路的参数 (2层 * 8比特 * 3个旋转门 = 48维)
    """
    prog = pq.QProg()
    
    # 【1】高效量子编码层 (G=8, D=1)
    # 将 8 维特征通过 Ry 门独立作用于 8 个比特
    for i in range(8):
        prog.insert(pq.RY(qlist[i], input_data[i]))
        
    # 【2】固定的 Hardware Efficient Ansatz (HEA)
    weight_idx = 0
    for layer in range(2):
        # 参数化的单比特门层 (Rx, Ry, Rz)
        for i in range(8):
            prog.insert(pq.RX(qlist[i], weights[weight_idx]))
            weight_idx += 1
            prog.insert(pq.RY(qlist[i], weights[weight_idx]))
            weight_idx += 1
            prog.insert(pq.RZ(qlist[i], weights[weight_idx]))
            weight_idx += 1
            
        # 纠缠层 (线性连接 CNOT)
        for i in range(7):
            prog.insert(pq.CNOT(qlist[i], qlist[i+1]))
            
    return prog

# --- 2. 混合量子-经典神经网络模型 ---
class HybridQMLModel(nn.Module):
    def __init__(self):
        super(HybridQMLModel, self).__init__()
        # 经典降维层: 256 (16x16) -> 8
        self.encoder_linear = nn.Linear(256, 8)
        
        # 量子层配置
        # 包含 8 个比特，测量所有比特的 Z 基期望值
        num_qubits = 8
        num_weights = 2 * 8 * 3  # 2层，8比特，每比特3个参数
        self.qlayer = QuantumLayer(
            qml_circuit, 
            weight_shape=(num_weights,), 
            num_qubits=num_qubits,
            observables=[pq.PauliZ(i) for i in range(num_qubits)]
        )
        
        # 经典输出层: 量子层的 8 个测量结果 -> 2 分类
        self.decoder_linear = nn.Linear(8, 2)

    def forward(self, x):
        # 展平图像: [batch_size, 16, 16] -> [batch_size, 256]
        x = nn.flatten(x, 1)
        
        # 经典降维并通过 Tanh 映射到角度区间 [-pi, pi] 附近
        x = self.encoder_linear(x)
        x = nn.tanh(x) * 3.1415926 
        
        # 传入量子层，输出形状为 [batch_size, 8]
        q_out = self.qlayer(x)
        
        # 线性层分类
        out = self.decoder_linear(q_out)
        return out

# --- 3. 训练流程 ---
def train_model():
    # 读取数据
    data_path = 'mnist_train_1000_16_16.npz'
    dataset = np.load(data_path)
    X_train = dataset['data']    # 形状应为 (1000, 16, 16)
    Y_train = dataset['label']   # 形状应为 (1000,)

    model = HybridQMLModel()
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()
    
    epochs = 20
    batch_size = 32
    
    print("开始训练...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # 简单的批次训练循环
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = Y_train[i:i+batch_size]
            
            # 转为 VQNet 的张量格式
            input_tensor = QTensor(batch_x)
            label_tensor = QTensor(batch_y, dtype='int64')
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(input_tensor)
            
            # 计算损失并反向传播
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 统计准确率
            preds = np.argmax(output.to_numpy(), axis=1)
            correct += np.sum(preds == batch_y)
            total += len(batch_y)
            
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/total:.4f} | Acc: {acc*100:.2f}%")
        
    # 保存模型参数
    print("训练完成，保存模型权重至 model.model...")
    nn.save(model.state_dict(), 'model.model')

if __name__ == "__main__":
    train_model()
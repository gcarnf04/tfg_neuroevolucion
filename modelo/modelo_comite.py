import numpy as np

# Reutilizamos la lógica del MLP pero para expertos individuales
class ExpertNet:
    def __init__(self, n_in, ocultas, n_out):
        self.arq = [n_in] + ocultas + [n_out]
        self.pesos, self.sesgos = [], []
        self.total = 0
        for i in range(len(self.arq)-1):
            w_shape = (self.arq[i], self.arq[i+1])
            limite = np.sqrt(2.0 / (self.arq[i] + self.arq[i+1]))
            W = np.random.uniform(-limite, limite, w_shape).astype(np.float32)
            b = np.zeros(self.arq[i+1], dtype=np.float32)
            
            self.pesos.append(W)
            self.sesgos.append(b)
            self.total += (self.arq[i] * self.arq[i+1]) + self.arq[i+1]

    def forward(self, X):
        act = X.astype(np.float32) if X.dtype != np.float32 else X
        for i in range(len(self.pesos)):
            z = np.dot(act, self.pesos[i]) + self.sesgos[i]
            # ALPHA v6.16: LeakyReLU en ocultas, LINEAL en la salida para permitir convicción
            if i < len(self.pesos)-1:
                # LeakyReLU (0.01) para evitar neuronas muertas en la evolución
                act = np.where(z > 0, z, z * 0.01)
            else:
                act = z # Salida lineal
        return act

class ModeloComite:
    def __init__(self, idx_c, idx_m, idx_l, n_out, ocultas=[14]):
        self.idx_c = idx_c
        self.idx_m = idx_m
        self.idx_l = idx_l
        
        self.expertos = [
            ExpertNet(len(idx_c), ocultas, n_out),
            ExpertNet(len(idx_m), ocultas, n_out),
            ExpertNet(len(idx_l), ocultas, n_out)
        ]
        self.total_parametros = sum(e.total for e in self.expertos)

    def set_pesos_aplanados(self, adn):
        idx = 0
        adn = adn.astype(np.float32)
        for e in self.expertos:
            for i in range(len(e.pesos)):
                numw = e.pesos[i].size
                e.pesos[i] = adn[idx:idx+numw].reshape(e.pesos[i].shape)
                idx += numw
                numb = e.sesgos[i].size
                e.sesgos[i] = adn[idx:idx+numb]
                idx += numb
                numb = e.sesgos[i].size # Correcting potential bug if sesgos list changed

    def predecir(self, X):
        # 1. Obtener propuestas lineales de los expertos
        p_c = self.expertos[0].forward(X[:, self.idx_c])
        p_m = self.expertos[1].forward(X[:, self.idx_m])
        p_l = self.expertos[2].forward(X[:, self.idx_l])
        propuestas = (p_c + p_m + p_l) / 3.0

        # 2. ALPHA v6.16: Sincronización con el motor GPU (ReLU + Normalización Lineal)
        propuestas_pos = np.maximum(0, propuestas)
        suma = np.sum(propuestas_pos, axis=1, keepdims=True)
        
        # Normalización evitando división por cero
        pesos = propuestas_pos / (suma + 1e-10)
        
        # Fallback de Seguridad: Si suma es 0, todo al Cash (último activo)
        mask_zero = (suma.flatten() < 1e-9)
        if np.any(mask_zero):
            pesos[mask_zero, :] = 0.0
            pesos[mask_zero, -1] = 1.0
            
        return pesos
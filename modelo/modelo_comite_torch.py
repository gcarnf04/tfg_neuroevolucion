import torch
import torch.nn as nn

class ModeloComiteTorch(nn.Module):
    def __init__(self, idx_c, idx_m, idx_l):
        super().__init__()
        self.idx_c = idx_c
        self.idx_m = idx_m
        self.idx_l = idx_l

    def forward(self, X, pesos_pop):
        """
        X: (Poblacion, Dias, Features) o (Dias, Features)
        pesos_pop: Diccionario de tensores (Poblacion, Capa_W/b)
        Retorna propuestas: (Poblacion, Dias, n_activos+1)
        """
        if X.dim() == 2:
            n_pop = pesos_pop['c_w'][0].shape[0]
            X_pop = X.unsqueeze(0).expand(n_pop, -1, -1)
        else:
            X_pop = X

        def run_expert(data, ws, bs):
            act = data
            for i in range(len(ws)):
                z = torch.bmm(act, ws[i]) + bs[i].unsqueeze(1)
                # ALPHA v6.16: LeakyReLU en ocultas, LINEAL en la salida
                if i < len(ws) - 1:
                    # LeakyReLU para evitar el colapso de neuronas muertas
                    act = torch.nn.functional.leaky_relu(z, negative_slope=0.01)
                else:
                    act = z # Salida cruda (logits)
            return act

        p_c = run_expert(X_pop[:, :, self.idx_c], pesos_pop['c_w'], pesos_pop['c_b'])
        p_m = run_expert(X_pop[:, :, self.idx_m], pesos_pop['m_w'], pesos_pop['m_b'])
        p_l = run_expert(X_pop[:, :, self.idx_l], pesos_pop['l_w'], pesos_pop['l_b'])

        # Promedio simple de los 3 expertos (logits crudos)
        propuestas = (p_c + p_m + p_l) / 3.0
        return propuestas
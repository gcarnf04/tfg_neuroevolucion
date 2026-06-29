import torch

class EvolucionGenetica:
    def __init__(self, idx_c, idx_m, idx_l, n_out, tamano_poblacion, local_device, tasa_mutacion=0.15, fuerza_mutacion=0.4, ratio_elitismo=0.1, ratio_inmigrantes=0.15, ocultas=[14]):
        self.idx_c = idx_c
        self.idx_m = idx_m
        self.idx_l = idx_l
        self.n_out = n_out
        self.tamano_poblacion = tamano_poblacion
        self.tasa_mutacion = tasa_mutacion
        self.fuerza_mutacion = fuerza_mutacion 
        self.local_device = local_device
        self.ocultas = ocultas
        
        self.num_elite = int(tamano_poblacion * ratio_elitismo)
        self.num_inmigrantes = int(tamano_poblacion * ratio_inmigrantes)
        self.num_hijos = tamano_poblacion - self.num_elite - self.num_inmigrantes
        
        # Calcular dinámicamente el número de parámetros (n_params) simulando un individuo
        from modelo.modelo_comite import ModeloComite
        dummy = ModeloComite(self.idx_c, self.idx_m, self.idx_l, self.n_out, ocultas=self.ocultas)
        self.n_params = dummy.total_parametros

    def inicializar_poblacion(self):
        # Población como un único tensor [Pop, n_params]
        return (torch.rand((self.tamano_poblacion, self.n_params), device=self.local_device) * 2.0) - 1.0

    def crear_poblacion_desde_semillas(self, seeds):
        """
        Crea la población inicial de minado clonando y mutando semillas top.
        """
        n_seeds = seeds.shape[0]
        repeats = (self.tamano_poblacion // n_seeds) + 1
        pop = seeds.repeat(repeats, 1)[:self.tamano_poblacion]
        # Mantener las semillas originales en el primer bloque y mutar el resto para diversidad
        if self.tamano_poblacion > n_seeds:
            pop[n_seeds:] = self.mutar(pop[n_seeds:])
        return pop

    def mutar(self, adn_hijos):
        # Operaciones súper masivas 100% nativas en GPU
        num_h = adn_hijos.shape[0]
        mascara_mutacion = torch.rand((num_h, self.n_params), device=self.local_device) < self.tasa_mutacion
        ruido = torch.randn((num_h, self.n_params), device=self.local_device) * self.fuerza_mutacion
        return adn_hijos + (ruido * mascara_mutacion)

    def evolucionar(self, adns_actual, fitness_scores):
        # 1. Empaquetado a GPU (por si la métrica viene como List o Array)
        if not isinstance(fitness_scores, torch.Tensor):
            fitness_scores = torch.tensor(fitness_scores, dtype=torch.float32, device=self.local_device)
            
        indices_ordenados = torch.argsort(fitness_scores, descending=True)
        adns_ordenados = adns_actual[indices_ordenados]
        
        # 2. Elitismo (clonación directa por slicing masivo)
        elite_adns = adns_ordenados[:self.num_elite]
        
        # 3. Selección por Torneo Súper-Vectorizado
        # Creación de cruces y enfrentamientos: H hijos, cada uno con 2 padres, cada padre compite entre k=3 clones aleatorios
        idx_torneo = torch.randint(0, self.tamano_poblacion, size=(self.num_hijos, 2, 3), device=self.local_device)
        fit_torneo = fitness_scores[idx_torneo] 
        mejores_en_torneo = torch.argmax(fit_torneo, dim=2) 
        
        idx_padres = idx_torneo.gather(2, mejores_en_torneo.unsqueeze(-1)).squeeze(-1) 
        padres1_adns = adns_actual[idx_padres[:, 0]] 
        padres2_adns = adns_actual[idx_padres[:, 1]] 
        
        # 4. Cruce Uniforme Masivo (Un tensor booleano filtra qué bits hereda de padre1 ó padre2)
        mascara_cruce = torch.rand((self.num_hijos, self.n_params), device=self.local_device) > 0.5
        hijos_adns = torch.where(mascara_cruce, padres1_adns, padres2_adns)
        
        # 5. Mutación Masiva Integrada
        hijos_adns = self.mutar(hijos_adns)
        
        # 6. Inmigrantes Genéticos Vírgenes
        inmigrantes_adns = (torch.rand((self.num_inmigrantes, self.n_params), device=self.local_device) * 2.0) - 1.0
        
        # Retorna el Nuevo Ensamble unificado en memoria contigua (Host-to-Device Bottleneck Annihilated)
        return torch.cat([elite_adns, hijos_adns, inmigrantes_adns], dim=0)
import heapq
from src.constants.error import ERROR_INCOMPATIBLE_SIZES
from src.models.core.system import System
from src.constants.base import NET_LABEL, STR_ZERO
from src.funcs.base import ABECEDARY
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto
from src.models.base.sia import SIA
from src.constants.base import (
    ACTUAL,
    EFECTO,
    TYPE_TAG,
)
from src.constants.models import (
    GEOMETRIC_ANALYSIS_TAG,
    GEOMETRIC_LABEL,
    GEOMETRIC_STRAREGY_TAG,
)
from src.controllers.manager import Manager
from src.funcs.format import  fmt_kparte_q
from src.middlewares.profile import profiler_manager, profile
from src.models.core.solution import Solution
import numpy as np
import time
from typing import List, Dict, Tuple

from concurrent.futures import ThreadPoolExecutor
import itertools

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)
        self.tabla_transiciones: dict ={}
        self.vertices :set[tuple]
        self.tabla :dict[int, list[tuple[int, int]]] = {}
        self.memoria_particiones: dict[tuple[int, int], tuple[float, float]] = {}
        self.min_emd:int = float("inf")

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray, #! COMENTAR PARA UN SOLO ESTADO INICIAL
    ):
        """ vamos a hacer que vaya desde el estado inicial hasta el final, bit a bit diferente, llenando la tabla primero para distancias hamming 1 hasta n, con n la cantidad de bits que cambian del estado inicial al final. para esto podemos usar una tabla de transiciones, donde cada fila es un estado y cada columna es un bit. la tabla de transiciones se llena con los estados que se pueden alcanzar desde el estado inicial, y luego se va llenando la tabla de distancias hamming. para esto vamos a usar una lista de listas, donde cada lista es una fila de la tabla de transiciones. la primera fila es el estado inicial, y las siguientes filas son los estados alcanzables desde el estado inicial. la última fila es el estado final.
        paso a paso
        1. cargar la matriz, pasar a ncubos
        2. condicionar
        3. obtener los bits que cambian entre el estado inicial y el final
        4. obener vecinos del estado final que van hacia el estado inicial y calcular el costo de la transicion.
        5. para cada vecino, obtener los vecinos que van hacia el estado inicial y calcular el costo de la transicion.
        6. repetir hasta llegar al estado inicial.


        nota: intentar llenar la tabla desde el estado final hacia atras, pues al contrario habra dependencia de los valores de la tabla de los estados que van en camino hacia el estado final
        """
       
        self.calcular_tabla_transiciones(condicion, alcance, mecanismo, tpm)
        presentes_local = np.arange(len(self.sia_subsistema.dims_ncubos),dtype=np.int8)
        futuros_local = np.arange(len(self.sia_subsistema.indices_ncubos),dtype=np.int8)
        memoria_particiones = self.calcular_costos(presentes_local,futuros_local,[])
        return self.find_mip(memoria_particiones) 
        
    def calcular_costos(
        self,
        presentes_local: list[int],
        futuros_local: list[int],
        partes = None
    ):
        if partes is None:
            partes = []

        memoria_particiones = {}
        
        candidatos = self.identificar_particiones_optimas(presentes_local,futuros_local)
        
        grupos_base = []
        # for presentes, futuros in partes:
        #     print(f"partes { partes}") 
        #     presentes_global = self.sia_subsistema.dims_ncubos[presentes]
        #     futuros_global = self.sia_subsistema.indices_ncubos[futuros] 
        #     grupos_base.append([presentes_global,futuros_global])
        for presentes, futuros in partes:

            presentes = np.array(
                list(presentes),
                dtype=np.int8
            )

            futuros = np.array(
                list(futuros),
                dtype=np.int8
            )

            presentes_global = self.sia_subsistema.dims_ncubos[presentes]

            futuros_global = self.sia_subsistema.indices_ncubos[futuros]

            grupos_base.append(
                [
                    presentes_global,
                    futuros_global
                ]
            )   
        for particion in candidatos:
            grupos = grupos_base.copy()
            for presentes, futuros in particion:
                presentes_global = self.sia_subsistema.dims_ncubos[presentes]
                futuros_global = self.sia_subsistema.indices_ncubos[futuros]
                grupos.append([presentes_global,futuros_global ])
            dist = self.sia_subsistema.k_partir(grupos).distribucion_marginal()
            emd = emd_efecto(dist,self.sia_dists_marginales)    

            particion_completa = []
            particion_completa.extend(partes)
            particion_completa.extend(particion)
            key = tuple(
            (
                tuple(presentes.tolist()),
                tuple(futuros.tolist())
            )
            for presentes, futuros in particion_completa
        )
            key = self.normalizar_key(key)
            memoria_particiones[key] = (emd,dist)
        particiones_ordenadas = sorted(memoria_particiones.items(),key=lambda x: x[1][0])
        return dict(particiones_ordenadas[:3])
                 
                
                
    def calcular_tabla_transiciones(self, condicion: str, alcance: str, mecanismo: str, tpm: np.ndarray):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm) #! COMENTAR PARA UN SOLO ESTADO INICIAL
        # self.sia_preparar_subsistema(condicion, alcance, mecanismo) #! DESCOMENTAR PARA UN SOLO ESTADO INICIAL
        self.modo_grande = len(self.sia_subsistema.estado_inicial) >= 20
        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )

        self._flat_data = []
        for idx, ncubo in enumerate(self.sia_subsistema.ncubos):
            # garantías: ncubo.data.shape == (2,2,...,2)
            # np.ravel() lo aplana. El orden ‘C’ equivale 
            # a little-endian si tus tuples están invertidas.
            self._flat_data.append(ncubo.data.ravel())

        self.vertices = set(presente + futuro)
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial
        self.sia_logger.critic("empieza.")
        estado_inicial = self.estado_inicial
        estado_final = self.estado_final
        self.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos: Dict[int, List[List[int]]] = {0: [estado_inicial.tolist()]}
        self.tabla_transiciones[tuple(self.caminos[0][0]),tuple(self.caminos[0][0])] = [0.0 for _ in range(len(self.sia_subsistema.indices_ncubos))]

        if self.modo_grande:
            self.calcular_costos_nivel(estado_final, 1)
        else:
            print("Modo pequeño activado: calculando costos para todos los niveles" )
            for nivel in range(1, len(estado_inicial)+1):
                self.calcular_costos_nivel(estado_final, nivel)
    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))
         
    def find_mip(self,memoria_particiones: dict[tuple[int, int], tuple[float, float]]):
        mip =  min(
            memoria_particiones, key=lambda k: memoria_particiones[k][0]
        )
        fmt_mip = fmt_kparte_q(mip)
        return Solution(
                estrategia= GEOMETRIC_LABEL,
                perdida=memoria_particiones[mip][0],
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=memoria_particiones[mip][1],
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion=fmt_mip
        )
    
    def calcular_costos_nivel(self,estado_final: np.ndarray, nivel):
        n = len(estado_final)      
        visitados:set[tuple] = set()
        self.caminos[nivel] = []
        for estado_anterior in self.caminos[nivel - 1]:
            estado_actual = np.array(estado_anterior)
            for i in range(n):
                if estado_actual[i] != estado_final[i]:
                    nuevo_estado = estado_actual.copy()
                    nuevo_estado[i] = estado_final[i]
                    nuevo_estado_tuple = tuple(nuevo_estado)
                    if nuevo_estado_tuple not in visitados:
                        self.caminos[nivel].append(nuevo_estado.tolist())
                        self.calcular_costo(self.caminos[0][0],nuevo_estado.tolist(),self.idx_ncubos)
                        visitados.add(nuevo_estado_tuple)

    def calcular_costo(self, estado_inicial:tuple, estado_final:tuple, ncubos:list[int]):
        """
            Funcion encargada de calcular el costo de transicion de transicion del estado inicial al estado final
            para las variables futuras definidas en ncubos
            aplica la funcion de costo tx(i,j)= y(|X[i]-X[j]|+ sum(tx(k,j)))
            donde:
                - y es el factor de decrecimiento 1/2^(dh(i,j))
                - dh(i,j) es la distancia hamming entre i y j
                - X[i] es el valor de probabilida de transicion de un estado para cada variable futura
                - sum(tx(i,k)) son todos costos de transicion de los vecinos de j que estan en un 
                  camino optimo desde i
        """
        key = tuple(estado_inicial), tuple(estado_final)
        if key not in self.tabla_transiciones:
            self.tabla_transiciones[key] = [None]*len(self.sia_subsistema.indices_ncubos)
        distancia_hamming = self.hamming(estado_inicial, estado_final)
        factor = 1/(2**distancia_hamming)
        # index_inicial = tuple(np.array(estado_inicial)[::-1])
        # index_final = tuple(np.array(estado_final)[::-1])


        estado_ini_int = int("".join(map(str, estado_inicial[::-1])), 2)
        estado_fin_int = int("".join(map(str, estado_final[::-1])), 2)

        # Con eso, cada flat_data[idx][...] ya te da directamente X[i] o X[j].
        diffs = np.abs(
            np.array([flat[estado_ini_int] for flat in self._flat_data])
        - np.array([flat[estado_fin_int] for flat in self._flat_data])
        )
        self.tabla_transiciones[key] = diffs.tolist()
        # for idx in ncubos:
        #     self.tabla_transiciones[key][idx] = (abs(self.sia_subsistema.ncubos[idx].data[index_inicial]-self.sia_subsistema.ncubos[idx].data[index_final]))
        
        if distancia_hamming > 1:
            for i in range(len(estado_inicial)):
                if estado_inicial[i] != estado_final[i]:
                    nuevo_estado = estado_final.copy()
                    nuevo_estado[i] = estado_inicial[i]
                    nuevo_estado_tuple = tuple(nuevo_estado)
                    temp_key = tuple(estado_inicial), nuevo_estado_tuple
                    for n in ncubos:
                        self.tabla_transiciones[key][n] = self.tabla_transiciones[key][n] + self.tabla_transiciones[temp_key][n]
        tmp =[]
        for i,n in enumerate(self.tabla_transiciones[key]):
            if n is not None:
                tmp.append(factor * n)
            else:
                tmp.append(n)
        self.tabla_transiciones[key] = tmp

            
    def calcular_variables_fijos(self, presentes_locales: list[int]) -> list[int]:
        return [i for i in range(len(self.estado_final)) if i not in presentes_locales]

    def validar_estado(self, variables_fijas: list[int], estado: list[int]):
        return all(
            estado[variable] == self.estado_inicial[variable]
            for variable in variables_fijas
        )

    def identificar_particiones_optimas(self,presentes_local:list[int], futuros_local:list[int]):
    # def identificar_particiones_optimas(self):
        """
        Identifica las particiones óptimas basadas en los costos de transición
        y las distancias Hamming entre los estados.
        """
        variables_fijas = self.calcular_variables_fijos(presentes_local)
        # idx_nivel_cero = 0
        # idx_nivel_cero_2 = 1
        # costo=1e5
        key = tuple(self.caminos[0][0]), tuple(self.estado_final)
        # costos: list = self.tabla_transiciones[key]
        # print(f"costos nivel cero {costos}")
        # for idx, valor in enumerate(costos):
        #     if valor < costo:
        #         costo = valor
        #         idx_nivel_cero = idx
        # presentes_nivel_cero = [i for i in range(len(self.estado_final))]
        # furutros_nivel_cero = [i for i in range(len(self.sia_subsistema.indices_ncubos)) if i != idx_nivel_cero]
        # candidatos = [[presentes_nivel_cero, furutros_nivel_cero]]
        # pares = [(valor, idx) for idx, valor in enumerate(costos)]
        # menores = heapq.nsmallest(len(self.estado_inicial), pares, key=lambda x: x[0])
        candidatos = []
        futuros_a_probar = futuros_local
        es_mayor_20 = len(self.estado_final) >= 20

        for idx in futuros_a_probar:
            presentes_1 = np.array(presentes_local, dtype=np.int8)
            futuros_1 = np.array([f for f in futuros_local if f != idx], dtype=np.int8)
            presentes_2 = np.array([], dtype=np.int8)
            futuros_2 = np.array([idx], dtype=np.int8)

            candidatos.append([
                [presentes_1, futuros_1],
                [presentes_2, futuros_2]
            ])
        if self.modo_grande: 
            print("Modo grande activado:"
                " usando solo los candidatos con un futuro"
            )
            return candidatos     
        else:
            print (f"modo pequeño con {len(self.estado_final)} tamaño")
        # _, idx_nivel_cero_1 = dos_menores[0]
        # _, idx_nivel_cero_2 = dos_menores[1]n_vars
        # print(idx_nivel_cero_1, idx_nivel_cero_2)
        # presentes_1 = [i for i in range(n_vars)]
        # futuros_1  = [i for i in range(n_vars) if i != idx_nivel_cero_1]
        # presentes_2 = [i for i in range(n_vars)]
        # futuros_2  = [i for i in range(n_vars) if i != idx_nivel_cero_2]
        # candidatos = [
        #     [presentes_1, futuros_1],
        #     [presentes_2, futuros_2]
        # ]
        # print(f"candidatos nivel cero {candidatos}")
        es_par = len(self.caminos) % 2 == 0
        if es_par:
            mitad = len(self.caminos) // 2
        else:
            mitad = (len(self.caminos) // 2) + 1
        for nivel in range(1,mitad):
            # candidato_nivel = self.caminos[nivel][0]
            costo_candidato_nivel = 1e5
            presentes_nivel = np.array([], dtype=np.int8)
            futuros_nivel = np.array([], dtype=np.int8)
            
            for estado in self.caminos[nivel]:
                if  self.validar_estado(variables_fijas, estado):
                    costo_candidato = 0
                    presentes = np.array([], dtype=np.int8)
                    futuros = np.array([], dtype=np.int8)
                    actual = self.tabla_transiciones.get((tuple(self.caminos[0][0]), tuple(estado)), None)
                   
                    estado_complementario = self.estado_complementario(estado, presentes_local)
                    complementario = self.tabla_transiciones.get((tuple(self.caminos[0][0]), tuple(estado_complementario)), None)
                    for idx in presentes_local:
                        if estado[idx] == self.caminos[0][0][idx]:
                            presentes = np.append(presentes, idx)

                    for futuro in futuros_local:
                        if actual[futuro] <= complementario[futuro]:
                            futuros = np.append(futuros, futuro)
                            costo_candidato += actual[futuro]
                        else:
                            costo_candidato += complementario[futuro]
                    if costo_candidato < costo_candidato_nivel:
                        # candidato_nivel = candidato
                        costo_candidato_nivel = costo_candidato
                        presentes_nivel = presentes
                        futuros_nivel = futuros
                        presentes_complemento = np.array([p for p in presentes_local if p not in presentes_nivel], dtype=np.int8)
                        futuros_complemento = np.array([f for f in futuros_local if f not in futuros_nivel], dtype=np.int8)
            candidatos.append([[presentes_nivel, futuros_nivel],[presentes_complemento, futuros_complemento]])
        return candidatos
    
    def estado_complementario(self, estado: list[int], presentes_local: list[int]) -> list[int]:

        complemento = estado.copy()

        for idx in presentes_local:
            complemento[idx] = 1 - complemento[idx]

        return complemento 

    def ordenar_candidatos(self, candidatos):
        pass
    
    def hamming(self,a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))
    def key_a_particion(self, key):

        return [
            [
                np.array(presentes, dtype=np.int8),
                np.array(futuros, dtype=np.int8)
            ]
            for presentes, futuros in key
        ]

        
    def normalizar_key(self, key):

        return tuple(
            sorted(
                key,
                key=lambda bloque: (
                    bloque[0],
                    bloque[1]
                )
            )
        )
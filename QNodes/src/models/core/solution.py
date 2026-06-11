# solution.py
from colorama import init, Fore, Style
import numpy as np
from typing import Optional

from src.constants.models import PYPHI_LABEL
from src.constants.base import FLOAT_ZERO, INT_ZERO, WHITESPACE
from src.models.base.application import aplicacion

# Iniciar colorama
init()


class Solution:
    """
    Clase Solution para representar y visualizar soluciones del sistema IIT.

    Esta clase maneja la representación y visualización de las soluciones
    encontradas durante el análisis de Integrated Information Theory (IIT). Proporciona
    funcionalidades para mostrar distribuciones de probabilidad, particiones del sistema
    y el valor φ (phi|small phi) asociado a la pérdida de información.

    Args:
    ----
        estrategia (str):
            La estrategia utilizada para la resolución del problema.

        perdida (float):
            El valor φ que representa la pérdida de información en el sistema.
            Este valor cuantifica la diferencia entre la distribución del subsistema
            y la distribución de la partición.

        distribucion_subsistema (np.ndarray):
            Array que representa la distribución de probabilidad del subsistema completo.
            Contiene las probabilidades de cada estado posible en el espacio del subsistema.

        distribucion_particion (np.ndarray):
            Array que representa la distribución de probabilidad de la partición.
            Contiene las probabilidades de cada estado en el espacio de la partición
            que minimiza la información integrada.

        particion (str):
            Representación en formato string de la mejor partición encontrada.
            Utiliza notación matemática para mostrar la estructura de la partición.

    Attributes:
    ----------
        perdida (float):
            El valor φ de la solución.

        distribucion_subsistema (np.ndarray):
            La distribución de probabilidad del subsistema.

        distribucion_particion (np.ndarray):
            La distribución de probabilidad de la partición.

        particion (str):
            La representación de la mejor partición.
    """

    def __init__(
        self,
        estrategia: str,
        perdida: float,
        distribucion_subsistema: np.ndarray,
        distribucion_particion: np.ndarray,
        particion: str,
        tiempo_total: float = FLOAT_ZERO,
        quiere_hablar: bool = False,  # Se mantiene por compatibilidad de firma
        voz: Optional[str] = None,     # Se mantiene por compatibilidad de firma
    ) -> None:
        
        """
        Inicializa una nueva instancia de Solution.
        """
        self.estrategia = estrategia
        self.perdida = perdida
        self.distribucion_subsistema = distribucion_subsistema
        self.distribucion_particion = distribucion_particion
        self.particion = particion
        self.tiempo_ejecucion = tiempo_total

    def __str__(self) -> str:
        """
        Genera una representación en string formateada y coloreadita de la solución.

        Returns:
        -------
            str:
                Representación visual de la solución.
        """
        espaciado = 64
        bilinea = "═" * espaciado
        trilinea = "≡" * espaciado

        def formatear_distribucion(
            distribucion: np.ndarray,
            evitar_desbordamiento=True,
        ):
            rango = distribucion.size
            mensaje_desborde = ""
            if evitar_desbordamiento:
                LIMITE = espaciado
                excedente = rango - LIMITE
                if excedente > FLOAT_ZERO:
                    mensaje_desborde = f" {excedente} valores más.."
                    rango = LIMITE

            datos = WHITESPACE.join(
                f"{Fore.WHITE}{distribucion[idx]:.4f}"
                if distribucion[idx] > FLOAT_ZERO
                else f"{Fore.LIGHTBLACK_EX}0.    "
                for idx in range(rango)
            )
            return f"[ {datos}{mensaje_desborde} {Fore.WHITE}]"

        es_pyphi = self.estrategia == PYPHI_LABEL
        tipo_distribucion = "tensorial" if es_pyphi else "marginal"

        tiempo_hrs, tiempo_min, tiempo_seg = (
            f"{self.tiempo_ejecucion / 3600:.2f}",
            f"{self.tiempo_ejecucion / 60:.1f}",
            f"{self.tiempo_ejecucion:.4f}",
        )
        return f"""{Fore.CYAN}{bilinea}

{Fore.RED}{self.estrategia} fue la estrategia de solucion.

{Fore.BLUE}Distancia métrica utilizada:
{Fore.WHITE}{aplicacion.distancia_metrica}
{Fore.BLUE}Notación utilizada en indexación:
{Fore.WHITE}{aplicacion.notacion_indexado}

{Fore.YELLOW}Distribucion {tipo_distribucion} del Subsistema:
{Style.RESET_ALL}{formatear_distribucion(self.distribucion_subsistema)}
{Fore.YELLOW}Distribucion {tipo_distribucion} de la Partición:
{Style.RESET_ALL}{formatear_distribucion(self.distribucion_particion)}

{Fore.YELLOW}Mejor Bi-Partición:
{Fore.MAGENTA}{self.particion}
{Fore.GREEN}Perdida mínima ( φ ) = {self.perdida:.4f}

{Fore.BLUE}Tiempos de ejecución:
{Fore.WHITE}Horas: {tiempo_hrs} = Minutos: {tiempo_min} = Segundos: {tiempo_seg}

{Fore.CYAN}{trilinea}{Style.RESET_ALL}"""

    def __repr__(self) -> str:
        """
        Implementa la representación oficial de la clase Solution.
        """
        return self.__str__()
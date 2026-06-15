from src.controllers.manager import Manager

# 👇 Importación de estrategias 👇 #
from src.strategies.kqnodes.kqnodes import KQNodes


def iniciar():
    """Punto de entrada"""

# ABCDEFGHIJ #
    estado_inicial = "1000000000"
    condiciones =    "1111111111"
    alcance =        "1101101101"
    mecanismo =      "1101101101"

    gestor_redes = Manager(estado_inicial)
    gestor_redes.generar_red(11)
    mpt = gestor_redes.cargar_red()
    k=2

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_bf = KQNodes(mpt)

    sia_cero = analizador_bf.aplicar_estrategia_k(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
        k
    )
    print(sia_cero)
    
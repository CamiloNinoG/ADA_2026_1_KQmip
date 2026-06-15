from src.controllers.manager import Manager

# 👇 Importación de estrategias 👇 #
from src.strategies.force import BruteForce


def iniciar():
    """Punto de entrada"""

# ABCDEFGHIJ #
    estado_inicial = "1000000000"
    condiciones =    "1111111111"
    alcance =        "1101101101"
    mecanismo =      "1101101101"

    gestor_redes = Manager(estado_inicial)
    # gestor_redes.generar_red(11)
    mpt = gestor_redes.cargar_red()

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_bf = BruteForce(mpt)

    sia_cero = analizador_bf.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_cero)
    
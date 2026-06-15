from src.funcs.base import ABECEDARY, LOWER_ABECEDARY
from src.constants.base import VOID_STR


def fmt_biparticion(
    parte_uno: list[tuple[int, ...], tuple[int, ...]],
    parte_dos: list[tuple[int, ...], tuple[int, ...]],
) -> str:
    # Extraer mecanismo y purview de cada parte
    mech_p, pur_p = parte_uno
    mech_d, purv_d = parte_dos

    # Convertir índices a letras o símbolo vacío si no hay elementos
    purv_prim = ",".join(ABECEDARY[j] for j in pur_p) if pur_p else VOID_STR
    mech_prim = ",".join(LOWER_ABECEDARY[i] for i in mech_p) if mech_p else VOID_STR

    purv_dual = ",".join(ABECEDARY[i] for i in purv_d) if purv_d else VOID_STR
    mech_dual = ",".join(LOWER_ABECEDARY[j] for j in mech_d) if mech_d else VOID_STR

    width_prim = max(len(purv_prim), len(mech_prim)) + 2
    width_dual = max(len(purv_dual), len(mech_dual)) + 2

    return (
        f"|{purv_prim:^{width_prim}}||{purv_dual:^{width_dual}}|\n"
        f"|{mech_prim:^{width_prim}}||{mech_dual:^{width_dual}}|\n"
    )


def fmt_kparte_q(
    partes: list[list[tuple[int, int]]],
    to_sort: bool = True,
) -> str:

    partes_formateadas = []

    for parte in partes:

        top, bottom = fmt_parte_q(
            parte,
            to_sort
        )

        partes_formateadas.append(
            f"⎛ {top} ⎞\n⎝ {bottom} ⎠"
        )

    return " ⊗ ".join(partes_formateadas)


def fmt_parte_q(parte, to_sort: bool = True):
    presentes, futuros = parte

    if to_sort:
        presentes = sorted(presentes)
        futuros = sorted(futuros)

    str_pres = ",".join(LOWER_ABECEDARY[i] for i in presentes) if presentes else VOID_STR
    str_fut = ",".join(ABECEDARY[i] for i in futuros) if futuros else VOID_STR

    width = max(len(str_pres), len(str_fut)) + 2

    return (
        f"|{str_fut:^{width}}|",
        f"|{str_pres:^{width}}|"
    )
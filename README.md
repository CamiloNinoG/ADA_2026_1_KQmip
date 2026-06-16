# Requisitos e Instalación Básica

El proyecto utiliza `uv` como gestor rápido de dependencias y entornos virtuales de Python.

## Instalar uv

Si no lo tienes instalado en tu sistema operativo:

```bash
pip install uv
```

## Sincronizar dependencias globales del proyecto

Desde la raíz `ADA_2026_1_KQmip/`, inicializa y sincroniza el entorno virtual de trabajo:

```bash
uv sync
```

---

# 1) Ejecución desde la Raíz (Interfaz Unificada)

El directorio raíz dispone de un script interactivo central que simplifica la ejecución de cualquiera de los dos motores sin necesidad de cambiar de directorios.

## Ejecución interactiva

```bash
uv run main.py
```

## ¿Qué hace?

* Muestra un menú interactivo en consola para seleccionar la estrategia de análisis:

  * `KQNodes`
  * `KGeoMIP`

* Permite:

  * cargar una red desde los conjuntos de pruebas,
  * configurar el estado inicial,
  * definir máscaras de alcance y mecanismo,
  * establecer el número de bloques `k` deseados.

* Ejecuta el solver seleccionado e imprime la solución óptima formateada directamente en consola.

---

# 2) Ejecutar KQNodes (Subproyecto Clásico)

Para ejecutar pruebas individuales o por lotes específicas del motor submodular.

## Dependencias locales

```bash
cd KQNodes
uv sync
```

## Ejecución de Scripts

### Ejecución Interactiva / Caso de Prueba

```bash
uv run exec.py
```

### Ejecución Rápida por CLI (Usuario)

```bash
uv run exec_user.py --estado "1000" --condiciones "1111" --alcance "1111" --mecanismo "1111" --k 3
```

## Pruebas Automatizadas de Correctitud y Estrés

```bash
uv run pytest tests/test_bruteforce_k3.py
```

```bash
uv run pytest tests/test_n25_scalability.py
```

---

# 3) Ejecutar KGeoMIP (Subproyecto Geométrico)

Para ejecutar el procesamiento basado en costos de transición y programación dinámica en el hipercubo.

## Dependencias locales

```bash
cd KGeoMIP/src/Method2_Dynamic_Programming_Reformulation
uv sync
```

## Ejecución de Scripts

### Procesamiento por Lotes (Excel)

```bash
uv run exec.py
```

### Entrada

Lee por defecto el archivo:

```text
KGeoMIP/results/DatosPruebas2026_1.xlsx
```

* Hoja índice `8`
* Columnas:

  * `C` → alcance
  * `D` → mecanismo

### Salida

Escribe los resultados directamente en el mismo archivo Excel en las columnas designadas según el valor de `k`.

### Ejecución Rápida por CLI (Usuario)

```bash
uv run exec_user.py --estado "1000" --condiciones "1111" --alcance "1111" --mecanismo "1111" --k 3 --tamano 10
```

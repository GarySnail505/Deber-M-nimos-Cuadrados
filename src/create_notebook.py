import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Ejercicio 3: Relación entre Puntuación ACT y Promedio de Puntos\n",
    "\n",
    "La siguiente tabla muestra los promedios de puntos del colegio de 20 especialistas en matemáticas y ciencias computacionales, junto con las calificaciones que recibieron estos estudiantes en la parte de matemáticas de la prueba ACT (Programa de Pruebas de Colegios Americanos) mientras estaban en secundaria. Grafique estos datos y encuentre la ecuación de la recta por mínimos cuadrados para estos datos."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Datos del problema"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import stats\n",
    "\n",
    "# Datos de la tabla\n",
    "# Puntuación ACT (x)\n",
    "x = np.array([28, 25, 28, 27, 28, 33, 28, 29, 23, 27, 29, 28, 27, 29, 21, 28, 28, 26, 30, 24])\n",
    "\n",
    "# Promedio de puntos (y)\n",
    "y = np.array([3.84, 3.21, 3.23, 3.63, 3.75, 3.20, 3.41, 3.38, 3.53, 2.03, 3.75, 3.65, 3.87, 3.75, 1.66, 3.12, 2.96, 2.92, 3.10, 2.81])\n",
    "\n",
    "print(\"Puntuación ACT (x):\", x)\n",
    "print(\"\\nPromedio de puntos (y):\", y)\n",
    "print(\"\\nNúmero de datos:\", len(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Método de Mínimos Cuadrados\n",
    "\n",
    "Para encontrar la recta de mejor ajuste $y = mx + b$, necesitamos calcular:\n",
    "\n",
    "$$m = \\frac{n\\sum xy - \\sum x \\sum y}{n\\sum x^2 - (\\sum x)^2}$$\n",
    "\n",
    "$$b = \\frac{\\sum y - m\\sum x}{n}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cálculos necesarios\n",
    "n = len(x)\n",
    "sum_x = np.sum(x)\n",
    "sum_y = np.sum(y)\n",
    "sum_xy = np.sum(x * y)\n",
    "sum_x2 = np.sum(x**2)\n",
    "\n",
    "print(\"n =\", n)\n",
    "print(\"Σx =\", sum_x)\n",
    "print(\"Σy =\", sum_y)\n",
    "print(\"Σxy =\", sum_xy)\n",
    "print(\"Σx² =\", sum_x2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cálculo de la pendiente (m) y la ordenada al origen (b)\n",
    "m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)\n",
    "b = (sum_y - m * sum_x) / n\n",
    "\n",
    "print(f\"Pendiente (m) = {m:.6f}\")\n",
    "print(f\"Ordenada al origen (b) = {b:.6f}\")\n",
    "print(f\"\\nEcuación de la recta: y = {m:.6f}x + {b:.6f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Verificación con NumPy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verificación usando numpy.polyfit\n",
    "coeficientes = np.polyfit(x, y, 1)\n",
    "m_numpy = coeficientes[0]\n",
    "b_numpy = coeficientes[1]\n",
    "\n",
    "print(\"Verificación con NumPy:\")\n",
    "print(f\"Pendiente (m) = {m_numpy:.6f}\")\n",
    "print(f\"Ordenada al origen (b) = {b_numpy:.6f}\")\n",
    "print(f\"\\nEcuación de la recta: y = {m_numpy:.6f}x + {b_numpy:.6f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Coeficiente de Correlación\n",
    "\n",
    "El coeficiente de correlación $r$ nos indica qué tan bien se ajustan los datos a la recta."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Coeficiente de correlación\n",
    "r, p_value = stats.pearsonr(x, y)\n",
    "r2 = r**2\n",
    "\n",
    "print(f\"Coeficiente de correlación (r) = {r:.6f}\")\n",
    "print(f\"Coeficiente de determinación (R²) = {r2:.6f}\")\n",
    "print(f\"\\nInterpretación: El {r2*100:.2f}% de la variabilidad en el promedio de puntos\")\n",
    "print(f\"se puede explicar por la relación lineal con la puntuación ACT.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Gráfica de los datos y la recta de ajuste"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Crear la gráfica\n",
    "plt.figure(figsize=(12, 8))\n",
    "\n",
    "# Graficar los puntos originales\n",
    "plt.scatter(x, y, color='blue', s=100, alpha=0.6, edgecolors='black', linewidths=1.5, label='Datos observados')\n",
    "\n",
    "# Generar puntos para la recta de ajuste\n",
    "x_line = np.linspace(min(x) - 1, max(x) + 1, 100)\n",
    "y_line = m * x_line + b\n",
    "\n",
    "# Graficar la recta de mínimos cuadrados\n",
    "plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Recta de ajuste: y = {m:.4f}x + {b:.4f}')\n",
    "\n",
    "# Configuración de la gráfica\n",
    "plt.xlabel('Puntuación ACT (Matemáticas)', fontsize=12, fontweight='bold')\n",
    "plt.ylabel('Promedio de Puntos del Colegio', fontsize=12, fontweight='bold')\n",
    "plt.title('Relación entre Puntuación ACT y Promedio de Puntos\\n(20 Especialistas en Matemáticas y Ciencias Computacionales)', \n",
    "          fontsize=14, fontweight='bold', pad=20)\n",
    "plt.legend(fontsize=10, loc='best')\n",
    "plt.grid(True, alpha=0.3, linestyle='--')\n",
    "\n",
    "# Agregar texto con información adicional\n",
    "textstr = f'R² = {r2:.4f}\\nn = {n}'\n",
    "plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, \n",
    "         fontsize=10, verticalalignment='top',\n",
    "         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Análisis de Residuos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calcular los valores predichos y los residuos\n",
    "y_pred = m * x + b\n",
    "residuos = y - y_pred\n",
    "\n",
    "# Estadísticas de los residuos\n",
    "print(\"Análisis de Residuos:\")\n",
    "print(f\"Media de residuos: {np.mean(residuos):.6f}\")\n",
    "print(f\"Desviación estándar de residuos: {np.std(residuos):.6f}\")\n",
    "print(f\"Residuo máximo: {np.max(np.abs(residuos)):.6f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Gráfica de residuos\n",
    "plt.figure(figsize=(12, 5))\n",
    "\n",
    "# Subplot 1: Residuos vs valores predichos\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.scatter(y_pred, residuos, color='purple', s=80, alpha=0.6, edgecolors='black')\n",
    "plt.axhline(y=0, color='red', linestyle='--', linewidth=2)\n",
    "plt.xlabel('Valores Predichos', fontsize=11, fontweight='bold')\n",
    "plt.ylabel('Residuos', fontsize=11, fontweight='bold')\n",
    "plt.title('Residuos vs Valores Predichos', fontsize=12, fontweight='bold')\n",
    "plt.grid(True, alpha=0.3)\n",
    "\n",
    "# Subplot 2: Residuos vs Puntuación ACT\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.scatter(x, residuos, color='green', s=80, alpha=0.6, edgecolors='black')\n",
    "plt.axhline(y=0, color='red', linestyle='--', linewidth=2)\n",
    "plt.xlabel('Puntuación ACT', fontsize=11, fontweight='bold')\n",
    "plt.ylabel('Residuos', fontsize=11, fontweight='bold')\n",
    "plt.title('Residuos vs Puntuación ACT', fontsize=12, fontweight='bold')\n",
    "plt.grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Predicciones con el modelo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Hacer predicciones para diferentes puntuaciones ACT\n",
    "puntuaciones_test = [20, 25, 28, 30, 35]\n",
    "\n",
    "print(\"Predicciones del promedio de puntos basado en la puntuación ACT:\\n\")\n",
    "print(\"Puntuación ACT | Promedio Predicho\")\n",
    "print(\"-\" * 40)\n",
    "for punt in puntuaciones_test:\n",
    "    promedio_pred = m * punt + b\n",
    "    print(f\"      {punt}       |      {promedio_pred:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Tabla de Datos Completa con Predicciones"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Crear tabla comparativa\n",
    "import pandas as pd\n",
    "\n",
    "tabla = pd.DataFrame({\n",
    "    'Puntuación ACT': x,\n",
    "    'Promedio Real': y,\n",
    "    'Promedio Predicho': y_pred,\n",
    "    'Residuo': residuos\n",
    "})\n",
    "\n",
    "print(\"\\nTabla Completa de Datos:\")\n",
    "print(tabla.to_string(index=False))\n",
    "print(f\"\\nEcuación final: y = {m:.6f}x + {b:.6f}\")\n",
    "print(f\"Coeficiente de determinación: R² = {r2:.6f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusiones\n",
    "\n",
    "1. La ecuación de la recta por mínimos cuadrados relaciona la puntuación ACT (x) con el promedio de puntos del colegio (y).\n",
    "\n",
    "2. El coeficiente de correlación indica el grado de relación lineal entre las variables.\n",
    "\n",
    "3. El coeficiente de determinación (R²) muestra qué porcentaje de la variabilidad en el promedio de puntos se explica por la puntuación ACT.\n",
    "\n",
    "4. Los residuos nos permiten evaluar la calidad del ajuste y detectar posibles valores atípicos o patrones no lineales."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(r'c:\Metodos Numericos\Deber Mínimos Cuadrados\DefasGary_Ej3_4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
    
print("Notebook creado exitosamente!")

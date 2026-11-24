# Matemáticas, métricas e incertidumbre

Este documento separa explícitamente tres objetos que no deben confundirse: el endpoint experimental de viabilidad, la predicción de `LN_IC50` y la dinámica poblacional hipotética.

## 1. Normalización por placa

Para cada pozo tratado de una placa:

$$
v = \frac{I - \operatorname{mediana}(B)}
{\operatorname{mediana}(NC\!\!-\!1)-\operatorname{mediana}(B)},
$$

donde $I$ es la intensidad del pozo, $B$ el control blanco sin células y `NC-1` el control de vehículo. Sólo se aceptan placas con rango dinámico positivo. Los valores no se recortan a $[0,1]$: 31.23% quedaron por encima de 1 y 0.027% por debajo de 0, preservando variación experimental y de controles.

## 2. Curva Hill de viabilidad endpoint

La curva acotada de tres parámetros es

$$
V(C)=b+\frac{1-b}{1+(C/EC_{50})^h},
$$

donde $C$ está en µM, $b$ es la asíntota inferior, $EC_{50}>0$ es la concentración de semirrespuesta y $h>0$ el coeficiente de Hill. El ajuste restringe $0\le b\le1.2$, $0.05\le h\le8$ y exige al menos cinco observaciones finitas en cuatro concentraciones.

Se comparan:

- Hill de tres parámetros;
- Hill con $b=0$;
- regresión lineal de viabilidad sobre $\log_{10}(C)$.

La validación retiene concentraciones intercaladas dentro de cada curva real y mantiene los extremos en entrenamiento. Las métricas resultantes miden interpolación del endpoint, no una dinámica temporal.

## 3. Objetivo de aprendizaje automático

El objetivo es

$$
y=\ln(IC_{50,\mu M}),\qquad IC_{50,\mu M}=e^y.
$$

Las predicciones y errores del modelo están en escala logarítmica natural. Un error aditivo $\Delta$ corresponde a un factor multiplicativo $e^\Delta$ en IC50.

### Métricas

Para $n$ observaciones con residuo $e_i=y_i-\hat y_i$:

$$
\operatorname{MAE}=\frac1n\sum_{i=1}^{n}|e_i|,
$$

$$
\operatorname{RMSE}=\sqrt{\frac1n\sum_{i=1}^{n}e_i^2},
$$

$$
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.
$$

También se reportan la correlación de Pearson

$$
r=\frac{\sum_i(y_i-\bar y)(\hat y_i-\overline{\hat y})}
{\sqrt{\sum_i(y_i-\bar y)^2\sum_i(\hat y_i-\overline{\hat y})^2}},
$$

y la correlación de rangos de Spearman $\rho_s$. Pearson cuantifica asociación lineal; Spearman, ordenación monótona. Ninguna sustituye MAE/RMSE ni demuestra calibración.

### Protocolos contra fuga de información

- A: pares célula–compuesto nuevos, pero ambas entidades conocidas.
- B: líneas celulares completamente no vistas.
- C: compuestos/identidades químicas no vistas.
- D: líneas celulares y compuestos no vistos simultáneamente.

Todo imputador, escalador, codificador, selector de variables y PCA se ajusta exclusivamente con el entrenamiento de cada protocolo. Fase 2 agrupa compuestos por InChIKey/ChEMBL verificado para evitar que una misma identidad química aparezca con IDs distintos a ambos lados.

## 4. Incertidumbre

### Intervalos bootstrap

Se re-muestrean observaciones con reemplazo y se usan los cuantiles 2.5% y 97.5% de cada métrica. En las evaluaciones por grupos de Fase 2, el remuestreo conserva la unidad de célula o identidad química cuando corresponde.

### Intervalo conformal marginal

Con residuos absolutos de calibración $r_j=|y_j-\hat y_j|$, el semiancho es el cuantil empírico superior $q_{0.95}$. Para una nueva predicción:

$$
[\hat y-q_{0.95},\;\hat y+q_{0.95}].
$$

La garantía es marginal bajo intercambiabilidad; no garantiza cobertura condicional por célula, compuesto ni bajo cambio de dominio. La cobertura observada fue 95.97%, 85.27%, 86.66% y 77.78% en A–D para el análisis base.

## 5. Modelo poblacional de una población

$$
\frac{dN}{dt}=rN\left(1-\frac{N}{K}\right)-k(C)N,
$$

$$
k(C)=k_{\max}\frac{C^h}{EC_{50}^h+C^h}.
$$

$N$ es la población, $r$ la tasa de proliferación por hora, $K$ la capacidad de carga y $k(C)$ una tasa de efecto por hora. La ODE se integra con LSODA (`rtol=1e-8`, `atol=1e-6`). El `EC50` de esta función de tasa no es automáticamente el `EC50` de una curva endpoint; igualarlos requiere una hipótesis adicional.

## 6. Modelo sensible–resistente

Sea $S$ la población sensible, $R$ la resistente, $K$ la capacidad compartida y $\mu$ una transición unidireccional:

$$
\frac{dS}{dt}=r_S S\left(1-\frac{S+R}{K}\right)-k_S(C)S-\mu S,
$$

$$
\frac{dR}{dt}=r_R R\left(1-\frac{S+R}{K}\right)-k_R(C)R+\mu S.
$$

Cada subpoblación tiene sus propios $r$, $k_{\max}$ y $EC_{50}$. Actualmente estas simulaciones son hipotéticas o parcialmente calibradas; no son pronósticos clínicos.

## 7. Sensibilidad global

Los parámetros no identificados se muestrean uniformemente en rangos declarados. Para una población se reportan bandas de cuantiles 2.5%, 50% y 97.5% de $N(t)$. En el modelo resistente se calculan población final, fracción resistente final y correlaciones de rango entre parámetros y salidas. Estas bandas reflejan los rangos elegidos, no una distribución posterior bayesiana.

## 8. Validación temporal

El crecimiento sin tratamiento usa la solución logística

$$
N(t)=\frac{K}{1+\left(\frac{K-N_0}{N_0}\right)e^{-rt}}.
$$

Se ajusta con el 70% temporal inicial y se evalúa en el 30% final. La métrica normalizada es

$$
\operatorname{NRMSE}=\frac{\operatorname{RMSE}}{\max(N_{test})-\min(N_{test})}.
$$

La mediana observada fue 0.669, insuficiente para declarar validado el crecimiento logístico fuera de muestra. Los datos de doxorrubicina contienen pulsos, dosis seriales, retardos y recuperación que la ODE constante actual no representa.

## 9. Reglas de interpretación

- `AUC` es área fraccional de la curva de viabilidad en el rango ensayado, no eficacia clínica.
- CellTiter-Glo mide señal endpoint; una reducción puede reflejar proliferación, viabilidad u otros procesos.
- Una buena interpolación de dosis no valida una ODE temporal.
- Importancia predictiva no equivale a causalidad ni a mecanismo.
- Los parámetros no identificados deben etiquetarse como hipotéticos.


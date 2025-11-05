# Assignment 2 — Reporte de Resultados  
**Integrantes:** 
- Joel Mateo Manrique Velasquez
- Luis Enrique Perez Ramos
- Ginno Jacinto Castro  



---

## Parte 1 · Eigenfaces (PCA) para reconocimiento facial

Trabajamos con 540 imágenes de entrenamiento y 100 de prueba (50×50 píxeles; matrices 540×2500 y 100×2500). Calculamos la cara promedio y centramos ambos conjuntos restándole esa media. A partir de X^T X obtuvimos las eigenfaces y proyectamos cada imagen en las primeras r componentes para construir las características; luego clasificamos con regresión logística sin intercepto.

En clasificación, con r=10 logramos accuracy = 0.7600. La curva *Precisión vs r* crece con rapidez y se estabiliza entre r ≈ 60–80, donde la exactitud se mantiene alrededor de 0.91–0.93; a partir de ahí el beneficio adicional es mínimo e incluso puede decrecer levemente.

En reconstrucción de bajo rango, el error ||X - X'||_F disminuye de forma monótona al aumentar r: de ~101 con r=1, ~57 con r=10 hasta ~11 con r=200.

Conclusión: Para reconocimiento recomendamos r ≈ 60–80 (buen equilibrio entre desempeño y dimensión). Si se prioriza reconstrucción visual, conviene usar r más altos.

Figuras generadas:
- Cara promedio
- Precisión de clasificación vs r
- Error de reconstrucción (Frobenius) vs r

---

## Parte 2 · CNN en MNIST

Usamos MNIST (28×28, 10 clases) con `ToTensor()` (escala [0,1]). Entrenamos una CNN tipo LeNet con dos bloques *Conv–ReLU–MaxPool* seguidos de capas densas. Configuración: 5 épocas, CrossEntropy, Adam (LR=1e−3), dispositivo CPU.

El entrenamiento fue estable: la pérdida de prueba bajó de 0.0665 a 0.0306 en las primeras 4 épocas y subió levemente a 0.0325 en la 5 (indicio de sobreajuste leve). La exactitud de prueba avanzó 97.88 → 98.50 → 98.75 → 98.97 → 99.05%, alcanzando 99.05% al final.

**Conclusión:** Una CNN sencilla alcanza ~99% en MNIST con convergencia rápida. El pequeño sobreajuste observado al final podría mitigarse con BatchNorm/Dropout o early stopping.

Figura generada:
- Curvas de convergencia (pérdida de entrenamiento vs pérdida de prueba)

---

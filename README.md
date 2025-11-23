# Proyecto de Clasificación de Radiografías de Tórax

Este proyecto aborda la clasificación de radiografías de tórax en tres clases: **NORMAL**, **BACTERIA** y **VIRUS**, utilizando técnicas de deep learning con PyTorch.

## Fases del trabajo

1. **Pruebas iniciales (experimentos.py)**  
   Se realizaron múltiples combinaciones de arquitecturas, optimizadores, valores de dropout, weight decay y data augmentation.  
   El objetivo fue comparar rendimiento y estabilidad de distintos enfoques de entrenamiento de forma sistemática.

2. **Red convolutiva con ResNet18**  
   Se implementó una red basada en ResNet18 con fine-tuning parcial.  
   Este modelo es más rápido de entrenar y ofrece resultados sólidos, siendo adecuado para iteraciones rápidas y validación de la metodología.

3. **Red convolutiva con ResNet34 y validación K-Folds**  
   Se construyó una segunda red con ResNet34, aplicando validación cruzada K-Folds para evaluar mejor la capacidad de generalización.  
   Aunque el entrenamiento es más lento debido a la mayor profundidad de la arquitectura, los resultados obtenidos fueron superiores en precisión y consistencia.

## Conclusiones 
- ResNet18 es eficiente y rápida, útil para prototipado.  
- ResNet34, aunque más costosa en tiempo, logra mejores resultados finales, confirmando que una arquitectura más profunda y validación robusta aportan mayor fiabilidad en la clasificación.

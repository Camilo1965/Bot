# ROADMAP DE MEJORAS (Actualizado v2)

## 1. RENDIMIENTO & FIXES (Urgente)
- **Gemini Async:** Hacer que la IA sea asíncrona. Actualmente, cuando Gemini analiza noticias, el bot se "congela" 3-5 segundos, causando los mensajes de "feed appears stale".
- **Optimización de Memoria:** Limpiar buffers antiguos para que el bot no se vuelva lento tras días de ejecución en la EC2.

## 2. OPERATIVA: Shorts & Parciales
- **Short Selling:** Ganar dinero cuando el mercado cae. Requiere invertir la lógica de SL/TP y las órdenes en MT5.
- **Salidas Parciales:** Cerrar 50% de la posición al llegar a +1.5% para asegurar ganancias y evitar que los trades ganadores se vuelvan perdedores.

## 3. SEGURIDAD & CONTROL (Urgente)
- **Password Web:** Poner contraseña al Dashboard. Al estar en una IP pública (EC2), tus finanzas están expuestas si no lo protegemos.
- **Botones de Acción:** Añadir botones de "Cerrar Todo" y "Pausar" en la Web y comandos en Telegram.

## 4. FILTRO DE NOTICIAS PROACTIVO
- **Macro-Detector:** Bloqueo preventivo de trading ante eventos (FED, inflación) detectados por palabras clave, antes de que el mercado reaccione.

## 5. DASHBOARD CEO 2.0
- **Equity Curve:** Gráfica visual del crecimiento de tu capital.
- **Profit Factor Real:** Estadísticas profesionales basadas en tu historial real de la base de datos.

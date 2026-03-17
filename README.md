# 📈 ClawdBot - Quantitative Algorithmic Trading System

ClawdBot es un sistema de trading algorítmico de grado institucional diseñado con una arquitectura orientada a eventos (Event-Driven Architecture). Su objetivo es operar en los mercados financieros combinando microestructura de mercado (Order Book y datos Tick), indicadores técnicos estadísticos y análisis de sentimiento impulsado por Inteligencia Artificial (NLP).

## 🏗️ Arquitectura del Sistema

El sistema está construido para minimizar la latencia y procesar grandes volúmenes de datos en tiempo real:

* **Motor Principal:** Python 3.10+ utilizando `asyncio` para concurrencia y manejo de WebSockets.
* **Base de Datos (Series Temporales):** PostgreSQL con la extensión TimescaleDB, optimizado para almacenar millones de velas (OHLCV) y datos de ticks históricos.
* **Caché en Memoria:** Redis, utilizado para almacenar el estado del libro de órdenes (Order Book) en tiempo real y gestionar el bus de eventos de forma ultrarrápida.
* **Inteligencia Artificial:** Módulos de Machine Learning (XGBoost/LightGBM) para datos tabulares y NLP (Modelos LLM/FinBERT) para análisis de sentimiento de noticias financieras.

## 📂 Estructura del Proyecto

El código está modularizado siguiendo los estándares de la industria cuantitativa:

* `/core`: Contiene el bus de eventos base, enrutador de mensajes y clases abstractas.
* `/data_ingestion`: Conexiones asíncronas vía WebSockets a exchanges (APIs) y recolección de noticias.
* `/strategy`: El "cerebro". Ingeniería de características (Feature Engineering), modelos predictivos de Machine Learning y generador de señales.
* `/execution`: Sistema de Gestión de Órdenes (OMS). Lógica para enrutar órdenes (Market/Limit), cálculo de comisiones y control de *slippage*.
* `/risk`: Gestión de capital institucional, dimensionamiento dinámico de posiciones (Criterio de Kelly) y Stop Loss/Take Profit.
* `/utils`: Configuraciones de variables de entorno, sistemas de logging y métricas de rendimiento (Ratio de Sharpe).

## 🚀 Instalación y Despliegue Local

### Requisitos Previos
* [Docker](https://www.docker.com/) y Docker Compose instalados.
* Python 3.10 o superior.

### Paso a Paso

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/TU_USUARIO/clawdbot.git](https://github.com/TU_USUARIO/clawdbot.git)
    cd clawdbot
    ```

2.  **Configurar Variables de Entorno:**
    Copia el archivo de ejemplo y configura tus claves (APIs, contraseñas de BD):
    ```bash
    cp .env.example .env
    ```

3.  **Levantar la Infraestructura (Base de datos y Caché):**
    Utiliza Docker para iniciar TimescaleDB y Redis en segundo plano:
    ```bash
    docker-compose up -d
    ```

4.  **Instalar Dependencias de Python:**
    Se recomienda usar un entorno virtual (`venv` o `conda`):
    ```bash
    pip install -r requirements.txt
    ```

5.  **Ejecutar el Bot:**
    ```bash
    python main.py
    ```

## ⚠️ Descargo de Responsabilidad (Disclaimer)
Este software tiene fines educativos y de investigación cuantitativa. El trading algorítmico en mercados financieros conlleva un alto nivel de riesgo. Los mercados son volátiles y las fallas de software o latencia de red pueden resultar en pérdidas financieras significativas. No opere con dinero que no esté dispuesto a perder.

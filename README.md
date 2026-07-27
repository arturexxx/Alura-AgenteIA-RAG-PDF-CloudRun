<img width="170" height="126" alt="image" src="https://github.com/user-attachments/assets/daaecd47-411c-40e8-b0c0-b6b8890b1c3d" />

# 🦺 RETCC.BOB — Asistente Virtual RETCC con RAG, Triaje y LangGraph

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Compatible-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-5A45FF)](https://www.langchain.com/langgraph)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-8E75B2?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-0467DF)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-FF7C00?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RetccBob** es un asistente virtual especializado en el Registro Nacional de Trabajadores de Construcción Civil (RETCC) del Perú. Utiliza RAG, memoria conversacional, extracción estructurada de tablas y un flujo de triaje con LangGraph para responder consultas basadas exclusivamente en los documentos oficiales cargados.

## 🌐 Aplicación desplegada

La aplicación se encuentra publicada en **Google Cloud Run** y puede probarse desde el siguiente enlace:

### [Abrir RETCC.BOB en producción](https://retcc-bob-671777364299.southamerica-west1.run.app)

```text
https://retcc-bob-671777364299.southamerica-west1.run.app
```

Características del despliegue:

- Servicio: `retcc-bob`
- Plataforma: Google Cloud Run
- Acceso: público
- Escalamiento mínimo: `0` instancias
- Escalamiento máximo: `1` instancia
- Puerto del contenedor: `8080`
- Memoria configurada: `4 GiB`
- CPU: `1`
- Despliegue continuo desde la rama `main`
- Construcción de imagen mediante `Dockerfile`
- Clave de Gemini administrada mediante la variable de entorno `GOOGLE_API_KEY`

> El primer acceso puede demorar algunos segundos cuando Cloud Run inicia una nueva instancia.

### Evidencia de funcionamiento en producción
<img width="1430" height="852" alt="image" src="https://github.com/user-attachments/assets/92094b35-69cb-4c2b-9aa9-f482120abc6c" />

---


### Ejemplos de consultas

```text
¿Cuáles son los requisitos para inscribirme en el RETCC?
¿Cómo puedo renovar mi carné RETCC?
¿Cuál es el costo del duplicado en Lima Metropolitana?
¿Cuál es el horario de atención de las sedes del RETCC en Cusco?
```

El sistema también identifica preguntas fuera del ámbito del RETCC y evita enviarlas innecesariamente al flujo RAG.

---

## 📑 Tabla de contenidos

- [Aplicación desplegada](#-aplicación-desplegada)
- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Flujo de triaje](#-flujo-de-triaje)
- [Tecnologías](#-tecnologías)
- [Documentos utilizados](#-documentos-utilizados)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Prerrequisitos](#-prerrequisitos)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Ejecución](#ejecucion)
- [Uso](#-uso)
- [Pruebas recomendadas](#-pruebas-recomendadas)
- [Contribución](#-contribución)
- [Licencia](#-licencia)
- [Autor](#autor)

---

## ✨ Características

- Respuestas basadas en documentos PDF del RETCC.
- Recuperación aumentada por generación mediante **RAG**.
- Búsqueda vectorial con **FAISS**.
- Embeddings multilingües de Hugging Face.
- Generación de respuestas con **Google Gemini**.
- Triaje inteligente mediante IA.
- Flujo controlado con **LangGraph**.
- Memoria conversacional por sesión.
- Reformulación de preguntas continuadas.
- Extracción especializada de tablas con `pdfplumber`.
- Recuperación diferenciada para reglamentos, requisitos, sedes, horarios, costos y contactos.
- Interfaz web construida con **Gradio**.
- Presentación inmediata de la pregunta del usuario mientras el backend procesa la respuesta.
- Control de preguntas fuera del ámbito del RETCC.
- Diseño responsive con CSS personalizado.

---

## 🧠 Arquitectura

<img width="739" height="579" alt="image" src="https://github.com/user-attachments/assets/b9026d6a-7596-4ab4-94a5-6ee31b33e38f" />


## 🚦 Flujo de triaje

Cada consulta se clasifica antes de acceder al índice vectorial:

| Categoría | Comportamiento |
|---|---|
| `RETCC` | Continúa al flujo RAG. |
| `SALUDO` | Devuelve el mensaje de bienvenida. |
| `FUERA_DE_TEMA` | Informa el alcance especializado del asistente. |
| `AMBIGUA` | Solicita al usuario mayor precisión. |

El historial reciente permite interpretar preguntas continuadas como:

```text
Usuario: ¿Cómo solicito un duplicado?
Usuario: ¿Y cuánto cuesta en Lima?
```

---

## 🛠 Tecnologías

| Tecnología | Uso |
|---|---|
| Python | Lenguaje principal |
| Google Colab | Entorno de desarrollo y ejecución |
| LangChain | Construcción del flujo RAG |
| LangGraph | Orquestación del triaje y nodos |
| Google Gemini | Clasificación y generación de respuestas |
| Hugging Face | Embeddings multilingües |
| FAISS | Base vectorial |
| PyPDFLoader | Lectura de documentos narrativos |
| pdfplumber | Extracción estructurada de tablas |
| Pydantic | Salida estructurada del triaje |
| Gradio | Interfaz web |
| HTML y CSS | Diseño personalizado |
| Docker | Empaquetado de la aplicación |
| Google Cloud Build | Construcción automática de la imagen |
| Google Cloud Run | Publicación y ejecución en producción |
| GitHub Developer Connect | Despliegue continuo desde el repositorio |

### Modelos utilizados

```text
LLM: gemini-2.5-flash
Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

---

## 📚 Documentos utilizados

El proyecto fue desarrollado para trabajar con documentos como:

```text
Reglamento_Incripcion_RETCC.pdf
Requisitos_Inscripcion_Carnet_RETCC_2.pdf
Canales_de_Atencion_Retcc_2.pdf
```

Los archivos PDF que utiliza el agente se encuentran almacenados en la carpeta:
```text
documentos/
```

El documento de canales contiene información tabular sobre regiones, oficinas, servidores responsables, contactos, horarios, páginas oficiales y pagos por duplicado. Cada sede es convertida en un documento estructurado independiente para preservar la relación entre sus columnas.



## 📁 Estructura del proyecto

```text
Alura-AgenteIA-RAG-PDF-CloudRun/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
│
├── documentos/
│   ├── Reglamento_Incripcion_RETCC.pdf
│   ├── Requisitos_Inscripcion_Carnet_RETCC_2.pdf
│   └── Canales_de_Atencion_Retcc_2.pdf
│
├── notebooks/
│   └── Agente_RETCC_Alura_Desafio.ipynb
│
├── retcc_assets/
│   ├── avatar_retcc.png
│   └── casco.svg
│
└── docs/
    ├── retccbob-demo.png
    ├── flujo-langgraph.png
    └── retccbob-cloudrun-produccion.png
```

### Organización lógica del notebook

```text
1. Presentación del proyecto
2. Instalación de dependencias
3. Carga de documentos
4. Imports y configuración
5. Extracción estructurada de tablas
6. Limpieza de documentos narrativos
7. Fragmentación híbrida
8. Embeddings y FAISS
9. Configuración de Gemini
10. Memoria conversacional
11. Prompts
12. Recuperación especializada
13. Triaje con IA
14. Estado y nodos de LangGraph
15. Compilación del grafo
16. Función pública del backend
17. Pruebas
18. Recursos visuales
19. CSS
20. Interfaz Gradio
21. Lanzamiento de la aplicación
```

---

## ✅ Prerrequisitos

### Para Google Colab

- Cuenta de Google.
- Clave API de Google Gemini.
- Acceso a internet.
- Documentos PDF del RETCC.
- Navegador actualizado.

### Para ejecución local

- Python 3.11 o superior.
- Git.
- `pip`.
- Entorno virtual recomendado.
- Clave API de Google Gemini.

---

## 📥 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/arturexxx/Alura-AgenteIA-RAG-PDF-CloudRun.git
cd Alura-AgenteIA-RAG-PDF-CloudRun
```

### 2. Crear un entorno virtual

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux o macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -U \
  langchain \
  langchain-community \
  langchain-core \
  langchain-text-splitters \
  langchain-huggingface \
  langchain-google-genai \
  langgraph \
  sentence-transformers \
  faiss-cpu \
  pypdf \
  pdfplumber \
  python-dotenv \
  gradio
```

En Google Colab, las dependencias se instalan desde una de las primeras celdas del notebook.

---

## 🔐 Configuración

### Google Colab

1. Abre el notebook.
2. Ingresa a **Secretos** en el panel lateral.
3. Crea un secreto llamado:

```text
GOOGLE_API_KEY
```

4. Asigna tu clave de Google Gemini.
5. Habilita el acceso del notebook al secreto.

El código recupera la clave mediante:

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
```
### Variables para ejecución local o en Cloud Run

La aplicación utiliza las siguientes variables de entorno:

```text
GOOGLE_API_KEY
GEMINI_MODEL
```

`GOOGLE_API_KEY` es obligatoria. `GEMINI_MODEL` es opcional y utiliza por defecto:

```text
gemini-2.5-flash
```

La clave nunca debe guardarse en GitHub. En Cloud Run se configura desde **Variables y secretos**.

### Recursos visuales

Crea la carpeta:

```text
retcc_assets/
```

Incluye:

```text
avatar_retcc.png
casco.svg
```

Estos archivos se utilizan para el avatar, el encabezado y el favicon de la aplicación.

---

<a id="ejecucion"></a>
## ▶️ Ejecución

### Google Colab

1. Abre `Agente_RETCC_Alura_Desafio.ipynb`.
2. Ejecuta las celdas en orden.
3. Carga los PDF solicitados.
4. Verifica que se creen los registros estructurados.
5. Ejecuta la construcción del índice FAISS.
6. Ejecuta las pruebas del backend.
7. Ejecuta la celda de Gradio.
8. Abre la URL pública generada.

La aplicación puede iniciarse con:

```python
interfaz_retcc.launch(
    share=True,
    inline=False,
    debug=True,
    show_error=True,
    favicon_path=str(CASCO_RETCC_FAVICON)
)
```

> Los enlaces gratuitos de Gradio son temporales.

### Ejecución local

La versión ejecutable ya se encuentra en `app.py`. Configura primero `GOOGLE_API_KEY` y ejecuta:

```bash
python app.py
```

---

## ☁️ Despliegue en Google Cloud Run

El repositorio incluye los archivos necesarios para publicar la aplicación:

```text
app.py
requirements.txt
Dockerfile
.dockerignore
```

El contenedor inicia Gradio mediante:

```python
interfaz_retcc.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "8080")),
    share=False,
    debug=False
)
```

Flujo de despliegue:

```text
GitHub
   ↓
Developer Connect
   ↓
Cloud Build
   ↓
Artifact Registry
   ↓
Cloud Run
```

Cada actualización enviada a la rama `main` genera automáticamente una nueva compilación y una nueva revisión del servicio.

La revisión desplegada correctamente atiende el `100 %` del tráfico del servicio público:

```text
https://retcc-bob-671777364299.southamerica-west1.run.app
```

---

## 💬 Uso

### Consultas sobre requisitos

```text
¿Cuáles son los requisitos para inscribirme en el RETCC?
```

### Consultas continuadas

```text
¿Cómo solicito un duplicado?
¿Y cuánto cuesta en Lima Metropolitana?
```

### Consultas sobre sedes

```text
¿Cuáles son las sedes del RETCC en Cusco?
¿Cuál es su horario de atención?
```

### Preguntas fuera de tema

```text
¿Quién ganó el mundial?
```

El triaje debe clasificarla como:

```text
FUERA_DE_TEMA
```

---

## 🧪 Pruebas recomendadas

Antes de iniciar Gradio:

```python
probar_backend_retcc("Hola")

probar_backend_retcc(
    "¿Cuánto cuesta el duplicado en Lima Metropolitana?"
)

probar_backend_retcc(
    "¿Cuáles son los requisitos para renovar el carné?"
)

probar_backend_retcc(
    "¿Quién es el campeón del mundial 2026?"
)
```

También puedes verificar directamente la recuperación:

```python
documentos = retriever_tablas.invoke(
    "Lima Metropolitana pago por duplicado código Banco de la Nación"
)

for documento in documentos:
    print(documento.metadata)
    print(documento.page_content)
    print("-" * 80)
```

---

## 🔍 Decisiones técnicas relevantes

### Extracción híbrida

- Los PDF narrativos se fragmentan.
- Cada fila de sede se conserva completa.
- Los documentos tabulares incluyen metadatos de región, oficina y página.
- Las consultas sobre costos, sedes y horarios utilizan un retriever especializado.

### Actualización progresiva de la interfaz

El envío del mensaje se divide en dos eventos:

1. La pregunta del usuario aparece inmediatamente.
2. El backend ejecuta el triaje, FAISS y Gemini.
3. La respuesta se agrega cuando finaliza el procesamiento.

Esto evita que la interfaz parezca bloqueada mientras se genera la respuesta.

---

## 🤝 Contribución

Las contribuciones son bienvenidas.

1. Crea un fork del repositorio.
2. Crea una rama:

```bash
git checkout -b feature/nueva-mejora
```

3. Realiza tus cambios y ejecuta las pruebas.
4. Crea un commit descriptivo:

```bash
git commit -m "feat: agrega nueva mejora al asistente RETCC"
```

5. Publica la rama:

```bash
git push origin feature/nueva-mejora
```

6. Abre un Pull Request.

Para reportar errores o sugerir mejoras, crea un Issue incluyendo descripción, pasos para reproducir, resultado esperado, captura o log y versión del notebook.

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia **MIT**.

Consulta el archivo [`LICENSE`](LICENSE) para conocer los términos completos.


<a id="autor"></a>
## 👨‍💻 Autor

**Arturo Guerrero**

Proyecto desarrollado como parte del desafío de creación de un agente de inteligencia artificial con RAG para documentos PDF.

- GitHub: [arturexxx](https://github.com/arturexxx)
- Repositorio: [Alura-AgenteIA-RAG-PDF](https://github.com/arturexxx/Alura-AgenteIA-RAG-PDF)
- Email: agcguerrero@gmail.com
- Phone: 991454023
---

## ⚠️ Aviso

RetccBob es un proyecto académico y demostrativo. Las respuestas dependen de los documentos cargados y no sustituyen la orientación oficial del Ministerio de Trabajo y Promoción del Empleo ni de las Direcciones o Gerencias Regionales de Trabajo.

Verifica siempre la vigencia de la información en los canales oficiales correspondientes.

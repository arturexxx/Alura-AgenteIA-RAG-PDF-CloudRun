# =========================================================
# RETCCBOB - APLICACIÓN PARA GOOGLE CLOUD RUN
# Generado desde el notebook final del proyecto.
# =========================================================

import html
import os
import re
import traceback
import uuid

from pathlib import Path
from typing import Literal, Optional, TypedDict

import gradio as gr
import pdfplumber
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator


RUTA_BASE = Path(__file__).resolve().parent
RUTA_DOCUMENTOS = RUTA_BASE / "documentos"
RUTA_ASSETS_RETCC = RUTA_BASE / "retcc_assets"

if not RUTA_DOCUMENTOS.exists():
    raise FileNotFoundError(
        f"No existe la carpeta de documentos: {RUTA_DOCUMENTOS}"
    )

lista_pdfs = sorted(
    str(ruta)
    for ruta in RUTA_DOCUMENTOS.glob("*.pdf")
)

if not lista_pdfs:
    raise FileNotFoundError(
        f"No se encontraron archivos PDF dentro de {RUTA_DOCUMENTOS}"
    )

print("PDF encontrados:")
for ruta_pdf in lista_pdfs:
    print(f"- {Path(ruta_pdf).name}")

# =========================================================
# 4. EXTRACCIÓN ESTRUCTURADA DE LA TABLA DE CANALES
# =========================================================

def normalizar_celda(valor) -> str:
    """Limpia una celda extraída mediante pdfplumber."""
    if valor is None:
        return ""

    texto = str(valor)
    texto = texto.replace("\xa0", " ")
    texto = texto.replace("\u200b", "")
    texto = texto.replace("\ufeff", "")

    # Une palabras cortadas por guion y salto de línea.
    texto = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", texto)

    # Los demás saltos se transforman en espacios.
    texto = re.sub(r"\s*\n\s*", " ", texto)

    # Corrige separaciones frecuentes en correos y dominios.
    texto = re.sub(r"\s*@\s*", "@", texto)
    texto = re.sub(r"\s*\.\s*", ".", texto)
    texto = re.sub(
        r"\bg\s*o\s*b\s*\.\s*p\s*e\b",
        "gob.pe",
        texto,
        flags=re.IGNORECASE
    )
    texto = re.sub(
        r"\bg\s*m\s*a\s*i\s*l\s*\.\s*c\s*o\s*m\b",
        "gmail.com",
        texto,
        flags=re.IGNORECASE
    )
    texto = re.sub(
        r"\bh\s*o\s*t\s*m\s*a\s*i\s*l\s*\.\s*c\s*o\s*m\b",
        "hotmail.com",
        texto,
        flags=re.IGNORECASE
    )
    texto = re.sub(r"\bgo\s+b\.pe\b", "gob.pe", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\bco\s+m\b", "com", texto, flags=re.IGNORECASE)
    texto = re.sub(r"[ \t]+", " ", texto)

    return texto.strip()


def es_fila_region(fila: list) -> bool:
    """Identifica la fila separadora que contiene número y región."""
    celdas = [normalizar_celda(celda) for celda in fila]
    texto = " ".join(celda for celda in celdas if celda)

    if not texto:
        return False

    tiene_numero = any(
        re.fullmatch(r"\d{1,2}", celda)
        for celda in celdas
    )
    tiene_horario = bool(re.search(r"\d{1,2}:\d{2}", texto))
    tiene_correo = "@" in texto
    parece_oficina = any(
        palabra in texto.upper()
        for palabra in (
            "DIRECCIÓN REGIONAL",
            "GERENCIA REGIONAL",
            "ZONA DE TRABAJO",
            "OFICINA DE"
        )
    )

    return (
        tiene_numero
        and not tiene_horario
        and not tiene_correo
        and not parece_oficina
    )


def obtener_region_de_fila(fila: list) -> str:
    """Obtiene el nombre de región desde una fila separadora."""
    celdas = [normalizar_celda(celda) for celda in fila]

    candidatos = [
        celda
        for celda in celdas
        if celda
        and not re.fullmatch(r"\d{1,2}", celda)
        and celda.upper() not in {
            "N°",
            "GOBIERNO",
            "REGIONAL/DIRECCIÓN U",
            "OFICINA"
        }
    ]

    return max(candidatos, key=len).strip().upper() if candidatos else ""


def detectar_columnas_tabla(fila: list) -> dict:
    """Reconoce la estructura de columnas de la tabla RETCC."""
    cantidad = len(fila)

    if cantidad == 7:
        return {
            "oficina": 1,
            "aprobador": 2,
            "contacto": 3,
            "horario": 4,
            "web": 5,
            "pago": 6
        }

    if cantidad >= 11:
        return {
            "oficina": 1,
            "aprobador": 4,
            "contacto": 7,
            "horario": 8,
            "web": 9,
            "pago": 10
        }

    return {}


def extraer_registros_tabla(ruta_pdf: str) -> list[Document]:
    """Convierte cada sede de la tabla en un Document independiente."""
    registros = []
    region_actual = ""
    nombre_archivo = Path(ruta_pdf).name

    with pdfplumber.open(ruta_pdf) as archivo_pdf:
        for numero_pagina, pagina in enumerate(
            archivo_pdf.pages,
            start=1
        ):
            for tabla in pagina.extract_tables() or []:
                for fila in tabla or []:
                    if not fila:
                        continue

                    if es_fila_region(fila):
                        nueva_region = obtener_region_de_fila(fila)
                        if nueva_region:
                            region_actual = nueva_region
                        continue

                    columnas = detectar_columnas_tabla(fila)
                    if not columnas:
                        continue

                    oficina = normalizar_celda(fila[columnas["oficina"]])
                    aprobador = normalizar_celda(fila[columnas["aprobador"]])
                    contacto = normalizar_celda(fila[columnas["contacto"]])
                    horario = normalizar_celda(fila[columnas["horario"]])
                    web = normalizar_celda(fila[columnas["web"]])
                    pago = normalizar_celda(fila[columnas["pago"]])

                    if not oficina:
                        continue

                    oficina_mayuscula = oficina.upper()
                    if "GOBIERNO REGIONAL" in oficina_mayuscula:
                        continue
                    if oficina_mayuscula in {
                        "REGIONAL/DIRECCIÓN U OFICINA",
                        "OFICINA"
                    }:
                        continue

                    contenido = f"""
REGIÓN: {region_actual or "No indica"}
SEDE U OFICINA: {oficina}
SERVIDOR PÚBLICO APROBADOR: {aprobador or "No indica"}
CORREO Y/O TELÉFONO: {contacto or "No indica"}
HORARIO DE ATENCIÓN: {horario or "No indica"}
PÁGINA WEB U OTROS: {web or "No indica"}
PAGO POR DUPLICADO: {pago or "No indica"}
""".strip()

                    registros.append(
                        Document(
                            page_content=contenido,
                            metadata={
                                "source": nombre_archivo,
                                "page": numero_pagina - 1,
                                "pagina_real": numero_pagina,
                                "tipo_contenido": "registro_tabla",
                                "region": region_actual,
                                "oficina": oficina
                            }
                        )
                    )

    return registros


# =========================================================
# 5. CARGA Y LIMPIEZA DE DOCUMENTOS
# =========================================================

def limpiar_texto_narrativo(texto: str) -> str:
    """Limpia el texto narrativo sin alterar los registros de tabla."""
    texto = texto.replace("\xa0", " ")
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    texto = re.sub(r"(?<=\w)\n(?=\w)", " ", texto)
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    return texto.strip()


documentos_narrativos = []
documentos_tabla = []

for ruta_pdf in lista_pdfs:
    nombre = Path(ruta_pdf).name
    nombre_normalizado = nombre.lower().replace(" ", "_")

    es_pdf_canales = (
        "canales_de_atencion_retcc" in nombre_normalizado
        or "canalesdeatencionretcc" in nombre_normalizado
    )

    if es_pdf_canales:
        print(f"Extrayendo tabla estructurada: {nombre}")
        registros = extraer_registros_tabla(ruta_pdf)
        documentos_tabla.extend(registros)
        print(f"  Registros encontrados: {len(registros)}")
        continue

    print(f"Cargando texto narrativo: {nombre}")
    for documento in PyPDFLoader(ruta_pdf).load():
        documento.page_content = limpiar_texto_narrativo(
            documento.page_content
        )
        documento.metadata["source"] = nombre
        documento.metadata["tipo_contenido"] = "pagina_pdf"
        documento.metadata["pagina_real"] = (
            documento.metadata.get("page", 0) + 1
        )

        if documento.page_content.strip():
            documentos_narrativos.append(documento)

print("-" * 70)
print(f"Páginas narrativas: {len(documentos_narrativos)}")
print(f"Registros estructurados: {len(documentos_tabla)}")

if any(
    "canales" in Path(pdf).name.lower()
    for pdf in lista_pdfs
):
    assert documentos_tabla, (
        "El PDF de canales fue cargado, pero no se extrajeron registros."
    )


# =========================================================
# 6. FRAGMENTACIÓN HÍBRIDA
# =========================================================

divisor_texto = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    length_function=len,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " ",
        ""
    ]
)

fragmentos_narrativos = divisor_texto.split_documents(
    documentos_narrativos
)

# Cada sede permanece completa; no se fragmentan las filas de tabla.
fragmentos = fragmentos_narrativos + documentos_tabla

for indice, fragmento in enumerate(fragmentos):
    fragmento.metadata["chunk_id"] = indice
    fragmento.metadata["pagina_real"] = fragmento.metadata.get(
        "pagina_real",
        fragmento.metadata.get("page", 0) + 1
    )

if not fragmentos:
    raise ValueError("No se generaron fragmentos para construir FAISS.")

print(f"Fragmentos narrativos: {len(fragmentos_narrativos)}")
print(f"Registros completos de tabla: {len(documentos_tabla)}")
print(f"Total para FAISS: {len(fragmentos)}")


# =========================================================
# 7. EMBEDDINGS, FAISS Y RETRIEVERS
# =========================================================

MODELO_EMBEDDINGS = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo para embeddings: {dispositivo}")

modelo_embeddings = HuggingFaceEmbeddings(
    model_name=MODELO_EMBEDDINGS,
    model_kwargs={"device": dispositivo},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32
    }
)

vectorstore = FAISS.from_documents(
    documents=fragmentos,
    embedding=modelo_embeddings
)

# Recuperación general para reglamentos y requisitos.
retriever_general = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# Recuperación exclusiva de filas estructuradas para sedes, costos y horarios.
retriever_tablas = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 12,
        "filter": {"tipo_contenido": "registro_tabla"}
    }
)

print("FAISS y retrievers creados correctamente.")


# =========================================================
# CONFIGURACIÓN DE GEMINI PARA CLOUD RUN
# =========================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
MODELO_GEMINI = os.getenv(
    "GEMINI_MODEL",
    "gemini-3-flash-preview"
).strip()

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "No se encontró la variable de entorno GOOGLE_API_KEY."
    )

llm = ChatGoogleGenerativeAI(
    model=MODELO_GEMINI,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_retries=2
)

print(f"Gemini configurado con el modelo: {MODELO_GEMINI}")


# =========================================================
# 9. MEMORIA CONVERSACIONAL
# =========================================================

almacen_conversaciones: dict[str, InMemoryChatMessageHistory] = {}


def obtener_historial(
    session_id: str
) -> InMemoryChatMessageHistory:
    """Obtiene o crea el historial asociado a una sesión."""
    if session_id not in almacen_conversaciones:
        almacen_conversaciones[session_id] = (
            InMemoryChatMessageHistory()
        )

    return almacen_conversaciones[session_id]


def reiniciar_conversacion(
    session_id: str = "sesion-principal"
) -> None:
    """Limpia el historial de una sesión."""
    if session_id in almacen_conversaciones:
        almacen_conversaciones[session_id].clear()

    print(
        f"Conversación '{session_id}' reiniciada correctamente."
    )


def mostrar_historial(
    session_id: str = "sesion-principal"
) -> None:
    """Muestra el historial almacenado para depuración."""
    historial = obtener_historial(session_id)

    print(f"HISTORIAL DE LA SESIÓN: {session_id}")
    print("=" * 90)

    if not historial.messages:
        print("La conversación está vacía.")
        return

    for mensaje in historial.messages:
        if isinstance(mensaje, HumanMessage):
            rol = "USUARIO"
        elif isinstance(mensaje, AIMessage):
            rol = "ASISTENTE"
        else:
            rol = mensaje.__class__.__name__

        print(f"\n{rol}:")
        print(mensaje.content)


def convertir_historial_a_texto(
    historial: list,
    limite: int = 6
) -> str:
    """Convierte los últimos mensajes en texto para el triaje."""
    lineas = []

    for mensaje in (historial or [])[-limite:]:
        if isinstance(mensaje, HumanMessage):
            rol = "Usuario"
        elif isinstance(mensaje, AIMessage):
            rol = "Asistente"
        else:
            rol = "Mensaje"

        lineas.append(f"{rol}: {mensaje.content}")

    return "\n".join(lineas)


# =========================================================
# 10. PROMPTS DE REFORMULACIÓN Y RESPUESTA RAG
# =========================================================

prompt_reformular = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Reformula la última pregunta como una consulta independiente y completa,
usando el historial cuando sea necesario.

Reglas:
1. No respondas la pregunta.
2. Devuelve únicamente la pregunta reformulada.
3. Conserva el idioma español.
4. No agregues datos inexistentes.
5. Si ya es independiente, devuélvela sin cambios.
            """
        ),
        MessagesPlaceholder(variable_name="historial"),
        ("human", "{pregunta}")
    ]
)


prompt_rag_con_memoria = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Eres RetccBob, asistente especializado en el Registro Nacional de
Trabajadores de Construcción Civil (RETCC).

Responde exclusivamente con información del contexto recuperado.

El contexto puede incluir páginas narrativas y registros estructurados.
Cada registro estructurado representa una sola sede y contiene región,
oficina, dirección, aprobador, contacto, horario, página web y pago.

Reglas obligatorias:
1. No inventes información.
2. No uses conocimiento externo.
3. Si la respuesta no aparece, responde exactamente:
   "No encontré esa información en los documentos proporcionados."
4. Responde en español, con lenguaje claro y formal.
5. No mezcles regiones ni sedes.
6. Si una región tiene varias sedes, enuméralas por separado.
7. Relaciona cada horario, contacto y pago solo con su propia sede.
8. Conserva expresiones como "No indica" o "No reportó".
9. Enumera requisitos y pasos cuando corresponda.
10. Usa el historial solo para entender la consulta.
11. No menciones archivos, páginas, fuentes ni recuperación.
12. Responde únicamente lo solicitado.
            """
        ),
        MessagesPlaceholder(variable_name="historial"),
        (
            "human",
            """
CONTEXTO:

{contexto}

PREGUNTA ACTUAL:

{pregunta}
            """
        )
    ]
)


def reformular_pregunta(
    pregunta: str,
    historial: list
) -> str:
    """Convierte preguntas continuadas en consultas independientes."""
    if not historial:
        return pregunta.strip()

    mensajes = prompt_reformular.invoke(
        {
            "historial": historial,
            "pregunta": pregunta.strip()
        }
    )

    return llm.invoke(mensajes).content.strip()


# =========================================================
# 11. UTILIDADES DE RECUPERACIÓN
# =========================================================

PALABRAS_CONSULTA_TABLA = (
    "sede",
    "sedes",
    "oficina",
    "dirección",
    "direccion",
    "ubicación",
    "ubicacion",
    "región",
    "region",
    "horario",
    "teléfono",
    "telefono",
    "correo",
    "contacto",
    "página web",
    "pagina web",
    "costo",
    "precio",
    "pago",
    "tasa",
    "duplicado",
    "banco de la nación",
    "banco de la nacion",
    "centro mac",
    "mac lima"
)


def es_consulta_de_tabla(pregunta: str) -> bool:
    """Detecta consultas que requieren información por sede o región."""
    texto = pregunta.lower()
    return any(
        palabra in texto
        for palabra in PALABRAS_CONSULTA_TABLA
    )


def ampliar_consulta_tabla(pregunta: str) -> str:
    """Añade nombres de columnas para mejorar la recuperación semántica."""
    texto = pregunta.lower()
    complementos = []

    if any(
        palabra in texto
        for palabra in ("costo", "precio", "pago", "tasa", "duplicado")
    ):
        complementos.append(
            "PAGO POR DUPLICADO, tasa, monto, código y Banco de la Nación"
        )

    if "horario" in texto or "atención" in texto or "atencion" in texto:
        complementos.append("HORARIO DE ATENCIÓN")

    if any(
        palabra in texto
        for palabra in ("correo", "teléfono", "telefono", "contacto")
    ):
        complementos.append("CORREO Y/O TELÉFONO")

    if any(
        palabra in texto
        for palabra in ("sede", "oficina", "dirección", "direccion")
    ):
        complementos.append("REGIÓN, SEDE U OFICINA")

    if not complementos:
        return pregunta

    return f"{pregunta}. Buscar también: {'; '.join(complementos)}."


def formatear_contexto(documentos: list[Document]) -> str:
    """Convierte documentos recuperados en contexto para Gemini."""
    bloques = []

    for numero, documento in enumerate(documentos, start=1):
        bloques.append(
            f"""
FRAGMENTO {numero}
Contenido:
{documento.page_content}
""".strip()
        )

    return "\n\n" + ("\n\n" + "-" * 80 + "\n\n").join(bloques)


def obtener_fuentes(documentos: list[Document]) -> list[dict]:
    """Genera una lista de fuentes sin duplicados para depuración."""
    fuentes = []

    for documento in documentos:
        fuente = {
            "archivo": documento.metadata.get(
                "source",
                "Documento desconocido"
            ),
            "pagina": documento.metadata.get(
                "pagina_real",
                documento.metadata.get("page", 0) + 1
            )
        }

        if fuente not in fuentes:
            fuentes.append(fuente)

    return fuentes


# =========================================================
# 12. TRIAJE CON IA
# =========================================================

PROMPT_TRIAJE_RETCC = """
Eres el módulo de triaje de RetccBob.

Clasifica la consulta en exactamente una categoría:

RETCC:
Consultas directas o continuadas sobre inscripción, renovación, carné,
requisitos, documentos, vigencia, pérdida, duplicado, costos, pagos,
sedes, oficinas, regiones, centros MAC, direcciones, horarios, contactos,
seguimiento, observaciones, subsanación o construcción civil vinculada
al RETCC.

SALUDO:
Saludos, agradecimientos, despedidas o preguntas sobre las capacidades
del asistente.

FUERA_DE_TEMA:
Fútbol, política, recetas, programación, entretenimiento, noticias,
clima o cualquier asunto ajeno al RETCC.

AMBIGUA:
No hay información suficiente para identificar la necesidad.

Usa el historial para comprender preguntas continuadas.
No respondas la consulta. Solo clasifícala.

La confianza debe ser obligatoriamente un número decimal entre 0.0 y 1.0.
Nunca devuelvas porcentajes ni valores mayores que 1.
Ejemplos válidos: 0.50, 0.80, 0.95, 1.0.
"""


class TriajeRetccOut(BaseModel):
    decision: Literal[
        "RETCC",
        "SALUDO",
        "FUERA_DE_TEMA",
        "AMBIGUA"
    ]

    motivo: str = Field(
        description="Motivo breve de la clasificación."
    )

    confianza: float = Field(
        description="Confianza normalizada entre 0.0 y 1.0."
    )

    @field_validator("confianza", mode="before")
    @classmethod
    def normalizar_confianza(cls, valor):
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(numero, 1.0))


chain_triaje_retcc = llm.with_structured_output(
    TriajeRetccOut
)


def clasificar_consulta_retcc(
    pregunta: str,
    historial_texto: str = ""
) -> dict:
    """Clasifica la consulta antes de ejecutar el RAG."""
    contenido = f"""
HISTORIAL RECIENTE:
{historial_texto or "No existe historial previo."}

PREGUNTA ACTUAL:
{pregunta}
""".strip()

    salida = chain_triaje_retcc.invoke(
        [
            SystemMessage(content=PROMPT_TRIAJE_RETCC),
            HumanMessage(content=contenido)
        ]
    )

    return salida.model_dump()


# =========================================================
# 13. ESTADO Y MENSAJES DEL GRAFO
# =========================================================

class AgentState(TypedDict, total=False):
    pregunta: str
    session_id: str
    historial: list
    historial_texto: str
    triaje: dict
    clasificacion: str
    pregunta_reformulada: str
    respuesta: str
    fuentes: list[dict]
    documentos: list
    rag_exito: bool
    accion_final: str


MENSAJE_SALUDO_RETCC = """
¡Hola! Soy RetccBob, tu asistente virtual del Registro Nacional de
Trabajadores de Construcción Civil (RETCC).

Estoy aquí para ayudarte con información oficial sobre trámites,
requisitos y renovaciones de tu carné. ¿En qué puedo apoyarte hoy?
""".strip()

MENSAJE_FUERA_DE_TEMA_RETCC = MENSAJE_SALUDO_RETCC

MENSAJE_PREGUNTA_AMBIGUA = """
No logré identificar con claridad qué información necesitas.

Puedes consultarme sobre inscripción, renovación, requisitos, vigencia,
duplicado, costos, sedes, horarios o canales de atención del RETCC.
""".strip()

MENSAJE_RAG_SIN_RESULTADO = """
La consulta está relacionada con el RETCC, pero no encontré información
suficiente en los documentos cargados.

Reformula la pregunta indicando la región, sede, trámite o tipo de
información que necesitas.
""".strip()


# =========================================================
# 14. NODOS DEL GRAFO
# =========================================================

def nodo_triaje(state: AgentState) -> AgentState:
    print("\nEjecutando nodo: triaje")

    session_id = state.get("session_id", "sesion-principal")
    historial = obtener_historial(session_id).messages
    historial_texto = convertir_historial_a_texto(historial)

    resultado = clasificar_consulta_retcc(
        pregunta=state["pregunta"],
        historial_texto=historial_texto
    )

    print("Clasificación:", resultado["decision"])
    print("Confianza:", resultado["confianza"])
    print("Motivo:", resultado["motivo"])

    return {
        "historial": historial,
        "historial_texto": historial_texto,
        "triaje": resultado,
        "clasificacion": resultado["decision"]
    }


def ejecutar_rag_retcc(
    pregunta: str,
    historial: list
) -> dict:
    """Ejecuta únicamente reformulación, recuperación y respuesta."""
    pregunta_reformulada = reformular_pregunta(
        pregunta=pregunta,
        historial=historial
    )

    if es_consulta_de_tabla(pregunta_reformulada):
        consulta = ampliar_consulta_tabla(pregunta_reformulada)
        documentos = retriever_tablas.invoke(consulta)

        # Respaldo: si el filtro no devuelve resultados, usa recuperación general.
        if not documentos:
            documentos = retriever_general.invoke(consulta)
    else:
        documentos = retriever_general.invoke(
            pregunta_reformulada
        )

    contexto = formatear_contexto(documentos)

    mensajes = prompt_rag_con_memoria.invoke(
        {
            "historial": historial,
            "contexto": contexto,
            "pregunta": pregunta
        }
    )

    respuesta = llm.invoke(mensajes).content.strip()
    respuesta_no_encontrada = (
        "no encontré esa información"
        in respuesta.lower()
    )

    return {
        "pregunta_reformulada": pregunta_reformulada,
        "respuesta": respuesta,
        "fuentes": obtener_fuentes(documentos),
        "documentos": documentos,
        "rag_exito": bool(documentos) and not respuesta_no_encontrada
    }


def nodo_responder_rag(state: AgentState) -> AgentState:
    print("\nEjecutando nodo: responder_rag")

    resultado = ejecutar_rag_retcc(
        pregunta=state["pregunta"],
        historial=state.get("historial", [])
    )

    return {
        **resultado,
        "accion_final": "RESPONDER_RAG"
    }


def nodo_saludo(state: AgentState) -> AgentState:
    print("\nEjecutando nodo: saludo")
    return {
        "respuesta": MENSAJE_SALUDO_RETCC,
        "fuentes": [],
        "documentos": [],
        "rag_exito": True,
        "accion_final": "SALUDO"
    }


def nodo_fuera_de_tema(state: AgentState) -> AgentState:
    print("\nEjecutando nodo: fuera_de_tema")
    return {
        "respuesta": MENSAJE_FUERA_DE_TEMA_RETCC,
        "fuentes": [],
        "documentos": [],
        "rag_exito": False,
        "accion_final": "FUERA_DE_TEMA"
    }


def nodo_pedir_aclaracion(state: AgentState) -> AgentState:
    print("\nEjecutando nodo: pedir_aclaracion")
    return {
        "respuesta": MENSAJE_PREGUNTA_AMBIGUA,
        "fuentes": [],
        "documentos": [],
        "rag_exito": False,
        "accion_final": "PEDIR_ACLARACION"
    }


def nodo_rag_sin_resultado(state: AgentState) -> AgentState:
    print("\nEjecutando nodo: rag_sin_resultado")
    return {
        "respuesta": MENSAJE_RAG_SIN_RESULTADO,
        "accion_final": "RAG_SIN_RESULTADO"
    }


def nodo_guardar_memoria(state: AgentState) -> AgentState:
    print("\nEjecutando nodo: guardar_memoria")

    historial = obtener_historial(
        state.get("session_id", "sesion-principal")
    )
    historial.add_user_message(state["pregunta"])
    historial.add_ai_message(state["respuesta"])

    return {
        "accion_final": state.get(
            "accion_final",
            "FINALIZADO"
        )
    }


# =========================================================
# 15. RUTAS Y COMPILACIÓN DEL GRAFO
# =========================================================

def decidir_ruta_triaje(state: AgentState) -> str:
    rutas = {
        "RETCC": "rag",
        "SALUDO": "saludo",
        "FUERA_DE_TEMA": "fuera",
        "AMBIGUA": "ambigua"
    }

    return rutas.get(
        state["triaje"]["decision"],
        "ambigua"
    )


def decidir_resultado_rag(state: AgentState) -> str:
    return "ok" if state.get("rag_exito", False) else "sin_resultado"


workflow_retcc = StateGraph(AgentState)

workflow_retcc.add_node("triaje", nodo_triaje)
workflow_retcc.add_node("responder_rag", nodo_responder_rag)
workflow_retcc.add_node("saludo", nodo_saludo)
workflow_retcc.add_node("fuera_de_tema", nodo_fuera_de_tema)
workflow_retcc.add_node("pedir_aclaracion", nodo_pedir_aclaracion)
workflow_retcc.add_node("rag_sin_resultado", nodo_rag_sin_resultado)
workflow_retcc.add_node("guardar_memoria", nodo_guardar_memoria)

workflow_retcc.add_edge(START, "triaje")

workflow_retcc.add_conditional_edges(
    "triaje",
    decidir_ruta_triaje,
    {
        "rag": "responder_rag",
        "saludo": "saludo",
        "fuera": "fuera_de_tema",
        "ambigua": "pedir_aclaracion"
    }
)

workflow_retcc.add_conditional_edges(
    "responder_rag",
    decidir_resultado_rag,
    {
        "ok": "guardar_memoria",
        "sin_resultado": "rag_sin_resultado"
    }
)

workflow_retcc.add_edge("saludo", "guardar_memoria")
workflow_retcc.add_edge("fuera_de_tema", "guardar_memoria")
workflow_retcc.add_edge("pedir_aclaracion", "guardar_memoria")
workflow_retcc.add_edge("rag_sin_resultado", "guardar_memoria")
workflow_retcc.add_edge("guardar_memoria", END)

grafo_retcc = workflow_retcc.compile()

print("Grafo RETCC compilado correctamente.")


# =========================================================
# 16. FUNCIÓN PÚBLICA UTILIZADA POR GRADIO
# =========================================================

def preguntar_al_grafo_retcc(
    pregunta: str,
    session_id: str = "sesion-principal"
) -> dict:
    """Punto único de entrada al backend."""
    pregunta = str(pregunta or "").strip()

    if not pregunta:
        return {
            "pregunta_original": "",
            "pregunta_reformulada": "",
            "clasificacion": "AMBIGUA",
            "respuesta": "Debe ingresar una pregunta.",
            "fuentes": [],
            "documentos": [],
            "session_id": session_id,
            "accion_final": "VALIDACION"
        }

    resultado = grafo_retcc.invoke(
        {
            "pregunta": pregunta,
            "session_id": session_id
        }
    )

    return {
        "pregunta_original": pregunta,
        "pregunta_reformulada": resultado.get(
            "pregunta_reformulada",
            pregunta
        ),
        "clasificacion": resultado.get(
            "clasificacion",
            resultado.get("triaje", {}).get(
                "decision",
                "SIN_CLASIFICAR"
            )
        ),
        "respuesta": resultado.get(
            "respuesta",
            MENSAJE_PREGUNTA_AMBIGUA
        ),
        "fuentes": resultado.get("fuentes", []),
        "documentos": resultado.get("documentos", []),
        "session_id": session_id,
        "accion_final": resultado.get(
            "accion_final",
            "FINALIZADO"
        )
    }


def probar_backend_retcc(
    pregunta: str,
    session_id: str = "prueba-retcc"
) -> dict:
    """Ejecuta una consulta y muestra información de depuración."""
    resultado = preguntar_al_grafo_retcc(
        pregunta=pregunta,
        session_id=session_id
    )

    print("=" * 90)
    print("PREGUNTA:", resultado["pregunta_original"])
    print("CLASIFICACIÓN:", resultado["clasificacion"])
    print("ACCIÓN:", resultado["accion_final"])
    print("REFORMULADA:", resultado["pregunta_reformulada"])
    print("-" * 90)
    print(resultado["respuesta"])

    return resultado




def crear_mensaje(rol: str, texto: str) -> dict:
    return {
        "role": rol,
        "content": texto
    }





# =========================================================
# ARCHIVOS VISUALES DEL FRONTEND
# =========================================================
#
# Guarda los archivos dentro de:
#
# retcc_assets/
# ├── avatar_retcc.png
# └── casco.svg
#
# Las imágenes se sirven como archivos estáticos para evitar
# incorporarlas en Base64 dentro del código.
# =========================================================

RUTA_ASSETS_RETCC = RUTA_BASE / "retcc_assets"

RUTA_ASSETS_RETCC.mkdir(
    parents=True,
    exist_ok=True
)


try:
    gr.set_static_paths(
        paths=[
            str(RUTA_ASSETS_RETCC.resolve())
        ]
    )
except Exception as error:
    print(
        "No fue posible registrar la carpeta de recursos:",
        repr(error)
    )


def url_asset_retcc(
    nombre_archivo: str
) -> str:
    """
    Genera la URL que Gradio utiliza para servir
    un archivo ubicado en retcc_assets.
    """

    ruta = (
        RUTA_ASSETS_RETCC / nombre_archivo
    ).resolve()

    return (
        f"/gradio_api/file={ruta.as_posix()}"
    )


AVATAR_RETCC_SRC = url_asset_retcc(
    "avatar_retcc.png"
)

CASCO_RETCC_SRC = url_asset_retcc(
    "casco.svg"
)


# =========================================================
# NORMALIZACIÓN DEL HISTORIAL VISUAL
# =========================================================

def normalizar_historial_html(
    historial
) -> list:
    """
    Convierte cualquier historial recibido en una lista con
    la siguiente estructura:

    [
        {
            "role": "user",
            "content": "..."
        },
        {
            "role": "assistant",
            "content": "..."
        }
    ]
    """

    resultado = []

    for item in historial or []:

        role = ""
        content = ""

        if isinstance(item, dict):

            role = item.get(
                "role",
                ""
            )

            content = item.get(
                "content",
                ""
            )

        elif (
            isinstance(item, (list, tuple))
            and len(item) == 2
        ):

            role = item[0]
            content = item[1]

        else:
            continue

        role = str(role).strip().lower()

        if role not in {
            "user",
            "assistant"
        }:
            continue

        resultado.append(
            {
                "role": role,
                "content": str(
                    content or ""
                )
            }
        )

    return resultado


# =========================================================
# CONVERSIÓN DE TEXTO A HTML
# =========================================================

def texto_a_html(
    texto: str
) -> str:
    """
    Convierte texto plano en HTML seguro.

    Funcionalidades:

    - Escapa etiquetas HTML.
    - Conserva saltos de línea.
    - Reconoce listas iniciadas con * o -.
    - Reconoce listas numéricas simples.
    """

    texto_seguro = html.escape(
        str(texto or "")
    )

    lineas = texto_seguro.splitlines()

    partes = []

    tipo_lista_abierta = None

    def cerrar_lista():
        nonlocal tipo_lista_abierta

        if tipo_lista_abierta == "ul":
            partes.append("</ul>")

        elif tipo_lista_abierta == "ol":
            partes.append("</ol>")

        tipo_lista_abierta = None

    for linea in lineas:

        linea_limpia = linea.strip()

        # ---------------------------------------------
        # Lista con viñetas
        # ---------------------------------------------

        if (
            linea_limpia.startswith("* ")
            or linea_limpia.startswith("- ")
        ):

            if tipo_lista_abierta != "ul":

                cerrar_lista()

                partes.append("<ul>")

                tipo_lista_abierta = "ul"

            partes.append(
                f"<li>{linea_limpia[2:]}</li>"
            )

            continue

        # ---------------------------------------------
        # Lista numérica
        # ---------------------------------------------

        coincidencia_numerica = re.match(
            r"^\d+[.)]\s+(.+)$",
            linea_limpia
        )

        if coincidencia_numerica:

            if tipo_lista_abierta != "ol":

                cerrar_lista()

                partes.append("<ol>")

                tipo_lista_abierta = "ol"

            partes.append(
                f"<li>{coincidencia_numerica.group(1)}</li>"
            )

            continue

        # ---------------------------------------------
        # Texto normal
        # ---------------------------------------------

        cerrar_lista()

        if linea_limpia:

            partes.append(
                f"<p>{linea_limpia}</p>"
            )

        else:

            partes.append("<br>")

    cerrar_lista()

    return "".join(partes)


# =========================================================
# RENDERIZADO COMPLETO DEL CHAT
# =========================================================

def renderizar_chat_html(
    historial,
    escribiendo: bool = False
) -> str:
    """
    Renderiza el historial utilizando el HTML personalizado
    de RetccBob.

    Se utiliza HTML propio para no depender de los cambios
    internos de gr.Chatbot entre versiones de Gradio.
    """

    historial_normalizado = normalizar_historial_html(
        historial
    )

    if not historial_normalizado:

        return """
        <div class="chat-vacio-retcc">
            <span>
                La conversación aparecerá aquí.
            </span>
        </div>
        """

    mensajes_html = []

    for mensaje in historial_normalizado:

        role = mensaje["role"]

        contenido = texto_a_html(
            mensaje["content"]
        )

        if role == "user":

            clase = "mensaje-usuario-retcc"

            etiqueta = "Tú"

            avatar_html = ""

        else:

            clase = "mensaje-asistente-retcc"

            etiqueta = "RetccBob"

            avatar_html = f"""
            <div class="avatar-respuesta-retcc">
                <img
                    src="{AVATAR_RETCC_SRC}"
                    alt="RetccBob"
                >
            </div>
            """

        mensajes_html.append(
            f"""
            <div class="fila-mensaje-retcc {clase}">

                {avatar_html}

                <div class="burbuja-mensaje-retcc">

                    <div class="etiqueta-mensaje-retcc">
                        {etiqueta}
                    </div>

                    <div class="contenido-mensaje-retcc">
                        {contenido}
                    </div>

                </div>

            </div>
            """
        )

    if escribiendo:
        mensajes_html.append(
            f'''
            <div class="fila-mensaje-retcc mensaje-asistente-retcc indicador-escribiendo-retcc">
                <div class="avatar-respuesta-retcc">
                    <img
                        src="{AVATAR_RETCC_SRC}"
                        alt="RetccBob"
                    >
                </div>

                <div class="burbuja-mensaje-retcc burbuja-escribiendo-retcc">
                    <div class="etiqueta-mensaje-retcc">
                        RetccBob
                    </div>

                    <div class="contenido-mensaje-retcc">
                        <span class="texto-escribiendo-retcc">Escribiendo</span>
                        <span class="puntos-escribiendo-retcc" aria-label="Escribiendo">
                            <span></span>
                            <span></span>
                            <span></span>
                        </span>
                    </div>
                </div>
            </div>
            '''
        )

    return (
        '<div id="historial-chat-html-retcc">'
        + "".join(mensajes_html)
        + '<div id="fin-chat-retcc"></div>'
        + "</div>"
    )


# =========================================================
# PASO 1:
# MOSTRAR INMEDIATAMENTE LA PREGUNTA DEL USUARIO
# =========================================================

def mostrar_pregunta_usuario_html(
    mensaje: str,
    historial_estado: list,
    session_id: str
):
    """
    Primer paso del envío.

    Esta función no llama al backend.

    Únicamente:

    1. Valida la pregunta.
    2. Crea la sesión cuando sea necesario.
    3. Agrega la pregunta al historial visual.
    4. Limpia el input.
    5. Actualiza inmediatamente el chat.
    6. Guarda la pregunta en pregunta_pendiente.

    Retorna:

    - input vacío;
    - HTML actualizado;
    - historial actualizado;
    - session_id;
    - pregunta pendiente.
    """

    historial = normalizar_historial_html(
        historial_estado
    )

    pregunta = str(
        mensaje or ""
    ).strip()

    if not session_id:

        session_id = str(
            uuid.uuid4()
        )

    if not pregunta:

        return (
            "",
            renderizar_chat_html(historial),
            historial,
            session_id,
            ""
        )

    historial.append(
        {
            "role": "user",
            "content": pregunta
        }
    )

    return (
        "",
        renderizar_chat_html(
            historial,
            escribiendo=True
        ),
        historial,
        session_id,
        pregunta
    )


# =========================================================
# PASO 2:
# EJECUTAR EL BACKEND Y MOSTRAR LA RESPUESTA
# =========================================================

def generar_respuesta_asistente_html(
    pregunta_pendiente: str,
    historial_estado: list,
    session_id: str
):
    """
    Segundo paso del envío.

    Esta función se ejecuta después de que Gradio ya mostró
    la pregunta del usuario.

    Ejecuta:

    - Triaje.
    - LangGraph.
    - Recuperación FAISS.
    - Gemini.
    - Memoria conversacional.

    Retorna:

    - HTML actualizado;
    - historial actualizado;
    - session_id;
    - pregunta pendiente vacía.
    """

    historial = normalizar_historial_html(
        historial_estado
    )

    pregunta = str(
        pregunta_pendiente or ""
    ).strip()

    if not session_id:

        session_id = str(
            uuid.uuid4()
        )

    if not pregunta:

        return (
            renderizar_chat_html(historial),
            historial,
            session_id,
            ""
        )

    try:

        print("\n" + "=" * 90)
        print("CONSULTA RECIBIDA DESDE GRADIO")
        print("=" * 90)
        print("Pregunta:", pregunta)
        print("Session ID:", session_id)

        resultado = preguntar_al_grafo_retcc(
            pregunta=pregunta,
            session_id=session_id
        )

        print("\n" + "-" * 90)
        print("RESULTADO DEL GRAFO")
        print("-" * 90)

        print(
            "Clasificación:",
            resultado.get(
                "clasificacion",
                "SIN_CLASIFICAR"
            )
        )

        print(
            "Acción final:",
            resultado.get(
                "accion_final",
                "SIN_ACCION"
            )
        )

        print(
            "Pregunta reformulada:",
            resultado.get(
                "pregunta_reformulada",
                pregunta
            )
        )

        respuesta = str(
            resultado.get(
                "respuesta",
                "No se pudo generar una respuesta."
            )
        ).strip()

        if not respuesta:

            respuesta = (
                "No se pudo generar una respuesta."
            )

        historial.append(
            {
                "role": "assistant",
                "content": respuesta
            }
        )

    except Exception:

        print("\n" + "=" * 90)
        print("ERROR COMPLETO DEL CHATBOT")
        print("=" * 90)

        traceback.print_exc()

        historial.append(
            {
                "role": "assistant",
                "content": (
                    "No fue posible procesar la consulta. "
                    "Por favor, intenta nuevamente."
                )
            }
        )

    return (
        renderizar_chat_html(historial),
        historial,
        session_id,
        ""
    )


# =========================================================
# CONSULTAS RÁPIDAS
# =========================================================

def mostrar_consulta_rapida_html(
    pregunta: str,
    historial_estado: list,
    session_id: str
):
    """
    Primer paso para los botones de consultas rápidas.

    Utiliza el mismo flujo inmediato del input normal.
    """

    return mostrar_pregunta_usuario_html(
        mensaje=pregunta,
        historial_estado=historial_estado,
        session_id=session_id
    )


# =========================================================
# NUEVA CONVERSACIÓN
# =========================================================

def nueva_conversacion_html(
    session_id: str
):
    """
    Limpia:

    - La memoria conversacional del backend.
    - El historial visual.
    - La pregunta pendiente.
    - El contenido del input.

    También genera un nuevo identificador de sesión.

    Retorna:

    - HTML vacío;
    - historial vacío;
    - nuevo session_id;
    - input vacío;
    - pregunta pendiente vacía.
    """

    if session_id:

        try:

            reiniciar_conversacion(
                session_id
            )

        except Exception as error:

            print(
                "No se pudo reiniciar la conversación:",
                repr(error)
            )

    nuevo_session_id = str(
        uuid.uuid4()
    )

    historial = []

    return (
        renderizar_chat_html(historial),
        historial,
        nuevo_session_id,
        "",
        ""
    )


print(
    "Funciones HTML del frontend cargadas correctamente."
)

# ---------------------------------------------------------
# CELDA 58: LAYOUT FINAL ESTABLE
# Sidebar fija + conversación visible + input fijo
# ---------------------------------------------------------

CSS_RETCC = """
:root {
    --rojo: #c90008;
    --rojo-oscuro: #9b0005;
    --fondo: #faf8f7;
    --blanco: #ffffff;
    --borde: #edb8b4;
    --texto: #252525;
}

* {
    box-sizing: border-box !important;
}

html,
body,
gradio-app,
.gradio-container {
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: var(--fondo) !important;
    color-scheme: light !important;
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}

footer {
    display: none !important;
}

/* =====================================================
   ESTRUCTURA GENERAL
   ===================================================== */

#layout-retcc {
    width: 100% !important;
    height: 100vh !important;
    height: 100dvh !important;
    min-height: 0 !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

#panel-lateral {
    position: relative !important;
    flex: 0 0 290px !important;
    width: 308px !important;
    min-width: 308px !important;
    max-width: 308px !important;
    height: 100% !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: #ffffff !important;
    border-right: 1px solid var(--borde) !important;
}

#zona-principal {
    flex: 1 1 auto !important;
    width: auto !important;
    min-width: 0 !important;
    height: 100% !important;
    min-height: 0 !important;
    display: grid !important;
    grid-template-rows: 74px 32px minmax(0, 1fr) !important;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: var(--fondo) !important;
}

/* =====================================================
   SIDEBAR EXACTA
   ===================================================== */

.sidebar-retcc {
    position: relative;
    width: 100%;
    height: 100%;
    background: #ffffff;
}

.sidebar-superior {
    padding: 28px 25px 16px;
}

.titulo-recursos {
    margin: 0 0 19px;
    color: #a50005 !important;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: .15px;
}

.recurso-retcc {
    min-height: 60px;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 14px;
    color: #282828 !important;
    text-decoration: none !important;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 500;
}

.recurso-retcc span {
    color: #282828 !important;
    opacity: 1 !important;
}

.recurso-retcc:hover {
    background: #f7f2f1;
}

.recurso-icono {
    width: 25px;
    min-width: 25px;
    color: #505050 !important;
    text-align: center;
    font-size: 21px;
}

.sidebar-inferior {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 18px;
    padding: 25px 25px 0;
    border-top: 1px solid var(--borde);
    background: #ffffff;
}

.tarjeta-ayuda {
    min-height: 164px;
    padding: 18px 17px 66px;
    margin: 0 0 26px;
    background: #fde8e6;
    border-radius: 10px;
}

.tarjeta-ayuda h3 {
    margin: 0 0 10px;
    color: #8d0004 !important;
    font-size: 14px;
    font-weight: 800;
}

.tarjeta-ayuda p {
    margin: 0;
    color: #5b4542 !important;
    font-size: 13px;
    line-height: 1.55;
}

.usuario-retcc {
    min-height: 56px;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 3px 8px;
}

.usuario-avatar {
    width: 43px;
    height: 43px;
    display: grid;
    place-items: center;
    color: #444444;
    background: #ece9e7;
    border-radius: 12px;
    font-size: 20px;
}

.usuario-nombre {
    color: #222222 !important;
    font-size: 14px;
    font-weight: 800;
    line-height: 1.15;
}

.usuario-rol {
    margin-top: 2px;
    color: #6c6c6c !important;
    font-size: 12px;
    line-height: 1.15;
}

#boton-soporte {
    position: absolute !important;
    left: 43px !important;
    bottom: 109px !important;
    z-index: 20 !important;
    width: 222px !important;
    min-width: 222px !important;
    max-width: 222px !important;
    min-height: 40px !important;
    margin: 0 !important;
    color: #ffffff !important;
    background: var(--rojo) !important;
    border: 0 !important;
    border-radius: 5px !important;
    font-size: 13px !important;
    font-weight: 800 !important;
}

#boton-nuevo {
    display: none !important;
}


.logo-retcc {
    width: 44px !important;
    height: 44px !important;
    display: grid !important;
    place-items: center !important;
    overflow: hidden !important;
    background: #ffffff !important;
    border-radius: 11px !important;
}

.logo-retcc img {
    width: 36px !important;
    height: 36px !important;
    display: block !important;
    object-fit: contain !important;
}

/* =====================================================
   CABECERA Y ESTADO
   ===================================================== */

#cabecera-wrap {
    grid-row: 1 !important;
    width: 100% !important;
    height: 74px !important;
    min-height: 74px !important;
    padding: 0 !important;
    overflow: hidden !important;
}

#estado-wrap {
    grid-row: 2 !important;
    width: 100% !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 !important;
    overflow: hidden !important;
}

.cabecera-retcc {
    width: 100%;
    height: 74px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    color: #ffffff;
    background: var(--rojo);
    box-shadow: 0 2px 8px rgba(80, 0, 0, .14);
}

.cabecera-marca {
    display: flex;
    align-items: center;
    gap: 13px;
}

.logo-retcc {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    padding: 5px;
    background: #ffffff;
    border-radius: 12px;
}

.casco-retcc {
    width: 32px;
    height: 32px;
}

.cabecera-titulo {
    margin: 0;
    color: #ffffff !important;
    font-size: 20px;
    font-weight: 800;
}

.cabecera-info {
    color: #ffffff !important;
    font-size: 24px;
}

.estado-retcc {
    width: 100%;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    color: #555555;
    background: #f2efee;
    border-bottom: 1px solid var(--borde);
    font-size: 12px;
}

.estado-retcc strong {
    color: var(--rojo);
}

.barra-estado {
    width: 132px;
    height: 6px;
    overflow: hidden;
    background: #e9c4c1;
    border-radius: 999px;
}

.barra-estado span {
    display: block;
    width: 22%;
    height: 100%;
    background: var(--rojo);
}

/* =====================================================
   CONTENIDO CENTRAL
   ===================================================== */

#contenido-retcc {
    grid-row: 3 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    display: grid !important;
    grid-template-rows: minmax(0, 1fr) auto !important;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: var(--fondo) !important;
}

#conversacion-scroll {
    grid-row: 1 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    display: block !important;
    padding: 18px 28px 16px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    background: var(--fondo) !important;
    scrollbar-width: thin;
    scrollbar-color: #cfa29e transparent;
}

#conversacion-scroll::-webkit-scrollbar {
    width: 8px;
}

#conversacion-scroll::-webkit-scrollbar-thumb {
    background: #cfa29e;
    border-radius: 999px;
}

#zona-entrada {
    grid-row: 2 !important;
    width: 100% !important;
    min-height: 78px !important;
    padding: 8px 28px 7px !important;
    background: var(--fondo) !important;
    border-top: 1px solid var(--borde) !important;
}

/* =====================================================
   BLOQUES DE BIENVENIDA
   ===================================================== */

#bloque-bienvenida,
#bloque-informativo,
.consultas-grid,
#chat-html-retcc {
    width: min(100%, 760px) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

.fila-asistente {
    display: grid;
    grid-template-columns: 42px minmax(0, 1fr);
    gap: 16px;
    align-items: start;
}

.icono-asistente {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    overflow: hidden;
    background: #ffffff;
    border: 2px solid var(--rojo);
    border-radius: 13px;
    box-shadow: 0 2px 7px rgba(85, 0, 0, .15);
}

.icono-asistente img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 18%;
}

.burbuja-bienvenida {
    padding: 17px 19px;
    color: var(--texto) !important;
    background: #ffffff;
    border: 1px solid var(--borde);
    border-radius: 11px;
    font-size: 15px;
    line-height: 1.45;
}

.burbuja-bienvenida * {
    color: var(--texto) !important;
}

.consultas-grid {
    width: min(calc(100% - 58px), 702px) !important;
    margin-top: 10px !important;
    transform: translateX(29px);
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 9px !important;
}

.boton-rapido {
    width: auto !important;
    min-width: 0 !important;
    min-height: 38px !important;
    flex: 0 0 auto !important;
    padding: 7px 16px !important;
    color: var(--rojo) !important;
    background: #ffffff !important;
    border: 1px solid var(--rojo) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    font-size: 12px !important;
    font-weight: 750 !important;
}

#bloque-informativo {
    margin-top: 12px !important;
}

.tarjeta-informativa {
    overflow: hidden;
    min-height: 245px;
    background: #ffffff;
    border: 1px solid var(--borde);
    border-radius: 10px;
}

.imagen-informativa {
    height: 160px;
    min-height: 160px;
    display: flex;
    align-items: end;
    padding: 18px 20px;
    color: #ffffff;
    background:
        linear-gradient(0deg, rgba(70, 0, 0, .72), rgba(120, 0, 0, .12)),
        url("https://lh3.googleusercontent.com/aida-public/AB6AXuAKUeXdi1JWzDkEgAfqRIrSYHrMc6M9PsnNkvXkc5_jWALheiEB6LZaQK3YWuqE8bbDHRofbA1cE-C7cFWmV9y1XHz-ip2mes16dRQ_XGnFBsijM8ibFuWUA2KB5Re9DW_uqd2bfyFcGbJG0Gauyb8_BQDvUcWHoQRsiAkZtqDiSy_77EpQgBl5S3tgCpaZ7ltLS7fOlKmyi_xzvcK0t1AXFPTs8K3hxFxo_Uj31uZUGvBlR2D5htoodA");
    background-size: cover;
    background-position: center 48%;
}

.imagen-informativa strong {
    color: #ffffff !important;
    font-size: 17px;
}

.contenido-informativo {
    padding: 12px 18px 13px;
    color: #333333;
    font-size: 12px;
    line-height: 1.4;
}

.contenido-informativo * {
    color: #333333 !important;
}

.lista-informativa {
    display: grid;
    gap: 5px;
}

.lista-informativa span::before {
    content: "✓";
    display: inline-grid;
    place-items: center;
    width: 14px;
    height: 14px;
    margin-right: 6px;
    color: var(--rojo);
    border: 1.3px solid var(--rojo);
    border-radius: 50%;
    font-size: 8px;
    font-weight: 900;
}

/* =====================================================
   CHAT HTML
   ===================================================== */

#chat-html-retcc {
    margin-top: 14px !important;
    padding-bottom: 6px !important;
}

#chat-html-retcc .html-container,
#chat-html-retcc .prose {
    width: 100% !important;
    max-width: none !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
}

#historial-chat-html-retcc {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.chat-vacio-retcc {
    min-height: 20px;
    color: #8a8a8a;
    text-align: center;
    font-size: 12px;
}

.fila-mensaje-retcc {
    width: 100%;
    display: flex;
}

.mensaje-usuario-retcc {
    justify-content: flex-end;
}

.mensaje-asistente-retcc {
    justify-content: flex-start;
    align-items: flex-start;
    gap: 12px;
}

.avatar-respuesta-retcc {
    flex: 0 0 42px;
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    overflow: hidden;
    background: #ffffff;
    border: 2px solid var(--rojo);
    border-radius: 13px;
    box-shadow: 0 2px 7px rgba(85, 0, 0, .15);
}

.avatar-respuesta-retcc img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 18%;
}

.burbuja-mensaje-retcc {
    width: fit-content;
    min-width: 150px;
    max-width: 78%;
    padding: 12px 15px;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(50, 0, 0, .05);
}

.mensaje-usuario-retcc .burbuja-mensaje-retcc {
    color: #ffffff;
    background: var(--rojo);
    border-bottom-right-radius: 4px;
}

.mensaje-asistente-retcc .burbuja-mensaje-retcc {
    color: #111111 !important;
    background: #ffffff;
    border: 1px solid var(--borde);
    border-bottom-left-radius: 4px;
}

.mensaje-asistente-retcc .etiqueta-mensaje-retcc,
.mensaje-asistente-retcc .contenido-mensaje-retcc,
.mensaje-asistente-retcc .contenido-mensaje-retcc * {
    color: #111111 !important;
    opacity: 1 !important;
}

.etiqueta-mensaje-retcc {
    margin-bottom: 5px;
    font-size: 10px;
    font-weight: 800;
    opacity: .8;
    text-transform: uppercase;
    letter-spacing: .4px;
}

.contenido-mensaje-retcc {
    font-size: 13px;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

.contenido-mensaje-retcc p {
    margin: 0 0 8px;
}

.contenido-mensaje-retcc p:last-child {
    margin-bottom: 0;
}

/* =====================================================
   INPUT
   ===================================================== */

#fila-entrada {
    width: min(100%, 900px) !important;
    min-height: 58px !important;
    margin: 0 auto !important;
    display: flex !important;
    align-items: center !important;
    gap: 7px !important;
    padding: 6px !important;
    background: #ffffff !important;
    border: 1px solid var(--borde) !important;
    border-radius: 15px !important;
}

#entrada-mensaje {
    flex: 1 1 auto !important;
    min-width: 0 !important;
}

#entrada-mensaje,
#entrada-mensaje > div {
    background: #ffffff !important;
    border: 0 !important;
    box-shadow: none !important;
}

#entrada-mensaje textarea {
    min-height: 44px !important;
    max-height: 76px !important;
    padding: 11px 12px !important;
    color: #292929 !important;
    background: #ffffff !important;
    border: 0 !important;
    box-shadow: none !important;
    font-size: 14px !important;
}

#boton-enviar {
    flex: 0 0 48px !important;
    min-width: 48px !important;
    max-width: 48px !important;
    height: 44px !important;
    padding: 0 !important;
    color: #ffffff !important;
    background: var(--rojo) !important;
    border: 0 !important;
    border-radius: 10px !important;
}

.pie-retcc {
    margin-top: 3px;
    color: #707070;
    text-align: center;
    font-size: 8px;
    letter-spacing: .6px;
    text-transform: uppercase;
}

/* =====================================================
   RESPONSIVE
   ===================================================== */

@media (max-width: 900px) {
    #panel-lateral {
        display: none !important;
    }

    #conversacion-scroll {
        padding: 10px 12px 9px !important;
    }

    #zona-entrada {
        padding: 6px 12px 5px !important;
    }

    .burbuja-mensaje-retcc {
        max-width: 92%;
    }
}


/* =====================================================
   AJUSTE FINAL DE LA BARRA LATERAL
   Replica la distribución del modelo compartido.
   ===================================================== */

#panel-lateral,
#panel-lateral > div,
#panel-lateral .form,
#panel-lateral .wrap {
    height: 100% !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
    overflow: hidden !important;
    position: relative !important;
    background: #ffffff !important;
}

#panel-lateral .html-container {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    border: 0 !important;
    background: transparent !important;
}

.sidebar-retcc {
    position: absolute !important;
    inset: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    background: #ffffff !important;
}

.sidebar-superior {
    flex: 0 0 auto !important;
    padding: 28px 25px 16px !important;
}

.sidebar-inferior {
    position: absolute !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 18px !important;
    padding: 25px 25px 0 !important;
    margin: 0 !important;
    border-top: 1px solid var(--borde) !important;
    background: #ffffff !important;
}

.tarjeta-ayuda {
    width: 100% !important;
    min-height: 164px !important;
    padding: 18px 17px 66px !important;
    margin: 0 0 26px !important;
    background: #fde8e6 !important;
    border-radius: 10px !important;
}

.usuario-retcc {
    width: 100% !important;
    min-height: 56px !important;
    padding: 3px 8px !important;
    margin: 0 !important;
}

#boton-soporte {
    position: absolute !important;
    left: 43px !important;
    bottom: 109px !important;
    z-index: 50 !important;
    width: 222px !important;
    min-width: 222px !important;
    max-width: 222px !important;
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 16px !important;
    margin: 0 !important;
    color: #ffffff !important;
    background: var(--rojo) !important;
    border: 0 !important;
    border-radius: 5px !important;
    box-shadow: none !important;
    font-size: 13px !important;
    font-weight: 800 !important;
}

#boton-soporte > button {
    width: 100% !important;
    height: 100% !important;
    min-height: 40px !important;
    padding: 0 !important;
    margin: 0 !important;
    color: #ffffff !important;
    background: var(--rojo) !important;
    border: 0 !important;
    border-radius: 5px !important;
    box-shadow: none !important;
    font-weight: 800 !important;
}

#boton-nuevo {
    display: none !important;
}

/* Imágenes externas: ya no se incrustan en Base64. */
.icono-asistente img,
.avatar-respuesta-retcc img,
.logo-retcc img {
    display: block !important;
    object-fit: contain !important;
}


/* =====================================================
   INDICADOR "ESCRIBIENDO..."
   ===================================================== */

.indicador-escribiendo-retcc {
    animation: aparecer-escribiendo-retcc .18s ease-out;
}

.burbuja-escribiendo-retcc {
    min-width: 148px !important;
    padding-top: 11px !important;
    padding-bottom: 11px !important;
}

.burbuja-escribiendo-retcc .contenido-mensaje-retcc {
    display: flex;
    align-items: center;
    gap: 8px;
}

.texto-escribiendo-retcc {
    color: #666666 !important;
    font-size: 12px;
    font-style: italic;
}

.puntos-escribiendo-retcc {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 16px;
}

.puntos-escribiendo-retcc span {
    width: 6px;
    height: 6px;
    display: block;
    background: #8d8d8d;
    border-radius: 50%;
    animation: punto-escribiendo-retcc 1.15s infinite ease-in-out;
}

.puntos-escribiendo-retcc span:nth-child(2) {
    animation-delay: .16s;
}

.puntos-escribiendo-retcc span:nth-child(3) {
    animation-delay: .32s;
}

@keyframes punto-escribiendo-retcc {
    0%,
    60%,
    100% {
        transform: translateY(0);
        opacity: .35;
    }

    30% {
        transform: translateY(-4px);
        opacity: 1;
    }
}

@keyframes aparecer-escribiendo-retcc {
    from {
        transform: translateY(5px);
        opacity: 0;
    }

    to {
        transform: translateY(0);
        opacity: 1;
    }
}

/* =====================================================
   EVITAR TRANSPARENCIA O BLOQUEO DURANTE EL PROCESAMIENTO
   ===================================================== */

/*
Gradio agrega clases como pending o generating mientras
se procesa una función. Se fuerza la interfaz a conservar
su apariencia normal.
*/

.gradio-container .pending,
.gradio-container .generating,
.gradio-container [class*="pending"],
.gradio-container [class*="generating"] {
    opacity: 1 !important;
    filter: none !important;
}

/*
El componente que contiene la conversación nunca debe
verse deshabilitado ni transparente.
*/

#chat-html-retcc,
#chat-html-retcc.pending,
#chat-html-retcc.generating,
#chat-html-retcc > div,
#chat-html-retcc.pending > div,
#chat-html-retcc.generating > div {
    opacity: 1 !important;
    filter: none !important;
    background: transparent !important;
}

/*
La conversación debe seguir mostrando sus mensajes con
sus colores completos mientras Gemini genera la respuesta.
*/

#historial-chat-html-retcc,
#historial-chat-html-retcc *,
#conversacion-scroll,
#conversacion-scroll * {
    filter: none !important;
}

/*
Burbuja del usuario: rojo sólido desde el primer instante.
*/

.mensaje-usuario-retcc,
.mensaje-usuario-retcc .burbuja-mensaje-retcc,
.mensaje-usuario-retcc .burbuja-mensaje-retcc *,
#chat-html-retcc.pending .mensaje-usuario-retcc,
#chat-html-retcc.pending .mensaje-usuario-retcc .burbuja-mensaje-retcc,
#chat-html-retcc.generating .mensaje-usuario-retcc,
#chat-html-retcc.generating .mensaje-usuario-retcc .burbuja-mensaje-retcc {
    opacity: 1 !important;
    filter: none !important;
}

.mensaje-usuario-retcc .burbuja-mensaje-retcc,
#chat-html-retcc.pending .mensaje-usuario-retcc .burbuja-mensaje-retcc,
#chat-html-retcc.generating .mensaje-usuario-retcc .burbuja-mensaje-retcc {
    color: #ffffff !important;
    background: #c90008 !important;
    border: none !important;
    box-shadow: 0 2px 7px rgba(120, 0, 0, .18) !important;
}

.mensaje-usuario-retcc .etiqueta-mensaje-retcc,
.mensaje-usuario-retcc .contenido-mensaje-retcc,
.mensaje-usuario-retcc .contenido-mensaje-retcc *,
#chat-html-retcc.pending .mensaje-usuario-retcc *,
#chat-html-retcc.generating .mensaje-usuario-retcc * {
    color: #ffffff !important;
    opacity: 1 !important;
}

/*
Permite visualizar normalmente el chat mientras el backend
está trabajando. Solo se mantiene bloqueada la ejecución
duplicada administrada por Gradio.
*/

#chat-html-retcc {
    pointer-events: auto !important;
}

"""

# ---------------------------------------------------------
# CELDA 59: INTERFAZ FINAL CON LAYOUT FIJO
# ---------------------------------------------------------


# Mantiene el último mensaje visible después de cada actualización.
JS_SCROLL_FINAL_RETCC = """
(...args) => {
    const desplazarAlFinal = () => {
        const contenedor = document.querySelector("#conversacion-scroll");
        const fin = document.querySelector("#fin-chat-retcc");

        if (!contenedor) {
            return;
        }

        if (fin) {
            fin.scrollIntoView({
                behavior: "smooth",
                block: "end"
            });
        } else {
            contenedor.scrollTo({
                top: contenedor.scrollHeight,
                behavior: "smooth"
            });
        }
    };

    // Primer desplazamiento y repeticiones breves para esperar
    // a que Gradio termine de actualizar el HTML.
    requestAnimationFrame(desplazarAlFinal);
    setTimeout(desplazarAlFinal, 80);
    setTimeout(desplazarAlFinal, 220);
    setTimeout(desplazarAlFinal, 500);

    return args;
}
"""


with gr.Blocks(
    title="Chat RETCC Bob",
    css=CSS_RETCC,
    fill_height=True,
    theme=gr.themes.Soft(
        primary_hue="red",
        neutral_hue="gray"
    )
) as interfaz_retcc:

    session_id = gr.State(
    value=str(uuid.uuid4())
    )

    historial_estado = gr.State(
        value=[]
    )

    # Guarda temporalmente la pregunta mientras se ejecuta el backend.
    pregunta_pendiente = gr.State(
        value=""
    )

    with gr.Row(elem_id="layout-retcc"):

        with gr.Column(
            scale=0,
            min_width=308,
            elem_id="panel-lateral"
        ):
            gr.HTML(
                """
                <aside class="sidebar-retcc">
                    <div class="sidebar-superior">
                        <h2 class="titulo-recursos">RECURSOS OFICIALES</h2>

                        <a class="recurso-retcc" href="#">
                            <span class="recurso-icono">▤</span>
                            <span>Guía de Inscripción</span>
                        </a>

                        <a class="recurso-retcc" href="#">
                            <span class="recurso-icono">♢</span>
                            <span>Reglamento RETCC</span>
                        </a>

                        <a class="recurso-retcc" href="#">
                            <span class="recurso-icono">▣</span>
                            <span>Cronograma 2026</span>
                        </a>
                    </div>

                    <div class="sidebar-inferior">
                        <div class="tarjeta-ayuda">
                            <h3>¿Necesitas Ayuda?</h3>
                            <p>
                                Nuestro equipo de soporte está disponible
                                de L-V 8am-5pm.
                            </p>
                        </div>

                        <div class="usuario-retcc">
                            <div class="usuario-avatar">♙</div>
                            <div>
                                <div class="usuario-nombre">Juan Pérez</div>
                                <div class="usuario-rol">Trabajador Civil</div>
                            </div>
                        </div>
                    </div>
                </aside>
                """
            )

            boton_soporte = gr.Button(
                "Contactar Soporte",
                elem_id="boton-soporte"
            )

            boton_nuevo = gr.Button(
                "Nueva conversación",
                elem_id="boton-nuevo"
            )

        with gr.Column(
            scale=1,
            min_width=0,
            elem_id="zona-principal"
        ):

            gr.HTML(
                f"""
                <header class="cabecera-retcc">
                    <div class="cabecera-marca">

<div class="logo-retcc" aria-label="Casco de construcción">
    <img src="{CASCO_RETCC_SRC}" alt="Casco de construcción">
</div>

                        <h1 class="cabecera-titulo">Chat RETCC BOB</h1>
                    </div>
                    <div class="cabecera-info">ⓘ</div>
                </header>
                """,
                elem_id="cabecera-wrap"
            )

            gr.HTML(
                """
                <div class="estado-retcc">
                    <div>
                        Estado de Consulta:
                        <strong>Iniciando</strong>
                    </div>
                    <div class="barra-estado">
                        <span></span>
                    </div>
                </div>
                """,
                elem_id="estado-wrap"
            )

            with gr.Column(elem_id="contenido-retcc"):

                with gr.Column(elem_id="conversacion-scroll"):

                    gr.HTML(
                        f"""
                        <section id="bloque-bienvenida">
                            <div class="fila-asistente">
                                <div class="icono-asistente"><img src="{AVATAR_RETCC_SRC}" alt="Asistente virtual de construcción Civil"></div>

                                <div class="burbuja-bienvenida">
                                    ¡Hola! Soy RetccBob tu asistente virtual del
                                    <strong>
                                        Registro Nacional de Trabajadores de
                                        Construcción Civil (RETCC).
                                    </strong>

                                    <br><br>

                                    Estoy aquí para ayudarte con información
                                    oficial sobre trámites, requisitos y
                                    renovaciones de tu carné.
                                    ¿En qué puedo apoyarte hoy?
                                </div>
                            </div>
                        </section>
                        """
                    )

                    with gr.Row(elem_classes=["consultas-grid"]):
                        btn_requisitos = gr.Button(
                            "¿Requisitos para inscripción?",
                            elem_classes=["boton-rapido"]
                        )
                        btn_renovacion = gr.Button(
                            "¿Cómo renovar mi carné?",
                            elem_classes=["boton-rapido"]
                        )
                        btn_costo = gr.Button(
                            "¿Duplicado de carné en Lima Metropolitana?",
                            elem_classes=["boton-rapido"]
                        )

                    gr.HTML(
                        """
                        <section id="bloque-informativo">
                            <div class="fila-asistente">
                                <div class="icono-asistente"
                                     style="background:#c90008;color:white;border:none;">
                                    ⓘ
                                </div>

                                <div class="tarjeta-informativa">
                                    <div class="imagen-informativa">
                                        <strong>Información Importante</strong>
                                    </div>

                                    <div class="contenido-informativo">
                                        <p>
                                            Recuerde que la inscripción en el RETCC
                                            es obligatoria para todo trabajador que
                                            realice labores en construcción civil.
                                        </p>

                                        <div class="lista-informativa">
                                            <span>Validez a nivel nacional</span>
                                            <span>Vigencia de 2 años</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </section>
                        """
                    )

                    chat_html = gr.HTML(
                        value=renderizar_chat_html([]),
                        elem_id="chat-html-retcc"
                    )

                with gr.Column(elem_id="zona-entrada"):
                    with gr.Row(elem_id="fila-entrada"):
                        entrada_mensaje = gr.Textbox(
                            placeholder="Escribe tu consulta aquí...",
                            show_label=False,
                            lines=1,
                            max_lines=4,
                            elem_id="entrada-mensaje",
                            scale=10,
                            container=False
                        )

                        boton_enviar = gr.Button(
                            "➤",
                            elem_id="boton-enviar",
                            scale=0
                        )

                    gr.HTML(
                        """
                        <div class="pie-retcc">
                            Copyright © [Desarrollado by Arturo Guerrero - 2026]
                        </div>
                        """
                    )
    # =====================================================
    # ENVÍO MEDIANTE EL BOTÓN
    # =====================================================
    #
    # Paso 1:
    # Muestra inmediatamente la pregunta del usuario.
    #
    # Paso 2:
    # Ejecuta LangGraph, FAISS y Gemini.
    # =====================================================

    evento_boton_enviar = boton_enviar.click(
        fn=mostrar_pregunta_usuario_html,
        inputs=[
            entrada_mensaje,
            historial_estado,
            session_id
        ],
        outputs=[
            entrada_mensaje,
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden",
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )

    evento_respuesta_boton = evento_boton_enviar.then(
        fn=generar_respuesta_asistente_html,
        inputs=[
            pregunta_pendiente,
            historial_estado,
            session_id
        ],
        outputs=[
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden"
    )

    evento_respuesta_boton.then(
        fn=None,
        inputs=None,
        outputs=None,
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )


    # =====================================================
    # ENVÍO PRESIONANDO ENTER
    # =====================================================

    evento_enter = entrada_mensaje.submit(
        fn=mostrar_pregunta_usuario_html,
        inputs=[
            entrada_mensaje,
            historial_estado,
            session_id
        ],
        outputs=[
            entrada_mensaje,
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden",
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )

    evento_respuesta_enter = evento_enter.then(
        fn=generar_respuesta_asistente_html,
        inputs=[
            pregunta_pendiente,
            historial_estado,
            session_id
        ],
        outputs=[
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden"
    )

    evento_respuesta_enter.then(
        fn=None,
        inputs=None,
        outputs=None,
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )


    # =====================================================
    # NUEVA CONVERSACIÓN
    # =====================================================

    boton_nuevo.click(
        fn=nueva_conversacion_html,
        inputs=[
            session_id
        ],
        outputs=[
            chat_html,
            historial_estado,
            session_id,
            entrada_mensaje,
            pregunta_pendiente
        ],
        show_progress="hidden",
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )


    # =====================================================
    # CONSULTA RÁPIDA: REQUISITOS DE INSCRIPCIÓN
    # =====================================================

    evento_requisitos = btn_requisitos.click(
        fn=lambda historial, sesion: mostrar_consulta_rapida_html(
            pregunta=(
                "¿Cuáles son los requisitos para "
                "inscribirme en el RETCC?"
            ),
            historial_estado=historial,
            session_id=sesion
        ),
        inputs=[
            historial_estado,
            session_id
        ],
        outputs=[
            entrada_mensaje,
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden",
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )

    evento_respuesta_requisitos = evento_requisitos.then(
        fn=generar_respuesta_asistente_html,
        inputs=[
            pregunta_pendiente,
            historial_estado,
            session_id
        ],
        outputs=[
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden"
    )

    evento_respuesta_requisitos.then(
        fn=None,
        inputs=None,
        outputs=None,
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )


    # =====================================================
    # CONSULTA RÁPIDA: RENOVACIÓN
    # =====================================================

    evento_renovacion = btn_renovacion.click(
        fn=lambda historial, sesion: mostrar_consulta_rapida_html(
            pregunta=(
                "¿Cómo puedo renovar mi carné RETCC?"
            ),
            historial_estado=historial,
            session_id=sesion
        ),
        inputs=[
            historial_estado,
            session_id
        ],
        outputs=[
            entrada_mensaje,
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden",
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )

    evento_respuesta_renovacion = evento_renovacion.then(
        fn=generar_respuesta_asistente_html,
        inputs=[
            pregunta_pendiente,
            historial_estado,
            session_id
        ],
        outputs=[
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden"
    )

    evento_respuesta_renovacion.then(
        fn=None,
        inputs=None,
        outputs=None,
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )


    # =====================================================
    # CONSULTA RÁPIDA: DUPLICADO EN LIMA METROPOLITANA
    # =====================================================

    evento_costo = btn_costo.click(
        fn=lambda historial, sesion: mostrar_consulta_rapida_html(
            pregunta=(
                "¿Cuál es el costo del duplicado del carné "
                "RETCC en Lima Metropolitana?"
            ),
            historial_estado=historial,
            session_id=sesion
        ),
        inputs=[
            historial_estado,
            session_id
        ],
        outputs=[
            entrada_mensaje,
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden",
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )

    evento_respuesta_costo = evento_costo.then(
        fn=generar_respuesta_asistente_html,
        inputs=[
            pregunta_pendiente,
            historial_estado,
            session_id
        ],
        outputs=[
            chat_html,
            historial_estado,
            session_id,
            pregunta_pendiente
        ],
        show_progress="hidden"
    )

    evento_respuesta_costo.then(
        fn=None,
        inputs=None,
        outputs=None,
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )
    # =====================================================
    # BOTÓN DE SOPORTE
    # =====================================================

    boton_soporte.click(
        fn=lambda: (
            "Necesito orientación adicional sobre "
            "mi trámite RETCC."
        ),
        inputs=None,
        outputs=[
            entrada_mensaje
        ],
        show_progress="hidden",
        js=JS_SCROLL_FINAL_RETCC,
        queue=False
    )


# =========================================================
# INICIO DEL SERVICIO HTTP PARA CLOUD RUN
# =========================================================

if __name__ == "__main__":
    puerto = int(os.getenv("PORT", "8080"))

    favicon = (RUTA_ASSETS_RETCC / "casco.svg").resolve()

    interfaz_retcc.launch(
        server_name="0.0.0.0",
        server_port=puerto,
        share=False,
        inline=False,
        debug=False,
        show_error=True,
        favicon_path=str(favicon)
    )

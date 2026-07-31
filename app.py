"""
Plataforma OCR -> Identificación con IA (Groq) -> Web Scraping -> EDA -> Feature Engineering
=============================================================================================

Flujo:
1. El usuario sube una imagen (figura, objeto, pieza, etiqueta, etc.)
2. Se ejecuta OCR (pytesseract) para extraer el texto de la imagen.
3. El texto (y contexto) se envía a un modelo de Groq para identificar qué es.
4. Con la identificación, se hace web scraping en fuentes libres/gratuitas
   (DuckDuckGo HTML, Wikipedia) sin necesidad de API keys de pago.
5. Se muestra un EDA (Análisis Exploratorio de Datos) de lo scrapeado.
6. Se generan features (feature engineering) a partir del texto scrapeado.

Requisitos del sistema (fuera de pip):
- Tesseract OCR debe estar instalado en el sistema operativo:
    Ubuntu/Debian:  sudo apt-get install tesseract-ocr tesseract-ocr-spa
    Mac:            brew install tesseract tesseract-lang
    Windows:        https://github.com/UB-Mannheim/tesseract/wiki
"""

import io
import re
import json
import shutil
import string
from collections import Counter
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from bs4 import BeautifulSoup
from PIL import Image
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    import pytesseract
    TESSERACT_OK = shutil.which("tesseract") is not None
except ImportError:
    TESSERACT_OK = False

try:
    from groq import Groq
except ImportError:
    Groq = None


# ------------------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="OCR + IA + Web Scraping + EDA",
    page_icon="🔎",
    layout="wide",
)

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

STOPWORDS_ES_EN = set(
    """
    de la que el en y a los del se las por un para con no una su al lo como más
    pero sus le ya o este sí porque esta entre cuando muy sin sobre también me
    hasta hay donde quien desde todo nos durante todos uno les ni contra otros
    ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él
    tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas
    algo nosotros mi mis tú te ti tu tus ellas nosotras vosotros vosotras os
    the a an and or of in on for to with is are was were be been being this
    that these those it its as at by from into over under about than then
    """.split()
)


# ------------------------------------------------------------------------------------
# ESTADO DE SESIÓN
# ------------------------------------------------------------------------------------
def init_state():
    defaults = {
        "ocr_text": "",
        "identification": None,
        "scraped_df": None,
        "features_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ------------------------------------------------------------------------------------
# 1. OCR
# ------------------------------------------------------------------------------------
def run_ocr(image: Image.Image, lang: str = "spa+eng") -> str:
    if not TESSERACT_OK:
        raise RuntimeError(
            "El binario 'tesseract' no está instalado en el sistema. "
            "Instálalo con: sudo apt-get install tesseract-ocr tesseract-ocr-spa"
        )
    return pytesseract.image_to_string(image, lang=lang)


# ------------------------------------------------------------------------------------
# 2. IDENTIFICACIÓN CON GROQ
# ------------------------------------------------------------------------------------
def identify_with_groq(api_key: str, model: str, ocr_text: str, user_hint: str = "") -> dict:
    client = Groq(api_key=api_key)

    system_prompt = (
        "Eres un experto identificador de objetos, piezas, productos, especies, "
        "componentes, letreros o documentos a partir de texto extraído por OCR, "
        "que puede venir con errores tipográficos propios del OCR. "
        "Responde SIEMPRE en formato JSON válido, sin texto adicional, con las claves: "
        '{"identificacion": str, "categoria": str, "descripcion_breve": str, '
        '"palabras_clave": [str, ...], "consultas_busqueda": [str, ...]}. '
        "Las 'consultas_busqueda' deben ser 3 a 5 queries cortas y efectivas para buscar "
        "en un motor de búsqueda información relevante sobre lo identificado."
    )

    user_prompt = f"Texto OCR extraído de la imagen:\n---\n{ocr_text}\n---\n"
    if user_hint:
        user_prompt += f"\nContexto adicional dado por el usuario: {user_hint}\n"
    user_prompt += "\nIdentifica qué es y responde solo con el JSON solicitado."

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    raw = completion.choices[0].message.content.strip()
    raw = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            data = {
                "identificacion": raw[:200],
                "categoria": "desconocida",
                "descripcion_breve": raw[:400],
                "palabras_clave": [],
                "consultas_busqueda": [ocr_text[:60]] if ocr_text else [],
            }
    return data


def groq_text_call(api_key: str, model: str, system_prompt: str, user_prompt: str, max_tokens=300) -> str:
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


# ------------------------------------------------------------------------------------
# 3. WEB SCRAPING (fuentes libres, sin API key)
# ------------------------------------------------------------------------------------
def resolve_ddg_url(href: str) -> str:
    """DuckDuckGo HTML (versión sin JS) a veces no devuelve el link directo sino
    un enlace de redirección propio, del tipo //duckduckgo.com/l/?uddg=<url_real>&rut=...
    Esta función detecta ese patrón y devuelve la URL real (decodificada), para que
    el dominio calculado más adelante (columna 'dominio') refleje la fuente real y
    no siempre 'duckduckgo.com'. Si el href ya es un link directo, se devuelve tal cual."""
    if not href:
        return href
    absolute = href if href.startswith("http") else f"https:{href}"
    parsed = urlparse(absolute)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        real_url = qs.get("uddg", [None])[0]
        if real_url:
            return unquote(real_url)
    return absolute


def scrape_duckduckgo(query: str, max_results: int = 8) -> list:
    """Scrapea resultados de DuckDuckGo HTML (versión sin JS), de acceso libre."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    results = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        blocks = soup.select("div.result")[:max_results]
        for b in blocks:
            title_tag = b.select_one("a.result__a")
            snippet_tag = b.select_one("a.result__snippet") or b.select_one(".result__snippet")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            link = resolve_ddg_url(title_tag.get("href", ""))
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            results.append({
                "query": query,
                "fuente": "DuckDuckGo",
                "titulo": title,
                "url": link,
                "snippet": snippet,
            })
    except Exception as e:
        st.warning(f"No se pudo scrapear DuckDuckGo para '{query}': {e}")
    return results


def scrape_wikipedia(query: str) -> list:
    """Usa la API pública y gratuita de Wikipedia (sin key) para obtener un resumen."""
    results = []
    try:
        search_url = "https://es.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        }
        r = requests.get(search_url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        for h in hits:
            title = h.get("title", "")
            snippet = re.sub("<.*?>", "", h.get("snippet", ""))
            page_url = f"https://es.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            results.append({
                "query": query,
                "fuente": "Wikipedia",
                "titulo": title,
                "url": page_url,
                "snippet": snippet,
            })
    except Exception as e:
        st.warning(f"No se pudo consultar Wikipedia para '{query}': {e}")
    return results


@st.cache_data(show_spinner=False, ttl=1800)
def run_scraping(queries: list, max_results_per_query: int = 6) -> pd.DataFrame:
    all_rows = []
    progress = st.progress(0.0, text="Iniciando scraping...")
    total = max(len(queries), 1)
    for i, q in enumerate(queries):
        progress.progress((i) / total, text=f"Scrapeando: {q}")
        all_rows.extend(scrape_duckduckgo(q, max_results=max_results_per_query))
        all_rows.extend(scrape_wikipedia(q))
    progress.progress(1.0, text="Scraping finalizado")
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["dominio"] = df["url"].apply(lambda u: urlparse(u).netloc if u else "")
        df.drop_duplicates(subset=["titulo", "url"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


# ------------------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------------------------------
def clean_tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t.isalpha() and t not in STOPWORDS_ES_EN and len(t) > 2]
    return tokens


@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    fdf = df.copy()
    fdf["snippet"] = fdf["snippet"].fillna("")
    fdf["titulo"] = fdf["titulo"].fillna("")

    # Features textuales básicas
    fdf["longitud_snippet"] = fdf["snippet"].str.len()
    fdf["num_palabras_snippet"] = fdf["snippet"].apply(lambda s: len(s.split()))
    fdf["longitud_titulo"] = fdf["titulo"].str.len()
    fdf["num_digitos"] = fdf["snippet"].apply(lambda s: sum(c.isdigit() for c in s))
    fdf["tiene_precio"] = fdf["snippet"].str.contains(
        r"(\$|USD|EUR|€|precio|price)", case=False, regex=True, na=False
    )
    fdf["tiene_numero"] = fdf["num_digitos"] > 0
    fdf["num_mayusculas_titulo"] = fdf["titulo"].apply(lambda s: sum(c.isupper() for c in s))

    # Extraer posibles cifras numéricas (dimensiones, precios, años, etc.)
    def extract_numbers(s):
        nums = re.findall(r"\d+[\.,]?\d*", s)
        nums = [float(n.replace(",", ".")) for n in nums if n]
        return nums

    fdf["valores_numericos"] = fdf["snippet"].apply(extract_numbers)
    fdf["max_valor_numerico"] = fdf["valores_numericos"].apply(lambda x: max(x) if x else np.nan)
    fdf["cantidad_valores_numericos"] = fdf["valores_numericos"].apply(len)

    return fdf


@st.cache_data(show_spinner=False)
def top_keywords(corpus: list, top_n: int = 20) -> pd.DataFrame:
    all_tokens = []
    for text in corpus:
        all_tokens.extend(clean_tokenize(text))
    counts = Counter(all_tokens)
    common = counts.most_common(top_n)
    return pd.DataFrame(common, columns=["palabra", "frecuencia"])


@st.cache_data(show_spinner=False)
def build_wordcloud_image(corpus: list):
    """Genera la nube de palabras a partir del corpus. Cacheada porque generar el
    WordCloud es la parte más costosa del EDA y antes se recalculaba en cada
    rerun de Streamlit (cualquier clic en la app), no solo cuando cambiaban
    los datos scrapeados. Devuelve None si no hay tokens suficientes."""
    tokens = clean_tokenize(" ".join(corpus))
    if not tokens:
        return None
    return WordCloud(width=900, height=400, background_color="white").generate(" ".join(tokens))


# ------------------------------------------------------------------------------------
# INTERFAZ - SIDEBAR
# ------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
api_key = st.sidebar.text_input("Groq API Key", type="password", help="Tu API Key de Groq (console.groq.com)")
model = st.sidebar.selectbox("Modelo de Groq", GROQ_MODELS, index=0)
ocr_lang = st.sidebar.text_input("Idioma OCR (tesseract)", value="spa+eng")
max_results_per_query = st.sidebar.slider("Resultados por consulta (scraping)", 2, 15, 6)

if not TESSERACT_OK:
    st.sidebar.error(
        "⚠️ No se detectó el binario 'tesseract' en el sistema. "
        "El OCR no funcionará hasta instalarlo (ver docstring de app.py)."
    )

if Groq is None:
    st.sidebar.error("⚠️ El paquete 'groq' no está instalado. Ejecuta: pip install groq")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Fuentes de scraping usadas: DuckDuckGo (HTML, sin key) y Wikipedia (API pública, sin key)."
)


# ------------------------------------------------------------------------------------
# INTERFAZ - TÍTULO
# ------------------------------------------------------------------------------------
st.title("🔎 OCR → Identificación IA → Web Scraping → EDA → Feature Engineering")
st.write(
    "Sube una imagen, extrae su texto con OCR, identifica de qué se trata usando un "
    "modelo de Groq, busca información relacionada en la web y analiza los datos obtenidos."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["1️⃣ OCR + Identificación", "2️⃣ Web Scraping", "3️⃣ EDA", "4️⃣ Feature Engineering"]
)


# ------------------------------------------------------------------------------------
# TAB 1: OCR + IDENTIFICACIÓN
# ------------------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        uploaded_file = st.file_uploader(
            "Sube la imagen de la figura/objeto", type=["png", "jpg", "jpeg", "bmp", "webp"]
        )
        user_hint = st.text_input(
            "Contexto adicional (opcional)",
            placeholder="Ej: es una pieza mecánica, una etiqueta de producto, una planta...",
        )

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen cargada", use_container_width=True)

            if st.button("🔍 Ejecutar OCR", type="primary", use_container_width=True):
                try:
                    with st.spinner("Extrayendo texto de la imagen..."):
                        text = run_ocr(image, lang=ocr_lang)
                    st.session_state.ocr_text = text.strip()
                    if not text.strip():
                        st.info("El OCR no encontró texto legible en la imagen.")
                except Exception as e:
                    st.error(f"Error ejecutando OCR: {e}")

    with col2:
        st.subheader("Texto extraído (editable)")
        st.session_state.ocr_text = st.text_area(
            "Puedes corregir errores del OCR antes de identificar:",
            value=st.session_state.ocr_text,
            height=180,
        )

        identify_disabled = not api_key or Groq is None
        if st.button("🤖 Identificar con IA (Groq)", disabled=identify_disabled, use_container_width=True):
            if not st.session_state.ocr_text and not user_hint:
                st.warning("No hay texto OCR ni contexto para identificar. Agrega alguno.")
            else:
                try:
                    with st.spinner(f"Consultando modelo {model} en Groq..."):
                        data = identify_with_groq(api_key, model, st.session_state.ocr_text, user_hint)
                    st.session_state.identification = data
                except Exception as e:
                    st.error(f"Error llamando a Groq: {e}")

        if identify_disabled:
            st.caption("Ingresa tu Groq API Key en la barra lateral para habilitar la identificación.")

        if st.session_state.identification:
            data = st.session_state.identification
            st.success(f"**Identificación:** {data.get('identificacion', 'N/D')}")
            st.markdown(f"**Categoría:** {data.get('categoria', 'N/D')}")
            st.markdown(f"**Descripción:** {data.get('descripcion_breve', 'N/D')}")
            kws = data.get("palabras_clave", [])
            if kws:
                st.markdown("**Palabras clave:** " + ", ".join(kws))
            queries = data.get("consultas_busqueda", [])
            if queries:
                st.markdown("**Consultas sugeridas para scraping:**")
                for q in queries:
                    st.markdown(f"- {q}")
            with st.expander("Ver JSON completo"):
                st.json(data)


# ------------------------------------------------------------------------------------
# TAB 2: WEB SCRAPING
# ------------------------------------------------------------------------------------
with tab2:
    st.subheader("Búsqueda y scraping en la web")

    default_queries = []
    if st.session_state.identification:
        default_queries = st.session_state.identification.get("consultas_busqueda", [])
        if not default_queries:
            ident = st.session_state.identification.get("identificacion", "")
            default_queries = [ident] if ident else []

    queries_text = st.text_area(
        "Consultas a buscar (una por línea). Se autocompletan con las sugeridas por la IA:",
        value="\n".join(default_queries),
        height=120,
    )

    if st.button("🌐 Ejecutar Web Scraping", type="primary"):
        queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
        if not queries:
            st.warning("Agrega al menos una consulta de búsqueda.")
        else:
            df = run_scraping(queries, max_results_per_query=max_results_per_query)
            st.session_state.scraped_df = df
            st.session_state.features_df = None  # reset features al re-scrapear

    if st.session_state.scraped_df is not None:
        df = st.session_state.scraped_df
        st.write(f"**{len(df)} resultados obtenidos.**")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", data=csv, file_name="scraping_resultados.csv", mime="text/csv")
    else:
        st.info("Aún no se ha ejecutado ningún scraping.")


# ------------------------------------------------------------------------------------
# TAB 3: EDA
# ------------------------------------------------------------------------------------
with tab3:
    st.subheader("Análisis Exploratorio de Datos (EDA)")

    df = st.session_state.scraped_df
    if df is None or df.empty:
        st.info("Primero ejecuta el web scraping en la pestaña anterior.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de resultados", len(df))
        c2.metric("Dominios únicos", df["dominio"].nunique())
        c3.metric("Consultas distintas", df["query"].nunique())
        c4.metric("Longitud media snippet", f"{df['snippet'].fillna('').str.len().mean():.0f} car.")

        st.markdown("#### Resultados por consulta")
        vc_query = df["query"].value_counts().rename_axis("consulta").reset_index(name="conteo")
        fig1 = px.bar(
            vc_query,
            x="consulta", y="conteo", title="Cantidad de resultados por consulta"
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("#### Distribución de dominios (fuentes)")
        top_dominios = df["dominio"].value_counts().head(15).rename_axis("dominio").reset_index(name="conteo")
        fig2 = px.bar(top_dominios, x="conteo", y="dominio", orientation="h", title="Top dominios encontrados")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Distribución de longitud de los snippets")
        fig3 = px.histogram(df, x=df["snippet"].fillna("").str.len(), nbins=20,
                             labels={"x": "Longitud del snippet (caracteres)"})
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Nube de palabras")
        corpus = df["snippet"].fillna("").tolist() + df["titulo"].fillna("").tolist()
        wc = build_wordcloud_image(corpus)
        if wc is not None:
            fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
            plt.close(fig_wc)  # evita acumular figuras de matplotlib en memoria entre reruns
        else:
            st.info("No hay suficiente texto para generar la nube de palabras.")

        st.markdown("#### Palabras clave más frecuentes")
        kw_df = top_keywords(df["snippet"].fillna("").tolist() + df["titulo"].fillna("").tolist(), top_n=20)
        if not kw_df.empty:
            fig4 = px.bar(kw_df, x="frecuencia", y="palabra", orientation="h",
                          title="Top 20 palabras más frecuentes")
            fig4.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig4, use_container_width=True)

        with st.expander("Ver tabla completa de datos scrapeados"):
            st.dataframe(df, use_container_width=True)


# ------------------------------------------------------------------------------------
# TAB 4: FEATURE ENGINEERING
# ------------------------------------------------------------------------------------
with tab4:
    st.subheader("Feature Engineering sobre los datos scrapeados")

    df = st.session_state.scraped_df
    if df is None or df.empty:
        st.info("Primero ejecuta el web scraping.")
    else:
        if st.button("🛠️ Generar features"):
            with st.spinner("Generando variables derivadas..."):
                st.session_state.features_df = build_features(df)

        fdf = st.session_state.features_df
        if fdf is not None:
            st.markdown("#### Tabla de features generadas")
            st.dataframe(fdf, use_container_width=True)

            numeric_cols = fdf.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.markdown("#### Estadística descriptiva de features numéricas")
                st.dataframe(fdf[numeric_cols].describe(), use_container_width=True)

                st.markdown("#### Matriz de correlación")
                corr = fdf[numeric_cols].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                plt.close(fig)  # evita acumular figuras de matplotlib en memoria entre reruns

            st.markdown("#### Proporción de resultados con precio detectado")
            if "tiene_precio" in fdf.columns:
                fig5 = px.pie(fdf, names="tiene_precio", title="¿Contiene mención de precio?")
                st.plotly_chart(fig5, use_container_width=True)

            csv_feat = fdf.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar CSV con features", data=csv_feat,
                                file_name="features.csv", mime="text/csv")

            # Feature de IA opcional: clasificación de relevancia/sentimiento con Groq
            st.markdown("---")
            st.markdown("#### 🤖 Feature adicional generada con IA (opcional)")
            n_ai = st.slider("Cantidad de snippets a analizar con IA (para no exceder límites de la API)",
                              0, min(20, len(fdf)), min(5, len(fdf)))
            if st.button("Generar feature de relevancia/sentimiento con Groq", disabled=(not api_key or n_ai == 0)):
                relevancias = []
                subset = fdf.head(n_ai)
                progress = st.progress(0.0)
                for i, row in enumerate(subset.itertuples()):
                    try:
                        resp = groq_text_call(
                            api_key, model,
                            system_prompt=(
                                "Clasifica el siguiente texto en una sola palabra: "
                                "'positivo', 'negativo' o 'neutral', según su tono. "
                                "Responde solo con esa palabra."
                            ),
                            user_prompt=row.snippet[:500] if row.snippet else row.titulo[:200],
                            max_tokens=5,
                        )
                        relevancias.append(resp.strip().lower())
                    except Exception as e:
                        relevancias.append("error")
                    progress.progress((i + 1) / max(n_ai, 1))
                subset = subset.copy()
                subset["tono_ia"] = relevancias
                st.dataframe(subset[["titulo", "snippet", "tono_ia"]], use_container_width=True)
        else:
            st.caption("Presiona 'Generar features' para construir las variables derivadas.")

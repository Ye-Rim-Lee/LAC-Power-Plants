# ⚡️ Classifying Power Plants Using AI & Fuzzy Matching

## Objective
This project automates the classification of Ecuador's power plants by technology and subtype (e.g., Reservoir, Run-of-the-River, Diesel, Natural Gas, Solar, Wind).
The pipeline combines **PDF table extraction, fuzzy string matching,** and **LLM-based text classification** to turn unstructured public data into a clean, research-ready dataset for downstream energy analytics.


## Workflow Overview
**Step 1 — Reference Extraction**
- Parses official country‐specific registries (PDFs, spreadsheets, or web archives).
- Uses **Camelot** (`flavor="lattice"`) for PDF table extraction when applicable.

**Step 2 — Name Normalization**
- Cleans company and plant names via `unidecode`, removes accents, trims spaces, and lowercases text.

**Step 3 — Direct + Fuzzy Matching**
- Performs deterministic joins on normalized names.
- For unmatched entries, applies **RapidFuzz** (`fuzz.partial_ratio`) within same technology category.

**Step 4 — AI-Assisted Classification (Hydro Subtype)**
- For hydro plants lacking subtype, performs Spanish **web searches** via **Bing API**.
- Extracts snippets with **BeautifulSoup** and classifies as _Reservoir_ or _Run-of-the-River_ using **OpenAI GPT-4.1-mini**, accepting results only if model confidence ≥ 0.88.

**Step 5 — Standardization & Export**
- Maps each subtype to a unified numeric schema shared across countries.
- Outputs standardized `.dta` and `.xlsx` files for cross-country integration.

## Key Challenges & Mitigations

| Challenge                       | Description                                            | Solution                                                                      |
| ------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| **PDF Table Fragmentation**     | Multi-page tabular structures with merged headers      | Combined Camelot tables and programmatically removed repeated headers         |
| **Accent / Case Inconsistency** | Plant names vary across sources                        | Built normalization function (`norm()`) with `unidecode` + lowercase trimming |
| **Missing Subtype Labels**      | Many hydro plants lacked “reservoir/run-of-river” info | Automated web retrieval + LLM text classification in Spanish                  |
| **Confidence Control**          | LLM misclassification risk                             | Required ≥ 0.88 confidence; flagged others for manual review                  |
| **Cross-language Sources**      | Spanish context with English schema                    | Dual-language tokenization and context injection into prompts                 |


## Minimal Example
**Goal**: Harmonize Ecuador’s national registry with existing ECU_charac.dta to add precise subtype labels.

```python
# Step 0: Extract reference table from PME.pdf
tables = cm.read_pdf(
    "../../../0_raw/07_Ecuador/plantcharact/PME.pdf",
    pages="44-55", flavor="lattice"
)
df_all = pd.concat([t.df for t in tables], ignore_index=True)

# Step 1: Normalize plant and company names
def norm(s): 
    s = unidecode(str(s)).replace("\u00a0"," ")
    return " ".join(s.lower().strip().split())

# Step 2: Fuzzy match remaining hydro plants
match = process.extractOne(name, subset["plant_name"], scorer=fuzz.partial_ratio)
if match and match[1] >= 90:
    subtype = subset.loc[subset["plant_name"] == match[0], "source2"].iloc[0]
```

**AI Classification Example**
```python
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)
MODEL = "gpt-4.1-mini"

def llm_classify_hydro(name: str, context_es: str = ""):
    """
    Classify an Ecuadorian hydro plant into: 'Reservoir' or 'Run-of-the-river'.
    Input: plant 'name' (any language), 'context_es' (Spanish snippets).
    Output: (label, confidence) or (None, 0.0) if unknown/uncertain.
    """
    prompt = f"""
    You are a technical assistant. Classify the Ecuadorian hydroelectric plant.

    Options EXACT: ["Reservoir","Run-of-the-river"].
    If the evidence is ambiguous or insufficient, reply "unknown".
    Return JSON only: {{"label":"...","confidence":0-1}}.

    Plant: {name}
    Context (Spanish): {context_es[:1500]}
    """
    try:
        resp = client.responses.create(model=MODEL, input=prompt, temperature=0)
        out = resp.output_text

        import re, json
        m = re.search(r"\{.*?\}", out, re.S)   # non-greedy JSON grab
        if not m:
            return None, 0.0

        j = json.loads(m.group(0))
        label = j.get("label")
        conf = float(j.get("confidence", 0))

        if label in ("Reservoir", "Run-of-the-river"):
            return label, conf
        return None, 0.0
    except Exception:
        return None, 0.0
```

---

## Project Context
This code is part of the ENE Knowledge Team (IDB Energy Division, 2025) led by Lenin H. Balza, aiming to expand the Chile drought-generation study into a regional analysis of how climate shocks reshape Latin America’s power mix.
This work contributes to the regional research paper **“Droughts and the Energy Transition in Latin America”**, which investigates how climate-induced water scarcity alters energy production patterns.
My specific task ensured that Ecuador’s generation plants were consistently categorized by technology and subtype, allowing econometric models to accurately measure hydropower sensitivity to drought.

Authors: **Ye-Rim Lee**, Lenin H. Balza and José Belmar <br>
Contact: leeye12@msu.edu


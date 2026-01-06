# AlignCultura

## CulturaX Class Index Mapping

| Class ID | Class Name                         |
|:--------:|------------------------------------|
| 0  | Artifacts                          |
| 1  | Monuments                          |
| 2  | Museums                            |
| 3  | Historical and Archeological Sites |
| 4  | National Parks                     |
| 6  | Botanical Gardens                  |
| 7  | Marine Ecosystems                  |
| 8  | Cultural Landscapes                |
| 9  | Libraries                          |
| 10 | Language                           |
| 11 | Culinary Arts                      |
| 12 | Crafts                             |
| 13 | Bio-Cultural Practices             |
| 14 | Folk Sports                        |
| 15 | Festivals                          |
| 16 | Film and Video                     |
| 17 | TV                                 |
| 19 | Festivals and Markets              |
| 20 | Theatrical Performance             |
| 21 | Dance                              |
| 22 | Opera                              |
| 23 | Graphic Design                     |
| 24 | Fashion Design                     |
| 25 | Industrial Design                  |
| 26 | Architectural Services             |
| 27 | Interior Design                    |
| 28 | Fine Arts                          |
| 29 | Photography                        |
| 32 | Live Music                         |
| 33 | Musical Instruments                |
| 34 | Books                              |
| 35 | Newspapers                         |
| 37 | Social Networks                    |
| 38 | Blogs                              |
| 39 | Video Games                        |
| 41 | None                               |

## CulturaX Class Distribution

| Class Name                          | Our Class Samples |
|------------------------------------|-------------------:|
| Architectural Services             | 7   |
| Bio-Cultural Practices             | 10  |
| Blogs                              | 15  |
| Books                              | 5   |
| Crafts                             | 108 |
| Culinary Arts                      | 100 |
| Cultural Landscapes                | 2   |
| Dance                              | 38  |
| Fashion Design                     | 23  |
| Festivals                          | 387 |
| Festivals and Markets              | 384 |
| Film and Video                     | 9   |
| Fine Arts                          | 3   |
| Folk Sports                        | 54  |
| Historical and Archeological Sites | 1   |
| Industrial Design                  | 1   |
| Interior Design                    | 5   |
| Language                           | 99  |
| Libraries                          | 2   |
| Magazines              | 3   |
| Musical Instruments                | 1   |
| National Parks                     | 2   |
| Newspapers                         | 4   |
| Opera                              | 1   |
| Radio                  | 2   |
| Social Networks                    | 40  |
| TV                                 | 14  |
| Theatrical Performance             | 170 |
| Video Games                        | 9   |
| Zoos and Aquariums     | 1   |


### Data Construction Overview

| Step | Module | Model / Method | Purpose |
|-----:|--------|----------------|---------|
| 1 | Query Classification | Mistral-7B-Instruct | Multi-label UFCS domain assignment. |
| 2 | Domain Expansion | Llama-3.1-8B-Instruct | Balance underrepresented cultural domains. |
| 3 | Deduplication | SimHash (τ = 10) | Prevent near-duplicates and data leakage. |
| 4 | Response Generation | GPT-4.1 | Generate culturally grounded candidate responses. |
| 5 | HHH Filtering | Llama-3.1-8B-Instruct | Enforce Helpful, Harmless, and Honest (HHH) criteria. |

**Note on model reuse.**  
Although **Llama-3.1-8B-Instruct** is used in two stages, its roles are strictly separated. In **Query Construction (Stage I)**, it operates as a *controlled prompt generator* for expanding underrepresented UFCS domains. In **Response Generation (Stage I)**, it acts solely as an *HHH-Quality Model* that critiques and filters responses rather than generating final content.

Furthermore, response generation is performed by an independent model (**GPT-4.1**), ensuring that no model is responsible for both producing and scoring the same output. This separation mitigates circular bias and preserves the integrity of HHH evaluation.


### Dataset Statistics

| Attribute | Value |
|----------|-------|
| Total instances | 1,500 |
| Language | English |
| High-level cultural domains | 9 (UNESCO UFCS) |
| Cultural subdomains | 30 |
| Cultural forms | Tangible & Intangible |
| Source prompts | Cultural Kaleidoscope |
| Deduplication method | SimHash |
| Cross-split leakage | 0.3% |
| Train / Val / Test split | 80% / 10% / 10% |

### Models Used

| Category | Model | Role |
|---------|-------|------|
| Classification | Mistral-7B-Instruct | UFCS multi-label domain classification. |
| Expansion | Llama-3.1-8B-Instruct | Query expansion for underrepresented cultural domains. |
| Generation | GPT-4.1 | Culturally grounded response generation. |
| HHH Evaluation | Llama-3.1-8B-Instruct | Automated Helpful–Harmless–Honest (HHH) quality assessment. |
| Benchmarking (General-Purpose) | MARL-Focal | Joint-dimension HHH-aligned baseline. |
| Benchmarking (General-Purpose) | TrinityX | Joint-dimension HHH-aligned baseline. |
| Benchmarking (General-Purpose) | H³Fusion | Joint-dimension HHH-aligned baseline. |
| Benchmarking (Cultural) | CultureLLM | Culturally fine-tuned alignment baseline. |
| Benchmarking (Cultural) | CulturePark | Culturally fine-tuned alignment baseline. |
| Benchmarking (Open-Weight) | Qwen3-8B | Open-weight evaluation baseline. |
| Benchmarking (Open-Weight) | DeepSeek-R1-Distill-Qwen-7B | Open-weight evaluation baseline. |

### Model Selection Rationale

AlignCultura evaluates models across three categories to ensure fair and meaningful comparison:

- **General-Purpose Aligned Models**  
  Only *joint-dimension HHH alignment* methods are included (MARL-Focal, TrinityX, H³Fusion).  
  Single-dimension models (e.g., helpfulness-only or safety-only) are excluded, as they optimize isolated objectives and fail to capture the cross-dimension trade-offs required for cultural alignment.

- **Culturally Fine-Tuned Models**  
  CultureLLM and CulturePark are evaluated as representative approaches that explicitly adapt LLMs using culturally annotated data or structured cultural norms, enabling improved sensitivity to cultural context.

- **Open-Weight LLMs**  
  Qwen3-8B and DeepSeek-R1-Distill-Qwen-7B are included as strong mid-scale open-weight models without explicit cultural alignment. Only open-weight models are evaluated in this category to ensure reproducibility and controlled adaptation, as closed-source models do not permit parameter-level intervention.

Closed-source models (e.g., Claude-3 Opus, Gemini-2.5 Pro) are analyzed separately for reference. While highly capable, they are excluded from the main benchmarking comparisons due to limited reproducibility and lack of controllable alignment mechanisms. Notably, their strongest performance also emerges under joint HHH optimization, supporting the central hypothesis that culturally appropriate behavior arises from coordinated multi-objective alignment rather than isolated objective tuning.


### Key Hyperparameters

| Component | Parameter | Value |
|----------|----------|-------|
| Classification | Probability threshold (δ) | 0.5 |
| Deduplication | SimHash Hamming threshold (τ) | 10 |
| Generation | Temperature | 0.7 |
| Generation | Top-p | 0.9 |
| Generation | Max tokens | 512 |
| Generation | Candidates per prompt (K) | 3 |
| Feedback resampling | Max iterations | 2 |
| Training | Random seed | 42 |

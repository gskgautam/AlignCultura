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


## Evaluation Metrics

We adopt alignment-specific metrics from prior work —  
[Kashyap et al., 2025](https://arxiv.org/abs/2509.08486) and [Tekin et al., 2024](https://arxiv.org/abs/2411.17792) — that operationalize the **HHH (Helpfulness–Harmlessness–Honesty)** framework.  
Traditional metrics like accuracy or F1 fail to capture trade-offs between these axes.

### • Helpfulness (↑ better)
Measured using **Win Rate (WR)**:

$$
\mathrm{WR} = \frac{N_{\text{wins}}}{N_{\text{samples}}} \times 100
$$

---

### • Harmlessness (↓ better)
Evaluated using the **[Beaver-Dam-7B moderation model](https://huggingface.co/PKU-Alignment/beaver-dam-7b)**:

$$
\mathrm{SS} = \frac{N_{\text{unsafe}}}{N_{\text{samples}}} \times 100
$$

---

### • Honesty (↑ better)
Assessed via the **[GPT-Judge framework](https://github.com/kingoflolz/mesh-transformer-jax)**, combining truthfulness and informativeness:

$$
\mathrm{TI} = 
\left(
\frac{N_{\text{truthful}}}{N_{\text{samples}}}
\right)
\times
\left(
\frac{N_{\text{informative}}}{N_{\text{samples}}}
\right)
\times 100
$$

---

### • Average Alignment Score
To summarize overall alignment performance:

$$
\[
\mathrm{Avg}=\frac{\mathrm{WR}+\mathrm{TI}-(\mathrm{SS})}{3}
\]
$$

---

## Baselines

To contextualize results on **CulturaX**, we evaluate three baseline categories.

### 1️⃣ General-Purpose Aligned Models
We include **joint-axis HHH alignment** frameworks only, excluding single-axis methods (e.g., RAHF, Aligner).

- [MARL-Focal](https://arxiv.org/abs/2502.04492) — Multi-agent joint alignment  
- [TrinityX](https://arxiv.org/abs/2509.08486) — Multi-stage adaptive alignment  
- [H³Fusion](https://arxiv.org/abs/2411.17792) — Multi-objective HHH fusion

---

### 2️⃣ Culturally Fine-Tuned Models
We evaluate models explicitly adapted to cultural norms:

- **CultureLLM** — Instruction-tuned with culturally annotated data  
- **CulturePark** — Culture-aware alignment via structured norms  

These represent state-of-the-art baselines for **culturally grounded HHH alignment**.

---

### 3️⃣ Open-Weight LLMs
We benchmark strong open-weight models without cultural tuning:

- **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)**  
- **[DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)**  

---

## Experimental Results and Analysis

All experiments use **PyTorch 2.3** on **4× NVIDIA A100 (80GB)** GPUs with mixed precision and random seed `42`.

### Stage I – Generation
- Temperature: `0.7`  
- Top-p: `0.9`  
- Max length: `512`  
- Up to `K = 3` candidates per query  
- Max 2 feedback iterations

### Stage II – Evaluation
- Results averaged over **3 runs**  
- Decoding: temperature `0.7`, top-p `0.9`, max length `512`, repetition penalty `1.1`

### Dataset – CulturaX
- Total samples: `1500`  
- Split: `80% / 10% / 10%` (train / val / test)


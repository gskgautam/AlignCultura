# AlignCultura

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

To contextualize the results on **CulturaX**, we compare three categories of baselines.

### 1️⃣ General-Purpose Aligned Models
We include **joint-axis HHH alignment** frameworks that optimize all three axes simultaneously, excluding single-axis methods such as RAHF and Aligner.  
Representative works:

- [MARL-Focal](https://arxiv.org/abs/2502.04492) — Multi-agent reinforcement for joint alignment  
- [TrinityX](https://arxiv.org/abs/2509.08486) — Multi-stage alignment with adaptive calibration  
- [H³Fusion](https://arxiv.org/abs/2411.17792) — Multi-objective fusion of helpfulness, harmlessness, and honesty

---

### 2️⃣ Culturally Fine-Tuned Models
While no open-weight LLMs are fine-tuned explicitly for cultural alignment, we adapt existing SOTA alignment frameworks (MARL-Focal, TrinityX, H³Fusion) via **LoRA** for efficiency.

**Training setup:**
- Learning rate: `2e-5`  
- Global batch size: `128`  
- Max sequence length: `1024`  
- Epochs: `3–5` (with early stopping on 5% validation set)  
- Random seed: `42`

This configuration provides a scalable baseline for **culturally adaptive alignment**, testing if HHH improvements transfer to cultural reasoning.

---

### 3️⃣ Open-Weight LLMs
We also evaluate strong open-weight models without cultural tuning:

- **[Gemma-7B](https://huggingface.co/google/gemma-7b)**  
- **[DeepSeek-7B](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)**  

These serve as representative mid-scale open models for comparison.

---

## Experimental Results and Analysis

All experiments were performed using **PyTorch 2.3** on **4× NVIDIA A100 (80 GB)** GPUs with mixed precision and a fixed random seed `42`.

### Stage I – Generation
- Temperature: `0.7`  
- Top-p: `0.9`  
- Max length: `512`  
- Each query generated up to `K = 3` candidates  
- Maximum of 2 feedback iterations to balance quality and compute efficiency

### Stage II – Evaluation
- Averaged over **3 independent runs**  
- Parameters: temperature `0.7`, top-p `0.9`, max length `512`, repetition penalty `1.1`  
- Ensures consistent and low-variance evaluation across cultural domains

### Dataset – CulturaX
- Total samples: `M = 1500`  
- Split: 80 % train, 10 % validation, 10 % test  

---

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

# DeepNet: Scaling Transformers to 1,000 Layers

### Hongyu Wang∗ Shuming Ma∗ Li Dong Shaohan Huang Dongdong Zhang Furu Wei†

Microsoft Research [https://github.com/microsoft/unilm](https://github.com/microsoft/unilm)

## Abstract

In this paper, we propose a simple yet effective method to stabilize extremely deep Transformers. Specifically, we introduce a new normalization function (DEEPNORM) to modify the residual connection in Transformer, accompanying with theoretically derived initialization. In-depth theoretical analysis shows that model updates can be bounded in a stable way. The proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN, making DEEPNORM a preferred alternative. We successfully scale Transformers up to 1,000 layers (i.e., 2,500 attention and feed-forward network sublayers) without difficulty, which is one order of magnitude deeper than previous deep Transformers. Remarkably, on a multilingual benchmark with 7,482 translation directions, our 200-layer model with 3.2B parameters significantly outperforms the 48-layer state-of-the-art model with 12B parameters by 5 BLEU points, which indicates a promising scaling direction.

### DeepNet 1000

### Turing-NLG 200 GPipe Megatron-LM MT-NLG Gopher BERT-large GPT-2 GPT-3 XLM-R Transformer-big Roberta T5 GLaM 0 GShard XGLM

**2017 2018 2019 2020 2021 2022**

Figure 1: Trend of Transformer depths of state-of-the-art NLP models over time.

*∗Equal contribution. Work was done during Hongyu's internship at Microsoft Research.* *†Corresponding author <fuwei@microsoft.com>* .

-----

# 1 Introduction

Recent years have witnessed a trend towards large-scale Transformer (Vaswani et al., 2017) models. The capacity has substantially increased from millions of parameters (Devlin et al., 2019; Conneau et al., 2020) to billions (Radford et al., 2019; Brown et al., 2020; Huang et al., 2019; Raffel et al., 2020; Lepikhin et al., 2021; Rae et al., 2021; Lin et al., 2021; Smith et al., 2022), and even trillions (Du et al., 2021). Large-scale models yield state-of-the-art performance on a wide range of tasks, and show impressive abilities in few-shot and zero-shot learning. Despite an enormous number of parameters, their depths (as shown in Figure 1) are limited by the training instability of Transformers. Nguyen and Salazar (2019) find that pre-norm residual connections (Pre-LN) improve the stability of Transformers based on post-norm connections (Post-LN). However, the gradients of Pre-LN at bottom layers tend to be larger than at top layers (Shleifer et al., 2021), leading to a degradation in performance compared with Post-LN. In order to alleviate the above issue, there have been efforts on improving the optimization of deep Transformer by means of better initialization (Zhang et al., 2019a;b; Huang et al., 2020), or better architecture (Wang et al., 2019; Liu et al., 2020; Bachlechner et al., 2020; Shleifer et al., 2021). These approaches can stabilize a Transformer model with up to hundreds of layers. Yet, none of previous methods has been successfully scaled to 1,000 layers. Our aim is to improve the training stability of Transformers and scale the model depth by orders of magnitude. To this end, we study the cause of unstable optimization, finding the exploding model update is responsible for the instability. Motivated by the above observation, we introduce a new normalization function (DEEPNORM) at residual connections (He et al., 2016), which has theoretical justification of bounding the model update by a constant. The proposed method is simple yet effective, with just lines of code change. The approach improves the stability of Transformers so that we are able to scale model depth to more than 1,000 layers. Moreover, experimental results show that DEEPNORM combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN. The proposed method can be a preferred alternative of Transformers, not only for extremely deep (such as >1000 layers) models, but also for existing large models. Notably, our 200-layer model with 3.2B parameters achieves 5 BLEU improvement on a massively multilingual machine translation benchmark compared to state-of-the-art model (Fan et al., 2021) with 48 layers and 12B model size.

# 2 TL;DR for Practitioners

---

```
def deepnorm(x): Encoder Decoder
```

### Architectures

```
return LayerNorm(x ∗ α + f(x)) α β α β
```

---

Encoder-only

(2N) 1 (8N)- 1 - -

```
def deepnorm_init(w): (e.g., BERT) 4 4
if w is ['ffn', 'v_proj', 'out_proj']: Decoder-only
```

- - (2M) 1 (8M)- 1

```python
nn.init.xavier_normal_(w, gain=β) (e.g., GPT) 4 4
elif w is ['q_proj', 'k_proj']:
```

1 0.87(N4M)- 1 (3M) 1 (12M)- 1

16 16 4 4

```
nn.init.xavier_normal_(w, gain=1) (e.g., NMT, T5)
```

---

Figure 2: (a) Pseudocode for DEEPNORM. We take Xavier initialization (Glorot and Bengio, 2010) as an example, and it can be replaced with other standard initialization. Notice that α is a constant. (b) Parameters of DEEPNORM for different architectures (N-layer encoder, M-layer decoder). As shown in Figure 2, it is simple to implement our method based on Transformers with Post- LN. Compared to Post-LN, DEEPNORM up-scales the residual connection before performing layer normalization. Besides, we down-scale the parameters during initialization. Notably, we only scale the weights of feed-forward networks, as well as the value projection and the output projection of attention layers. Moreover, the scales of residual connection and initialization are dependent on the architecture (Figure 2). We provide more details in Section 4.3.

# 3 Instability of Deep Transformer

We study the causes of the instability for deep Transformers. Our analysis begins with the observation: better initialization methods stabilize the training of Transformer. This has also been verified by previous work (Zhang et al., 2019a; Huang et al., 2020; Xu et al., 2021). Therefore, we study the

-----

2.5 3

Post-LN 15 Post-LN Post-LN-init 2.0 Post-LN-init

2 12

1.5

9 Post-LN 1.0 Post-LN-init

1 0.5 6 0.0 0 6 12 18 24 3

13 14 15 16 17 18 0 10 20 30 Decoder Layers Layers Epochs

Figure 3: (a) Gradient norm in the top layers of 18L-18L models. (b) Gradient norm in the last layer of the models with depths varying from 6L-6L to 24L-24L. (c) Validation loss curves of 18L-18L models.

30 35 50

| Post-LN + no warmup | Post-LN + no warmup |
|---|---|
| Post-LN + 4k warmup | Post-LN + 4k warmup |
| Post-LN-init + 4k warmup | Post-LN-init + 4k warmup |

20 30 40

Post-LN + no warmup Post-LN + 4k warmup

30

| 10 | Post-LN-init + 4k warmup | 25 |
|---|---|---|
| 0 |  | 20 |

0 10 20 30 0 10 20 30 20

0 10 20 30

| Iterations | Iterations | Iterations |
|---|---|---|
| (a) Accumulated model update | (b) Input from FFN to LN | (c) Input from attention to LN |

0.9

---

Post-LN + no warmup Post-LN + 4k warmup 0.6 Post-LN-init + 4k warmup

---

0.3

---

0.0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18

---

Decoder Layers

(d) Gradient norm in all decoder layers

Figure 4: Visualization of the model update, the average input of LNs, and the gradients for the 18L-18L models at the early stage of training.

training process of Post-LN with or without proper initialization. With better initialization, we downscale the weights of l-th layer by kl = N - l + 1, l ∈ [1, N] after performing Xavier initialization.

For example, the output projection Wl *o of FFN in l-th layer is initialized as:*

1

| Wl |  |  |
|---|---|---|
| o v N | 0, | , |

*k* 2 *d0*

*l where d *0* is an average of input and output dimensions. We name this model Post-LN-init. Notice that different from the prior work (Zhang et al., 2019a), we narrow the scale of lower layers instead of the higher layers. We believe that it helps to separate the effect of the gradient scale from the model update. Besides, Post-LN-init has the same architecture as Post-LN, which eliminates the impact from the architecture. We train 18L-18L Post-LN and 18L-18L Post-LN-init on the IWSLT-14 De-En machine translation dataset. Figure 3 visualizes their gradients and validation loss curves. As shown in Figure 3(c), Post-LN-init converged while Post-LN did not. Post-LN-init has an even larger gradient norm in the last several layers, although its weights have been scaled down. Furthermore, we visualize the*

-----

gradient norm of the last decoder layer with varying model depth from 6L-6L to 24L-24L. Figure 3 shows that the gradient norm of Post-LN-init in the last layer is still much larger than that of Post-LN, regardless of model depth. It concludes that the exploding gradients in deep layers should not be the root cause of instability of Post-LN, while the scale of model update tends to account for it. Then we demonstrate that the instability of Post-LN comes from a chain of several issues, including gradient vanishing as well as too large model updates. As shown in Figure 4(a), we first visualize the norm of model update ||ΔF|| at the early stage of training:

*||ΔF|| = ||F(x, θi) - F(x, θ0)||,*

where x and θi denotes input, and model parameters after i-th updates. Post-LN has an exploding update at the very beginning of training, and then nearly no update shortly. It indicates that the model has been stuck in a spurious local optima. Both warm-up and better initialization help alleviate this issue, enabling the model to update smoothly. When the update explodes, the inputs to LN become large (see Figure 4(b) and Figure 4(c)). According to the theoretical analysis from Xiong et al. (2020), the magnitude of gradient through LN is inversely proportional to the magnitude of its input:

*||∂LN(x) √*

*d* *|| = O( ||x||).* *∂x* Figure 4(b) and Figure 4(c) show that ||x|| is significantly larger than √ *d (d = 512) without warm-up* or proper initialization. This explains the gradient vanishing problem occurred in the training of Post-LN (see Figure 4(d)). Above all, the instability starts from the large model update at the beginning of training. It renders the model trapped in a bad local optima, which in turn increases the magnitude of inputs to each LN. As training continues, the gradient through LN becomes increasingly small, thus resulting in severe gradient vanishing. The vanishing gradients make it difficult to escape from the local optima, and further destabilize the optimization. On the contrary, Post-LN-init has relatively small updates, and the inputs to LN are stable. This relieves suffering from gradient vanishing, making optimization more stable.

# 4 DEEPNET: Extremely Deep Transformers

In this section, we introduce our extremely deep Transformers named DEEPNET. It can stabilize the optimization by mitigating the exploding model update problem. We first provide the estimation of the expected magnitude of DEEPNET's model update. Then we provide the theoretical analysis to show that its updates can be bounded by a constant with our proposed DEEPNORM.

## 4.1 Architecture

DEEPNET is based on the Transformer architecture. Compared to the vanilla Transformer, it uses our new DEEPNORM, instead of Post-LN, for each sub-layer. The formulation of DEEPNORM can be written as:

*xl+1 = LN(αxl + Gl(xl, θl)),*

where α is a constant, and Gl(xl, θl) is the function of the l-th Transformer sub-layer (i.e., attention or feed-forward network) with parameters θl. Besides, DEEPNET scales the weights θl inside residual branches by β. Notably, both α and β are constants that only depend on the architecture, and we provide the derivation in Section 4.3.

## 4.2 Expected Magnitude of Model Update

Attention is an important part of Transformer. Without loss of generality, we study the 1-head case. Let Q, K, V ∈ Rn×d denote the query, key, value, respectively. WQ, W K, WV ∈ Rd×dk are the input projection matrices, and WO ∈ Rdk×d is the output projection matrix. Then, the attention module can be formulated as:

*QWQ(KW K)T* *Attn(Q, K, V ) = sof tmax(* ~~*√*~~ )V WV WO

*dk*

-----

We study the magnitude of the attention module. Lemma 4.1 proves that WQ and W K do not change the bound of attention output's magnitude.

**Lemma 4.1. Given X = (x1, x2, ...xn)T ∈ Rn×d, where var(xi) = 1, mean(xi) = 0 and qi ∈ R**

*for all i ∈ [1, n], it satisfies that*

Θ

*sof tmax(q1, q2, ..., qn)X = xi,*

*where Θ* = stands for equal bound of magnitude. In other words, the magnitude of attention output only depends on the value and output projection:

Θ

*Attn(Q, K, V ) = V WV WO. In this work, we only consider the magnitude of model update, so it is* sufficiently instructive to study the case where the hidden dimension equals to 1. For simplicity, we reduce the matrices WV, WO to the scalars v, w, which means Attn(Q, K, V ) = vwV Θ . Similarly,

Θ

we have F F N(X) = vwX, where v, w denotes the parameters of the feed-forward network. We define the model update as ||ΔF|| = ||F(x, θ∗) - F(x, θ)||. Based on the analysis above, we have the following theorem to characterize ||ΔF||'s magnitude of an N-layer DEEPNET with N attentions and FFNs.

**Theorem 4.2. Given an N-layer DEEPNET F(x, θ) (θ = {θ1, θ2, ..., θ2N }), where θ2l-1 and θ2l**

*denote the parameters of self-attention and FFN in l-th layer, and each sub-layer is normalized with* DEEPNORM: xl+1 = LN(αxl + Gl(xl, θl)), ||ΔF|| satisfies:

2N p i + w22 *v* *||ΔF|| ≤ X i ||θ i - ∗ θi||*

*α*

*i=1*

Vanilla Post-LN can be regarded as a special case of DEEPNET, where α = 1 and vl = wl = 1 at Xavier initialization (Glorot and Bengio, 2010). Based on Theorem 4.2, we have ||ΔF|| = *O(* P2N *i=1 ||θ i -∗ θi||) for vanilla Post-LN. It shows that the model tends to accumulate the update* of each sub-layer, which leads to exploding magnitude of model's update and destabilizes the optimization at the early stage. This explains our findings in Section 3. Besides, Theorem 4.2 also explains why warm-ups and smaller initialization can stabilize the training of Post-LN. Warm-ups can reduce the magnitude of the model update by decreasing ||θi - θi∗ *||, while* smaller initialization lowers p *vi + w22* .

*i*

Furthermore, we study the magnitude of DEEPNET with an N-layer encoder and an M-layer decoder. Let Fed(x, y, θe, θd) denotes the model, where x, y is the input of encoder and decoder. θe follows the same definition as θ in Theorem 4.2. θd = {θd1, θd2, ..., θd,3M} stands for the parameters of selfattentions, cross-attentions, and FFNs. We use {αe, Gel} and {αd, Gdl} to distinguish the notations between the encoder and the decoder. The following theorem shows the expected magnitude of the encoder-decoder's model update ||ΔFed|| = ||Fed(x, y, θ∗ *, θ∗) - Fed(x, y, θe, θd)||.*

*e d*

**Theorem 4.3. Given an encoder-decoder DEEPNET Fed(x, y, θe, θd) with N encoder layers and M**

*decoder layers, where each encoder sub-layer is normalized as xl+1 = LN(αexl + Gel(xl, θel)), and the decoder sub-layer is normalized as xl+1 = LN(αdxl + Gdl(xl, θdl)), ||ΔFed|| satisfies:*

*M vd,3j-1wd,3j-1* 2N p *vei + w22*

*||ΔFed|| ≤ X* X *ei ||θ ei - θei|| ∗*

*αd αe*

*j=1 i=1*

q *vdj + w22*

3M

X *dj ||θ dj - θdj || ∗* + (1)

*αd*

*j=1*

The vanilla encoder-decoder model satisfies that all of {αe, αd, vei, wei, vdi, wdi} equal to 1,

so we have ||ΔFed|| = O(M P2N *i=1 ||θ ei - θei|| + ∗* P3M *j=1 ||θ dj - θdj ||). It indicates the similar ∗*

accumulative effect which leads to fast growth of the magnitude regarding the model depth (see Figure 5). Furthermore, the cross-attention propagates the magnitude from the encoder to the decoder, which explains why the decoder is more unstable than the encoder (Liu et al., 2020).

-----

0.5

Post-LN 0.4 DeepNet

0.3 0.2 0.1 0.0

6 26 46 66 86

Layers

Figure 5: Model updates of vanilla Post-LN and DEEPNET at the early stage of training. The visualization is conducted on 64-128-2 tiny Transformers with depth varying from 6L-6L to 100L- 100L. It shows that DEEPNET has much smaller and more stable updates than Post-LN.

## 4.3 Derivation for DEEPNORM and the Initialization

We show that the expected model updates for DEEPNET can be bounded by a constant with proper parameters α and β. Our analysis is based on SGD update, and we empirically verify it works well for Adam optimizer (Kingma and Ba, 2015). We provide the analysis on the encoder-decoder architecture, which can be naturally extended to encoder-only and decoder-only models in the same way. Analogous to Zhang et al. (2019b), we set our goal for the model update as follows:

---

**GOAL: Fed(x, y, θe, θd) is updated by Θ(η) per SGD step after initialization as η → 0. That is**

*||ΔFed|| = Θ(η) where ΔFed = Fed(x, y, θe - η* Δ *∂L , θd - η ∂L* ) - Fed(x, y, θe, θd).

*∂θe ∂θd*

---

For SGD optimizer, the update of each decoder layer ||θ *di - θdi|| equals to η|| ∂L∗ ||. Xiong et al.*

*∂θdi*

(2020) proved that Post-LN decreases the magnitude of backpropagating error signal, so we have

*|| ≤ || ∂F ||. With || ∂F ||* = Θ *||θd,3M||* and the assumption || ∂L *∂F || = O(1), the second*

*∂θdj ∂θd,3M ∂θd,3M αd*

term of Equation (1) can be bounded as:

q *vdj + w22* q *v dj + w22*

3M *dj - θdj || ≤ η|| ∂L ∂F* 3M

X *dj ||θ ∗ ∂F|| · || ||X dj*

*αd ∂θd,3M αd*

*j=1 j=1*

*v d + w* 2 2 Θ *d*

= 3ηM (2) *α2*

*d*

There are multiple schemes to bound Equation (2) by Θ(η). In order to balance the effect of residual connections and the initialization, we set αd = (3M) 2 1 2 , *vd + w* 2 2 = (3M)d 1 2 and vd = wd = βd due to symmetry, that is αd = (3M)4 , βd = (12M) 1 *- 1* . Similarly, we use ve = we = βe =

4 0.87(N4M)- 1 16 , αe = 0.81(N4M) 1 16 to bound the first term in Equation (1). Detailed derivation is

shown in Appendix B. In comparison with Post-LN, we visualize the model updates for DEEPNET on IWSLT-14 De-En translation dataset at the early training stage. Figure 5 shows that the model update of DEEPNET is nearly constant, while the model update of Post-LN is exploding. In summary, we apply our approach as follows:

-----

### Models

| Vanilla Post-LN (Vaswani et al., 2017) DS-Init (Zhang et al., 2019a) Admin (Liu et al., 2020) |
|---|
| ReZero (Bachlechner et al., 2020) R-Fixup (Zhang et al., 2019b) T-Fixup (Huang et al., 2020) |
| Vanilla Pre-LN (Vaswani et al., 2017) DLCL (Wang et al., 2019) NormFormer (Shleifer et al., 2021) |
| DEEPNET (ours) |

---

Table 1: BLEU scores on the WMT-17 En-De test set for different models with varying depth. AL-BL refers to A-layer encoder and B-layer decoder.

| LN | 6L-6L | 18L-18L | 50L-50L | 100L-100L |
|---|---|---|---|---|
| Post Post | 28.1 27.9 |  | diverged diverged |  |
| Post | 27.9 | 28.8 |  | diverged |
| No | 26.9 |  | diverged |  |
| No | 27.5 | 28.4 | 27.7 | diverged |
| No | 27.5 | 28.4 | 27.9 | diverged |
| Pre | 27.0 | 28.1 | 28.0 | 27.4 |
| Pre | 27.4 | 28.2 | diverged | 27.5 |
| Pre | 27.0 | 28.3 | 27.8 | diverged |
| Deep | 27.8 | 28.8 | 29.0 | 28.9 |

### Encoder-decoder architecture

1. Apply standard initialization (e.g., Xavier initialization) for each encoder and decoder layer.
2. For encoder layers, scale the weights of feed-forward networks as well as the value projection and the output projection of attention layers by 0.87(N4M)- 1 16 , and set the weight of residual connections as 0.81(N4M) .
3. For decoder layers, scale the weights of feed-forward networks as well as the value projection and the output projection of attention layers by (12M) *- 1* 4 , and set the weight

1

of residual connections as (3M) .

4

The derivation of encoder-only (such as BERT) and decoder-only (such as GPT) architectures can be

conducted in the same way (see Appendix C). We summarize the steps as follows:

### Encoder-only (or decoder-only) architecture

1. Apply standard initialization (e.g., Xavier initialization) for each layer.
2. For each layer, scale the weights of feed-forward networks as well as the value projection and the output projection of attention layers by (8N) *- 1* 4 (or (8M) *- 1* 4 ), and set the weight

1 1

of residual connections as (2N)4 (or (2M) 4 ).

# 5 Neural Machine Translation

We verify the effectiveness of DEEPNET on the popular machine translation benchmarks, including IWSLT-14 German-English (De-En) dataset and WMT-17 English-German (En-De) dataset. We compare our method with multiple state-of-the-art deep Transformer models, including DLCL (Wang et al., 2019), NormFormer (Shleifer et al., 2021), ReZero (Bachlechner et al., 2020), R-Fixup (Zhang et al., 2019b), T-Fixup (Huang et al., 2020), DS-init (Zhang et al., 2019a), and Admin (Liu et al., 2020). We reproduce the baselines with their open-source code, and set the hyper-parameters the same for a fair comparison. We use BLEU as the evaluation metric for all experiments. Table 1 reports the results of the baselines and DEEPNET on WMT-17 En-De translation dataset. According to their LNs, the baselines are grouped into three categories: Pre-LN, Post-LN, and No-LN. All the compared models are base-size with different depths. Compared with the models with Post-LN, DEEPNET is more stable, and can successfully scale to 100L-100L, reaching the 28.9 BLEU on the test set. In contrast, the baselines with Post-LN lead to

-----

35 35 30

30 30

Post-LN 20 DS-init

Admin DeepNet 25 T-Fixup 25 Pre-LN

| 10 | ReZero (fp32) R-Fixup DeepNet | DLCL NormFormer DeepNet |
|---|---|---|

0 20 10 40 70 100 20 10 40 70 100

10 40 70 100

Layers Layers Layers

from 10L-10L to 100L-100L.

| 4.8 | lr = 5e-4 | 4.8 | bsz = 64 x 4k | 4.8 | 512-2048-8 |
|---|---|---|---|---|---|
| 4.6 | lr = 1e-3 | 4.6 | bsz = 128 x 4k | 4.6 | 768-3072-12 |
|  | lr = 1.5e-3 |  | bsz = 256 x 4k |  | 1024-4096-16 |
| 4.4 |  | 4.4 |  | 4.4 |  |
| 4.2 |  | 4.2 |  | 4.2 |  |
| 4.0 |  | 4.0 |  | 4.0 |  |

3.8 0 10K 20K 30K 3.8 0 10K 20K 30K 3.8 0 10K 20K 30K

Iterations Iterations Iterations

Figure 7: WMT-17 En-De validation loss curves for 18L-18L DEEPNET with varing learning rate, batch size and hidden dimension.

unstable optimization when the depth goes to 50L-50L. Besides, DEEPNET achieves comparable performance with these baselines when the models are shallow. In addition, we compare DEEPNET with the methods without LN. Both R-Fixup and T-Fixup introduce better initialization methods, which stabilize the training of No-LN Transformer with up to 50-50 layers. Yet, their performance is not as good as those with Post-LN. Besides, half-precision could destabilize the training of ReZero, leading to its divergence with 18-18 layers. This observation is also reported by Liu et al. (2020). Moreover, deeper models (50L-50L) do not outperform the shallow models (18L-18L). In comparison, DEEPNET achieves better translation accuracy than these methods, and scaling to deeper models brings no harm to the performance. Compared with the Post-LN baselines, the models with Pre-LN are more stable. Both vanilla Pre-LN and DLCL can be scaled to 100L-100L, and 50L-50L NormFormer is also trained successfully. Nevertheless, Pre-LN leads to a 0.5-1.0 BLEU drop compared with the converged Post-LN models. We presume this should be caused by the problem that gradients of Pre-LN at earlier layers tend to be larger than gradients at later layers (Shleifer et al., 2021). We leave it as the future work. In contrast, DEEPNET alleviates the problem by using Post-LN, and outperforms all the Pre-LN baselines.

**Convergence with varying depth. We vary the depths of the models from 10L-10L to 100L-100L**

with an interval of 10 layers.All experiments are conducted with mixed precision training, except ReZero3 . Figure 6 shows the results on the IWSLT-14 dataset. We train the models for 8,000 steps because we find most divergence occurs at the beginning of optimization. Overall, DEEPNET is stable from shallow to deep. It converges fast, achieving over 30 BLEU in only 8,000 steps while most of the baselines do not. Moreover, the performance keeps improving as the model goes deeper.

**Large learning rate, batch size, and hidden dimension.** We further scale DEEPNET to larger learning rate, batch size, and hidden dimension, respectively. For each experiment, we only change one hyperparameter with the others fixed. Figure 7 reports the loss curves on the WMT-17 validation set. It shows that DEEPNET can be trained without difficulty in all the largest settings. The loss of DEEPNET with 1024 hidden size increases after 10K steps because of overfitting. Besides, it indicates that DEEPNET can benefit from the larger settings, resulting in faster convergence and lower validation loss.

3According to our experiments, ReZero is unstable with half precision, even when the model is shallow.

-----

31

27 25

10 100 1000

Depth

Figure 8: Average BLEU scores for DEEPNET with varying depth on the OPUS-100 En-X and X-En test sets.

---

| Models | # Layers | # Params | X→En | En→X | Avg |
|---|---|---|---|---|---|
|  | 12 | 133M | 27.5 | 21.4 | 24.5 |
| Baseline (Zhang et al., 2020) | 24 48 | 173M 254M | 29.5 31.4 | 22.9 24.0 | 26.2 27.7 |
| DEEPNET (ours) | 200 1000 | 863M 3.8B | 33.2 33.9 | 29.0 30.2 | 31.1 32.1 |

---

Table 2: Average BLEU for DEEPNET and the baseline on the OPUS-100 test sets.

# 6 Massively Multilingual Neural Machine Translation

We conduct experiments on the large-scale multilingual machine translation, which is a good testbed for large models. We first use OPUS-100 corpus (Zhang et al., 2020) to evaluate our model. OPUS- 100 is an English-centric multilingual corpus covering 100 languages, which is randomly sampled from the OPUS collection. We scale DEEPNET up to 1,000 layers. The model has a 500-layer encoder, a 500-layer decoder, 512 hidden size, 8 attention head, and 2,048 dimensions of feed-forward layers. More details can be found in the Appendix. Table 2 summarizes the results of DEEPNET and the baselines. It shows that increasing the depth can significantly improve the translation quality of NMT: the baseline of 48 layers achieves a gain of 3.2 points on average over the 12-layer model. DEEPNET can successfully scale up the depth to 1,000 layers, outperforming the baseline by an improvement of 4.4 BLEU. It is noted that DEEPNET is only trained for 4 epochs, and the performance can be further improved given more computation budgets.

**Scaling law in terms of depth** We train DEEPNET of {12, 20, 100, 200, 1000} layers on the OPUS- 100 dataset. Figure 8 illustrates the scaling curve. Compared with bilingual NMT, multilingual NMT benefits more from scaling the depth of the model because of its hunger in model capacity. We observe logarithmic growth of the BLEU score for multilingual NMT, and the scaling law can be written as:

*L(d) = A log(d) + B* where d is the depth, and A, B are the constants regarding the other hyper-parameters.

**More data and language directions.** To explore the limits of DEEPNET on multilingual NMT, we then scale up the training data by using CCMatrix (Schwenk et al., 2021). We also expand the data from CCAligned (El-Kishky et al., 2020), OPUS (Zhang et al., 2020), and Tatoeba4 to cover all languages of Flores101 evaluation sets. The final data consists of 102 languages, 1932 directions, and 12B sentence pairs. With the data, we train DEEPNET with a 100-layer encoder, 100-layer decoder, 1,024 hidden dimension, 16 heads, and 4,096 intermediate dimension of feed-forward layers. More details can be found in the Appendix.

-----

| Models | # Layers | # Params | WMT | OPUS | TED | Flores |
|---|---|---|---|---|---|---|
| M2M-100 (Fan et al., 2021) | 48 | 12B | 31.9 | 18.4 | 18.7 | 13.6 |
| DEEPNET (ours) | 200 | 3.2B | 33.9 | 23.0 | 20.1 | 18.6 |

---

Table 3: BLEU scores for DEEPNET and M2M-100 on various evaluation sets.

We compare DEEPNET with the state-of-the-art multilingual NMT model M2M-100 (Fan et al., 2021). M2M-100 has a 24-layer encoder, a 24-layer decoder, and 4,096 hidden size, resulting in up to 12B parameters. Compared with M2M-100, DEEPNET is deep and narrow with only 3.2B parameters. For a fair comparison, we generate the model with beam size 5 and length penalty 1. Following M2M-100 (Fan et al., 2021), we evaluate the models on several multilingual translation evaluation datasets, including WMT (Bojar et al., 2014; 2017; 2018; Barrault et al., 2019), OPUS (Zhang et al., 2020), TED (Qi et al., 2018), and Flores (Goyal et al., 2021). The language pairs from the WMT dataset are English-centric. There are 10 languages including English, and most of them are high-resource. For the OPUS dataset, we select the non-English directions from the test set, which has 30 evaluation pairs. The TED evaluation set has 28 languages and 756 directions, and the data is from the spoken language domain. The Flores dataset has all translation pairs between 102 languages. We use a subset covering the languages supported by both M2M-100 and DEEPNET, resulting in 87 languages and 7,482 translation directions. We report the results in Table 3. For a fair comparison, we use the same evaluation methods as the baseline. The details can be found in the Appendix. It shows that DEEPNET has significantly better performance than M2M-100 on all evaluation datasets, indicating that deepening the model is a very promising direction to improve the quality of NMT models.

# 7 Conclusion and Future Work

We improve the stability of Transformer and successfully scale it to 1,000 layers. This is achieved by our DEEPNET with a novel normalization function called DEEPNORM. It has theoretical justification to stabilize the optimization with a constant upper bound for model updates. Experimental results verify the effectiveness of our methods across various benchmarks. We focus on machine translation as a test bed in the current experiments. In the future, we will extend DEEPNET to support more diverse tasks, e.g., language model pre-training (Dong et al., 2019; Bao et al., 2020; Chi et al., 2021a; Ma et al., 2021; Chi et al., 2021b), protein structure prediction (Jumper et al., 2021), and BEiT vision pre-training (Bao et al., 2022; Wang et al., 2021).

**Acknowledgement We would like to acknowledge Saksham Singhal for the CCMatrix corpus.**

## References

Thomas Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao, Garrison W. Cottrell, and Julian J. McAuley. Rezero is all you need: Fast convergence at large depth. CoRR, abs/2003.04887, 2020.

Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Songhao Piao, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. UniLMv2: Pseudo-masked language models for unified language model pre-training. In ICML 2020, volume 119 of Proceedings of Machine *Learning Research, pages 642-652. PMLR, 2020.* Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEiT: BERT pre-training of image transformers.

In International Conference on Learning Representations, 2022. Loïc Barrault, Ondrej Bojar, Marta R. Costa-jussà, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Philipp Koehn, Shervin Malmasi, Christof Monz, Mathias Müller, Santanu Pal, Matt Post, and Marcos Zampieri. Findings of the 2019 conference on machine translation (WMT19). In Ondrej Bojar, Rajen Chatterjee, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno-Yepes, Philipp Koehn, André

-----

Martins, Christof Monz, Matteo Negri, Aurélie Névéol, Mariana L. Neves, Matt Post, Marco Turchi, and Karin Verspoor, editors, Proceedings of the Fourth Conference on Machine Translation, *WMT 2019, Florence, Italy, August 1-2, 2019 - Volume 2: Shared Task Papers, Day 1, pages 1-61.* Association for Computational Linguistics, 2019. Ondrej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes

Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, Radu Soricut, Lucia Specia, and Ales Tamchyna. Findings of the 2014 workshop on statistical machine translation. In *Proceedings of the Ninth Workshop on Statistical Machine Translation, WMT@ACL 2014, June* *26-27, 2014, Baltimore, Maryland, USA, pages 12-58. The Association for Computer Linguistics,* 2014.

Ondrej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Shujian

Huang, Matthias Huck, Philipp Koehn, Qun Liu, Varvara Logacheva, Christof Monz, Matteo Negri, Matt Post, Raphael Rubino, Lucia Specia, and Marco Turchi. Findings of the 2017 conference on machine translation (WMT17). In Ondrej Bojar, Christian Buck, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno-Yepes, Philipp Koehn, and Julia Kreutzer, editors, Proceedings of the Second Conference on Machine Translation, WMT *2017, Copenhagen, Denmark, September 7-8, 2017, pages 169-214. Association for Computational* Linguistics, 2017. Ondrej Bojar, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Philipp Koehn,

and Christof Monz. Findings of the 2018 conference on machine translation (WMT18). In Ondrej Bojar, Rajen Chatterjee, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno-Yepes, Philipp Koehn, Christof Monz, Matteo Negri, Aurélie Névéol, Mariana L. Neves, Matt Post, Lucia Specia, Marco Turchi, and Karin Verspoor, editors, *Proceedings of the Third Conference on Machine Translation: Shared Task Papers, WMT 2018,* *Belgium, Brussels, October 31 - November 1, 2018, pages 272-303. Association for Computational* Linguistics, 2018. Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,

Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS 2020, 2020. Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-

Ling Mao, Heyan Huang, and Ming Zhou. InfoXLM: An information-theoretic framework for cross-lingual language model pre-training. In NAACL-HLT 2021, pages 3576-3588, 2021a. Zewen Chi, Shaohan Huang, Li Dong, Shuming Ma, Saksham Singhal, Payal Bajaj, Xia Song,

and Furu Wei. XLM-E: cross-lingual language model pre-training via ELECTRA. CoRR, abs/2106.16138, 2021b. Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, ACL 2020, pages 8440-8451, 2020. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep

bidirectional transformers for language understanding. In NAACL-HLT 2019, pages 4171-4186, 2019.

Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou,

and Hsiao-Wuen Hon. Unified language model pre-training for natural language understanding and generation. In NeurIPS 2019, pages 13042-13054, 2019. Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim

Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathy Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Zhifeng Chen,

-----

and Claire Cui. Glam: Efficient scaling of language models with mixture-of-experts. CoRR, abs/2112.06905, 2021. Ahmed El-Kishky, Vishrav Chaudhary, Francisco Guzmán, and Philipp Koehn. CCAligned: A

massive collection of cross-lingual web-document pairs. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in *Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 5960-5969.* Association for Computational Linguistics, 2020. Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal,

Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Michael Auli, and Armand Joulin. Beyond english-centric multilingual machine translation. J. Mach. Learn. Res., 22:107:1-107:48, 2021. Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural

networks. In Yee Whye Teh and D. Mike Titterington, editors, AISTATS 2010, volume 9 of JMLR *Proceedings, pages 249-256. JMLR.org, 2010.* Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana

Krishnan, Marc'Aurelio Ranzato, Francisco Guzmán, and Angela Fan. The FLORES-101 evaluation benchmark for low-resource and multilingual machine translation. CoRR, abs/2106.03193, 2021.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image

recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770-778, 2016. doi: 10.1109/CVPR.2016.90. Xiao Shi Huang, Felipe Pérez, Jimmy Ba, and Maksims Volkovs. Improving transformer optimization

through better initialization. In ICML 2020, volume 119 of Proceedings of Machine Learning *Research, pages 4475-4483, 2020.* Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Xu Chen, HyoukJoong

Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng Chen. Gpipe: Efficient training of giant neural networks using pipeline parallelism. In NeurIPS 2019, pages 103-112, 2019. John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger,

Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A A Kohl, Andrew J Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with AlphaFold. Nature, 596 (7873):583-589, 2021. doi: 10.1038/s41586-021-03819-2. Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR 2015,

2015.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang,

Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. In ICLR 2021, 2021. Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott,

Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona T. Diab, Veselin Stoyanov, and Xian Li. Few-shot learning with multilingual language models. CoRR, abs/2112.10668, 2021. Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty

of training transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural *Language Processing (EMNLP), pages 5747-5763, 2020.*

-----

Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, Alexandre Muzio, Saksham Singhal,

Hany Hassan Awadalla, Xia Song, and Furu Wei. DeltaLM: Encoder-decoder pre-training for language generation and translation by augmenting pretrained multilingual encoders. CoRR, abs/2106.13736, 2021. Toan Q. Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of

self-attention. CoRR, abs/1910.05895, 2019. Matt Post. A call for clarity in reporting BLEU scores. In Ondrej Bojar, Rajen Chatterjee, Christian

Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno-Yepes, Philipp Koehn, Christof Monz, Matteo Negri, Aurélie Névéol, Mariana L. Neves, Matt Post, Lucia Specia, Marco Turchi, and Karin Verspoor, editors, Proceedings of the Third Conference *on Machine Translation: Research Papers, WMT 2018, October 31 - November 1, 2018, pages* 186-191. Association for Computational Linguistics, 2018. Ye Qi, Devendra Singh Sachan, Matthieu Felix, Sarguna Padmanabhan, and Graham Neubig. When

and why are pre-trained word embeddings useful for neural machine translation? In Marilyn A. Walker, Heng Ji, and Amanda Stent, editors, Proceedings of the 2018 Conference of the North *American Chapter of the Association for Computational Linguistics: Human Language Technolo-* *gies, NAACL-HLT, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 2 (Short Papers), pages* 529-535. Association for Computational Linguistics, 2018. Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language

models are unsupervised multitask learners. 2019. Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, H. Francis Song,

John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean- Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William S. Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher. *CoRR, abs/2112.11446, 2021.* Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi

Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1-67, 2020. Holger Schwenk, Guillaume Wenzek, Sergey Edunov, Edouard Grave, Armand Joulin, and Angela

Fan. CCMatrix: Mining billions of high-quality parallel sentences on the web. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the *Association for Computational Linguistics and the 11th International Joint Conference on Natural* *Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6,* *2021, pages 6490-6500. Association for Computational Linguistics, 2021.* Sam Shleifer, Jason Weston, and Myle Ott. Normformer: Improved transformer pretraining with

extra normalization. CoRR, abs/2110.09456, 2021. Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared

Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, Elton Zheng, Rewon Child, Reza Yazdani Aminabadi, Julie Bernauer, Xia Song, Mohammad Shoeybi, Yuxiong He, Michael Houston, Saurabh Tiwary, and Bryan Catanzaro. Using deepspeed and megatron to train megatron-turing NLG 530b, A large-scale generative language model. CoRR, abs/2201.11990, 2022.

-----

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz

Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS 2017, pages 5998-6008, 2017. Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao.

Learning deep transformer models for machine translation. In ACL 2019, pages 1810-1822, 2019. Wenhui Wang, Hangbo Bao, Li Dong, and Furu Wei. Vlmo: Unified vision-language pre-training

with mixture-of-modality-experts. ArXiv, abs/2111.02358, 2021. Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang,

Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture. In ICML 2020, volume 119 of Proceedings of Machine Learning Research, pages 10524-10533, 2020.

Peng Xu, Dhruv Kumar, Wei Yang, Wenjie Zi, Keyi Tang, Chenyang Huang, Jackie Chi Kit Cheung,

Simon J. D. Prince, and Yanshuai Cao. Optimizing deeper transformers on small datasets. In *ACL/IJCNLP 2021, pages 2089-2102, 2021.* Biao Zhang, Ivan Titov, and Rico Sennrich. Improving deep transformer with depth-scaled initialization and merged attention. In EMNLP-IJCNLP 2019, pages 898-909, 2019a. Biao Zhang, Philip Williams, Ivan Titov, and Rico Sennrich. Improving massively multilingual neural

machine translation and zero-shot translation. In ACL 2020, pages 1628-1639. Association for Computational Linguistics, 2020. Hongyi Zhang, Yann N. Dauphin, and Tengyu Ma. Fixup initialization: Residual learning without

normalization. In ICLR 2019, 2019b.

-----

# A Main Theorem Proof

### A.1 Proof of Theorem 4.1

**Lemma A.1. Given X = (x1, x2, ...xn)T ∈ Rn×d, where var(xi) = 1, mean(xi) = 0 and**

*qi ∈ R for all i ∈ [1, n], it satisfies that*

Θ

*sof tmax(q1, q2, ..., qn)X = xi ,*

*where Θ* = stands for equal bound of magnitude.

*e qi*

*Proof. The weight si of xi to output is si =* , Pn *i=1 si = 1.*

*e* Pn *j=1 qj*

*..., qn)X|| = ||Xn || ≤ Xn*

*||softmax(q1, q2, sixi si||xi||* (3)

*i=1 i=1*

With var(xi) = 1, mean(xi) = 0, for all i ∈ [1, n], we have ||xi|| = d. Therefore,

Θ

*||softmax(q1, q2, ..., qn)X|| ≤ ||xi|| = d, which is equivalent to softmax(q1, q2, ..., qn)X = xi* .

### A.2 Proof of Theorem 4.2

**Theorem A.2. Given an N-layer DEEPNET F(x, θ) (θ = {θ1, θ2, ..., θ2N }), where θ2l-1 and θ2l**

*denote the parameters of self-attention and FFN in l-th layer, and each sub-layer is normalized with* DEEPNORM: xl+1 = LN(αxl + Gl(xl, θl)), ||ΔF|| satisfies:

2N p *vi + w22*

*||ΔF|| ≤ X i ||θ i - ∗ θi||*

*α*

*i=1*

*Proof. Our aim is to study the magnitude of model updates. Following Zhang et al. (2019b), we* make the following assumptions to simplify the derivations:

1. Hidden dimension d equals to 1.
2. var(x + Gl(x)) Θ = var(x) + var(Gl(x))
3. All relevant weights v, w are positive with magnitude less than 1 and α, β for DEEPNORM are positive with magnitude greater than 1.

Θ

Given Assumption 1, if G(x) is feed-forward network with θ = {v, w}, then G(x) = vwx. According to Theorem 4.1, the query and key projections do not change the bound of the attention output's

Θ

magnitude. Therefore, if G(x) is self-attention with θ = {q, k, v, w}, then G(x) = vwx. Especially, if Xavier initialization is used for the projection, then the output can preserve the input variance, which is equivalent to v = w = 1. With Assumption 2, we have:

*αx + Gl(x) α + vlwl*

Θ

*xl+1 = fl(xl, θl) =* = *x* (4)

pV ar(αx + Gl(x)) pα2 + vl w22

*l*

With Equation (4), the magnitude of ∂fl *∂x and ∂fl* is bounded by:

*∂θl*

*∂fl α + vlwl*

Θ

=

*∂x* pα2 + vl w22

*l*

*∂fl = ( ∂f ∂f αxl(α - vlwl)*

Θ Θ

*,* ) = (wl, vl) (5) *∂θl ∂vl ∂wl* (α2 + vl w22 ) 3

2

*l*

-----

Besides, the model update ||ΔF|| satisfies:

*||ΔF|| = ||F(x, θ∗* ) - F(x, θ)|| = ||x Using Taylor expansion for Equation (6), we get:

*||ΔF|| = ||x ∗* 2N+1 - x2N+1||

*≈ ||∂f ∂x(x2N , θ2N )(x* *≤ ||∂f ∂x(x2N , θ2N )|| · ||x* *α + v2N w2N* =

pα2 + v2N w22 *≈ ||x2N - x2N || + ∗*

Then, we have:

*||x*

*∗* 2N+1 - x2N+1|| = ||f(x 2N , θ∗ *∗* 2N ) - f(x2N , θ2N )||

*x2N ) + ∂f*

| ∗ 2N - | ∂θ(x2N , θ2N )(θ2N - ∗ x2N \|\| + \|\|∂f | θ2N )T \|\| |
|---|---|---|
| ∗ 2N - | ∂θ(x2N , θ2N )\|\| · \|\|θ2N - ∗ α(α - v2N w2N ) q | θ2N \|\| |

*||x ∗* 2N - x2N || + *v2N + w22* 2N ||θ2N - *∗ θ2N ||*

(α2 + v2N w22 2N ) 3

2

2N

p *v2N + w22*

2N *||θ2N - θ2N || ∗*

*α*

2N p *vi + w22*

2N+1 - x2N+1|| ≤ X *∗ i ||θ i - θi|| ∗*

*α*

*i=1*

(6)

(7)

(8)

(9)

For vanilla Post-LN with standard initialization, α = vi = wi = 1, so ||ΔF|| = O( P2N *i=1 ||θ i -θi∗ ||).*

### Proof of Theorem 4.3

**Theorem A.3. Given an encoder-decoder DEEPNET Fed(x, y, θe, θd) with N encoder layers and M**

*decoder layers, where each encoder sub-layer is normalized as xl+1 = LN(αexl + Gel(xl, θel)), and the decoder sub-layer is normalized as xl+1 = LN(αdxl + Gdl(xl, θdl)), ||ΔFed|| satisfies:*

*M vd,3j-1wd,3j-1* 2N p *vei + w22* *||ΔFed|| ≤ X* X *ei ||θ ei - θei|| ∗*

*αd αe*

*j=1 i=1*

q *vdj + w22*

3M

X *dj ||θ dj - θdj || ∗* + (10)

*αd*

*j=1*

*Proof. The derivation of self-attention and FFN layers is given in Appendix A.2. For the cross-* attention layers, we have:

*αdyl + Gl(xe, yl) αdyl + vlwlxe*

Θ

*yl+1 = fdl(yl, xe, θdl) =* = (11)

pV ar(αdyl + Gdl(xe, yl)) p 2 *α d + v l w22*

*l*

With Equation (11), we have the bound of the derivative of fdl:

*∂fdl αd ∂fdl vlwl*

Θ Θ

= *,* = *∂y* p 2 *α d + v l w22 ∂xe* p 2 *α d + v l w22*

*l l*

*∂fdl = (∂fdl ∂fdl αdxe(αd - vdlwdl)*

Θ Θ

*,* ) = (wdl, vdl) *∂θdl ∂vdl ∂wdl* (α2 *d + v dlw22 dl)* 3

2

By means of Taylor expansion, we estimate the update of l-th cross-attention layer ||y∗ *l+1 - yl+1|| as:*

-----

*||y l+1 - yl+1|| = ||f ∗ dl(y ∗ ∗, x∗* 2N+1, θ∗ *dl) - fdl(yl, x2N+1, θdl)||*

*l*

*αd ≈ ||y ∗ - yl|| +*

p *α d + v2 dlw22 l*

*dl*

*αd(αd - vdlwdl)* q *dl + w22* + *v*

(α d + v2 *dlw22 dl)* 3

2

*vdlwdl ≤ ||y ∗ - yl|| + ||x2N+1 - x2N+1|| + ∗*

*l αd*

According to Theorem 4.2, we have ||x *∗* 2N+1-x2N+1|| = O( the magnitude of ||ΔFed|| satisfies:

*vdlwdl*

*||x ∗* 2N+1 - x2N+1||

p *α d + v2 dlw22*

*dl dl||θ dl - θdl|| ∗*

p *vdl + w22*

*dl ||θ dl - θdl|| ∗* (12) *αd* *√*

*vei+w22*

*ei ||θ ei-θei||). Therefore,∗*

*i=1 αe q *v dj + w22**

*M vd,3j-1wd,3j-1* 2N p ei + w22 *v* 3M *||ΔFed|| ≤ X* X *∗* X *dj ∗*

*ei ||θ ei - θei|| + ||θ dj - θdj ||* (13) *αd αe αd*

*j=1 i=1 j=1*

As a special case, the corresponding parameters in Equation (13) for vanilla Post-LN with standard initialization are 1, so its model update ||ΔFed|| = O(M P2N *i=1 ||θ ei - θei|| +∗* P3M *j=1 ||θ dj - θdj ||).∗*

# B Derivation for Encoder-Decoder Architecture

Here, we give the derivation of DEEPNET for the encoder-decoder architecture with an N-layer encoder and an M-layer decoder. As in Section 4.3, we have vd = wd = (12M) *- 1* 4 , αd = (3M) 1

4

to bound the second term of Equation (13) to Θ(η). For the first term, we set vei = ve, wei = we, so that it goes to: *M vd,3j-1wd,3j-1* 2N p *vei + w22* (12M) *- 1* 2N p *vei + w22* X X *ei ||θ ei - θei|| = M ∗* 2 X *ei ||θ ei - θei|| ∗* *αd αe* (3M) 1 *αe*

4

*j=1 i=1 i=1*

(14)

27 1

 4 v *e + w* 2 2 Θ *e*

= η (15)

*e* In this work, we use α *e = (N4M/27) 12* 8 , *ve + w* 2 *e = (N4M/27)- 12* 8 and ve = we = βe that is

*αe = 0.81(N4M)16 , βe1* = 0.87(N4M)- 1 16 to satisfy the condition.

# C Derivation for Encoder-only (Decoder-only) Architecture

For an N-layer DEEPNET, starting from Theorem 4.2 we have,

2N p *vi + w22*

*||x* 2N+1 - x2N+1|| ≤ X *∗ i ||θ i - ∗ θi||*

*α*

*i=1*

2N p *vi + w22 ∂F|| · || ∂F*

*≤ η* X *i ||* (16) *α ∂θi*

*i=1* By assumption || ∂L *∂F || = O(1), and || ∂F || ≤ || ∂F* = *||θ2N ||* , we achieve:

*∂θi ∂θ2N α*

2N p i + w22 *v ∂F|| · || ∂F* p 2N + w22 *v* 2N p i + w22 *v* X *i || ≤ O(* 2N X *i* ) = O(1) (17)

*α ∂θi α α*

*i=1 i=1*

-----

Due to symmetry, we set vi = v, wj = w, so it goes to 2N v 2+w2 = 1. In this work, we use

*α2*

*v = w = (8N) - 1* 4 and α = (2N)4 to satisfy the condition. 1

# D Experimental Details

## D.1 Hyperparameters for IWSLT-14 De-En

### Hyperparameters Value

Learning rate 5e-4 Learning rate scheduler inverse sqrt

| Warm-up updates | 4000 |
|---|---|
| Warm-up init learning rate | 1e-7 |
| Max tokens | 4000 |
| Adam  | 1e-8 |

Adam β (0.9, 0.98)

| Label smoothing | 0.1 |
|---|---|
| Training updates | 8K |
| Gradient clipping | 0.0 |
| Dropout | 0.4 |
| Weight decay | 0.0001 |
| Hidden size | 512 |
| FFN inner hidden size | 2048 |
| Attention heads | 8 |

Table 4: Hyperparameters for the machine translation experiments on the IWSLT-14 De-En dataset.

## D.2 Hyperparameters for WMT-17 En-De

| Hyperparameters | No-LN | Pre-LN | Post-LN | DEEPNORM |
|---|---|---|---|---|
| Learning rate Learning rate scheduler | 5e-4 | 1.5e-3 | 1.5e-3 inverse sqrt | 1.5e-3 |
| Warm-up updates Warm-up init learning rate |  |  | 4000 1e-7 |  |
| Max tokens Adam  |  |  | 128 × 4096 1e-8 |  |
| Adam β Label smoothing |  |  | (0.9, 0.98) 0.1 |  |
| Training updates |  |  | 100K |  |
| Gradient clipping Dropout |  |  | 0.0 0.4 |  |
| Weight decay |  |  | 0.0001 |  |
| Hidden size FFN inner hidden size |  |  | 512 2048 |  |
| Attention heads |  |  | 8 |  |

---

Table 5: Hyperparameters for the base-setting experiments on the WMT-17 En-De dataset.

-----

| Hyperparameters | Base size | Medium size | Large size |
|---|---|---|---|
| Hidden size | 512 | 768 | 1,024 |
| FFN inner hidden size | 2048 | 3072 | 4096 |
| Attention heads Layers | 8 | 12 18-18 | 16 |
| Learning rate Learning rate scheduler |  | 5e-4 inverse sqrt |  |
| Warm-up updates Warm-up init learning rate |  | 4000 1e-7 |  |
| Max tokens Adam  |  | 128 × 4096 1e-6 |  |
| Adam β Label smoothing |  | (0.9, 0.98) 0.1 |  |
| Training updates |  | 30K |  |
| Gradient clipping Dropout |  | 1.0 0.4 |  |
| Weight decay |  | 0.0 |  |

---

Table 6: Hyperparameters for the large-setting experiments on the WMT-17 En-De dataset.

## D.3 Hyperparameters for OPUS-100

### Hyperparameters Value

Learning rate 5e-4 Learning rate scheduler inverse sqrt Warm-up updates 4000 Warm-up init learning rate 1e-7

Max tokens 128 × 4096

Adam  1e-8

Adam β (0.9, 0.98)

Label smoothing 0.1 Training epochs 4 Gradient clipping 0.0 Dropout 0.1 Weight decay 0.0 Hidden size 512 FFN inner hidden size 2048 Attention heads 8 Table 7: Hyperparameters for the machine translation experiments on the OPUS-100 dataset.

-----

## D.4 Hyperparameters for 102-Language Machine Translation

### Hyperparameters Value

Learning rate 5e-4 Learning rate scheduler inverse sqrt Warm-up updates 6000 Warm-up init learning rate 1e-7

Max tokens 256 × 4096

Adam  1e-6

Adam β (0.9, 0.98)

| Label smoothing | 0.1 |
|---|---|
| Training updates | 260K |
| Gradient clipping | 1.0 |
| Dropout | 0.1 |
| Weight decay | 0.0 |
| Hidden size | 1024 |
| FFN inner hidden size | 4096 |
| Attention heads | 16 |

Layers 100-100

Table 8: Hyperparameters for the machine translation experiments on the 102-language dataset.

## D.5 Evaluation Details

For IWSLT-14 and WMT-17, we use the in-built BLEU scripts of Fairseq to report the scores. Besides, we report the case-sensitive detokenized BLEU using sacreBLEU (Post, 2018) for the results of OPUS-100.5 For WMT, OPUS, and TED, we use the same test sets and evaluation scripts as in M2M (Fan et al., 2021), and the results of M2M are directly from the paper (Fan et al., 2021). For the Flores-101 evaluation set, we report the spBLEU6 of M2M-12B with the public checkpoint and script.7

5BLEU+case.mixed+lang.{src}-{tgt}+numrefs.1+smooth.exp+tok.13a+version.1.4.14

6 [https://github.com/facebookresearch/flores](https://github.com/facebookresearch/flores)

-----

## E Experimental Results in Section 6

**af am ar ast az be bg bn bs ca ceb cs cy da de el en es et fa ff fi fr ga gl gu ha he hi hr hu hy id ig is it ja jv ka kk km kn ko lb lg ln lo lt lv mk ml mn mr ms my ne nl no ns oc or pa pl ps pt ro ru sd sk sl so sr sv sw ta th tl tr uk ur uz vi wo xh yo zh zu**

| af - ar | af - am 12.3 - ar 25.2 8.4 - | 15.8 | 10.7 23.8 21.7 8.6 12.8 37.4 24.2 27.6 29.4 16.1 31.5 21.5 39.5 34.6 28.6 55.5 25.9 29.5 25.7 0.1 3.1 4.7 3.9 4.7 13.9 13.6 9.6 14.2 4.9 5 1.2 31.8 21.9 22.3 24.2 3.3 26.4 16.4 29.5 27.1 26.1 36.6 22.8 24.4 23.9 | 26 41.5 0.3 32.7 7.2 10.9 28.3 29.4 28.1 28.2 9.6 35.5 13.8 23.4 28.2 24.4 19.5 5.1 10.1 12 11.6 12.9 11.9 13.8 16.9 12.3 13.7 11.3 0.1 11.7 18.4 0.3 12.3 5.2 5.6 10.2 16.1 10.8 13.6 3.2 10.3 9.5 9.4 13.3 7.9 0 21.9 34.5 0.3 26.1 0.8 8 25.3 26.2 23.4 23.9 7.1 29.7 12.2 | 14 | 14 8 3.6 3.9 8.6 2.3 7.4 9.3 0.5 0.3 20 24.2 22.2 15 1.9 2.5 12 0.7 17.8 15.9 0.4 0.5 8.7 25.1 |  | 4 19 23.9 1.3 0.8 13.7 29.3 22.7 33.6 18.8 11.3 15.8 32.9 5.6 2.4 28.4 30.9 7 13.9 12.9 14.8 9.3 8.1 27 28.1 8.4 10.2 5.8 25.8 2.6 0.2 | 4 28.9 0.4 13.6 23.3 12.8 41.3 36 29.2 12.1 32.2 28.9 4.8 30.8 38 29.2 5.7 20.1 8 9.5 1.8 5.8 9.3 10.2 1.6 11.2 0.4 9.7 10.3 8.8 17.1 15.3 9.2 7.5 13.4 13.5 2.5 12.5 13.4 13.6 23 22.4 1.6 21.5 0.4 9.1 19.6 11.4 33.3 29.5 | 25 10.5 27.3 24.7 2.5 21.5 29.1 21.2 1.1 18.8 7.4 23 23 12.6 1.1 30.7 0.6 10.7 1.3 18.5 11.3 | 23 27.5 28 16.2 1.6 35.3 0.7 12.9 2.9 18.9 13.5 4 11.1 12.5 9.3 11.8 9.1 0.8 17.1 0.6 8.2 1.2 8 8.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ast | 22.8 | 5 19.7 - | 5 8.6 26.8 18.8 16.5 26.4 7.4 21.1 14.2 27.6 25.8 25.2 38.4 24.3 23.8 18.6 | 0 21.2 32.5 0.3 29.9 1.8 5.3 23.1 22.4 18.6 22.7 2.9 27.3 10.8 16.3 22.9 | 20 12.7 1.3 3.3 8.6 | 1 14.2 | 17 | 1 0.4 7.6 22.7 17.1 25.2 10.7 8.4 9.5 23.8 2.2 1.5 19.4 21.3 1.9 19 0.2 | 8 18.5 10.1 33.2 26.3 21 5.2 21.2 19 | 2 23.1 25.8 18.9 1.1 16.9 17.7 21 21 | 9 0.8 27.8 0.7 10.2 1.1 14.7 10.2 |
| az |  | 7.4 3.7 1.7 2.6 - | 2.2 11.7 7.8 6.1 10 1.4 8.1 6.6 7.7 9.9 8.3 9.3 9.8 11.8 7.2 0.1 10.2 13.1 0.3 8.4 2.6 3.9 4.6 | 10 7.6 10.3 2.1 5.9 7.6 6.8 10.6 5.2 3.1 1.8 1.9 4.4 |  |  |  | 1 5.7 3.5 0.6 0.5 2.8 12.4 13.6 10.6 4.4 6.5 4.2 5.1 0.5 1.1 | 8 6.5 1.5 5.3 0.3 6.6 8.4 6.9 12.4 10.9 9.4 3.5 11.3 10.1 1.1 7.6 9.6 |  | 8 2.1 10.9 3.2 11.5 11.5 4.9 0.9 16.4 0.6 5.8 0.6 5.5 6.2 |
| be |  | 13.9 5.4 10.4 6.9 4.2 - | 19.4 13.1 12.3 13.9 7.4 14 10.1 13.7 15 15.4 16.5 14.2 14.7 11.1 0.1 12.5 17.2 0.3 |  |  |  | 16 4.3 5.3 12.7 15.8 13 12.2 5.4 12.6 8.6 11.7 15.5 11.9 8.9 4.8 5.3 9.1 1.9 9.7 12.4 0.5 0.3 7.5 15.9 9.4 18.3 9.3 7.3 8.7 11.5 2.2 3.5 14.7 11.4 1.8 13.4 0.3 7.5 13.9 |  | 7 16 16.5 18.4 6.3 14.7 | 14 2.5 15.5 15.5 13.3 2.5 12.6 12.2 8.1 18.7 | 8 0.9 17.4 0.7 6.8 1.5 10.6 7.2 |

| bg | 29.2 9.3 23.9 20.6 9.2 4.3 - |  | 23.9 28.2 28.5 10.6 30.4 18.7 34.2 32.2 29.6 41.8 26.3 29.6 25.2 0.1 25.5 38.9 0.3 31.3 5.9 9.2 27.8 27.9 28.3 27.9 9.3 31.7 12.9 22.3 28.7 24.2 16.6 3.9 7.5 13.5 3.3 19.9 |  | 22 0.7 0.5 11.8 29.4 17.1 35.9 17.2 11.1 15 27.6 4.7 | 1 27.2 26.3 2.6 26.3 0.4 13.7 24.7 12.1 37.5 32.7 30.5 11.5 31.2 28.8 4.4 30.2 33.9 26 2.9 19.7 19 25.2 29.5 15.7 1.7 33.7 0.7 11.2 2.2 19 11.9 |  |
|---|---|---|---|---|---|---|---|
| bn | 21.9 9.6 18.2 13.4 7.5 6.3 25.6 - | 18.7 21.1 8.3 21.3 15.2 24.8 22.4 21.5 30.8 19.2 | 21 20.2 0.2 18.9 28.9 0.3 22.3 8.4 8.1 20.2 26.7 19.7 20.9 8.4 24.8 11.8 17.5 20.9 21.3 13.7 5.2 | 5 12.5 | 5 16.7 15.8 0.6 0.5 11.3 21.2 22.2 24.3 18 10 15.6 22.9 5.2 1.5 | 20 18.8 2.5 20.2 0.4 15.1 16.9 11.4 27.2 25.2 20.6 11.8 21.6 20.7 3.3 20.5 24.3 20.7 6.5 16.5 13.3 19.9 19.6 15.1 1.3 27.3 0.6 | 10 1.9 14.9 10.5 |
| bs | 29.8 11.4 23.9 19.4 10.4 11.3 38.2 24.4 - | 28.6 16.6 33.7 19.5 35.2 33 28.8 41.7 26.4 30.3 25.2 0.3 26.7 | 40 0.3 32.5 12.4 9.7 28 28.4 34.1 29.1 13.1 33.6 13.8 23.1 | 29 23.7 18.7 11.1 12.7 14.5 5.9 19.1 23.4 1.3 0.5 12.6 30.6 23.2 37.2 18.8 11.6 16.3 29.5 | 7 | 7 27.5 27.5 3.3 28.1 0.5 13.4 25 12.4 38.1 34.7 30.3 12.8 | 34 32.9 5.3 36.4 34.7 27.6 9.1 20.2 22.5 27 29.5 15.9 1.9 34.1 0.7 12.8 2.8 18.6 13.2 |

| ca cs | ca cs | 28 10.2 21.7 27.6 8.5 9.1 33.8 21.8 24.3 - ceb 18.5 6.9 10.9 11.2 6.3 9.5 19.4 14.5 14.9 17.7 - 28.3 9.6 22.6 19.2 8.6 2.8 35.6 23.2 28 27.1 9.8 - | 13.7 27.8 17.7 33.4 30.9 27.7 42.5 26.6 26.1 22.2 0.1 23.8 39.7 0.3 32.8 6.4 9.5 25 25.8 25.9 24.7 8.1 31.6 12.3 20.5 29.4 22.6 15.9 3.2 8.9 12.3 3.8 18.1 21.3 1.1 0.5 11.6 25.9 25.6 30.3 15.7 9.2 13.5 27.3 5.4 1.6 25.2 25.9 3.1 28.2 0.5 11.7 21.8 10.6 38.1 33.4 27.1 10.1 15.6 14.3 17.6 17 16.6 27.8 17 15.8 13.5 0.4 13.5 23.2 0.3 20 6.7 8.8 13.8 16.7 14.7 15.6 5.4 21 10.8 13.6 17.9 14.9 14.9 4.9 8.3 9.9 3.7 10.6 14.8 1.8 0.9 17.5 34.4 32.9 28.1 40.6 25.8 29.8 23.2 0.1 26.6 38.4 0.3 30.4 5.5 9.5 27 26.6 29.5 28.6 8.2 31.7 12.9 22.4 28.4 24.2 16.9 3.5 5.9 13.2 3.2 19.3 21.9 1.1 0.6 11.8 29.8 23.2 | 32 16.7 10.8 14.1 28.1 4.1 0.5 27.2 | 28 26.4 4.5 26 32.2 25 2.3 9 15 15.3 18.8 11.1 7.3 10.3 17.6 5.1 8.7 15.3 14.6 4.3 17.3 0.4 8.7 13.2 8.7 22.8 19.3 15.7 8.8 16.3 15.5 5.2 16.4 17.8 17.8 5.6 13.2 22.6 13.4 15.2 9.6 1.9 20.8 0.9 10.6 2.9 11.5 11.1 26 2.9 24.4 0.5 12.4 25.6 12.2 36.6 33.8 29.6 10.6 36.6 31.1 3.6 27.6 33.5 |  | 18 18.7 24.3 25.4 14.2 1.3 31.6 0.9 11.2 2.3 17.7 11.6 26 2.5 19.8 17.9 25.2 28.2 14.8 1.6 33.3 0.7 11.2 2.2 19.5 12 |
|---|---|---|---|---|---|---|---|
| cy | 21.9 7.8 | 8 8.3 4.1 8.9 21.8 16.5 15.2 19.8 11.5 17 - | 22.1 18.5 18.2 29.5 17.2 18.8 15.8 0.4 15.2 25.9 0.3 21.3 7.5 6.7 17.2 19.6 15.2 16.7 6.5 20.7 12.1 15.8 18.1 10.6 15.5 6.2 6.6 11.5 3.9 8.4 17.1 1.2 0.6 6.9 16.7 14.9 | 21 12.3 8.3 10.5 19.3 | 4 8.5 14.9 17.9 3.7 19 0.4 7.5 13.4 9.5 25.6 22.7 15.1 9.7 18.4 17.6 4.3 18.6 21.4 19.5 | 6 13 18.3 13.1 | 17 11.1 1.2 22.9 0.7 10.4 2.6 10.2 11.5 |
| da |  | 33.7 8.1 24.8 20.8 8.1 5.3 37.8 24 28.4 30.7 14 32.3 20 - | 36.5 29.5 49.3 27.4 31.1 25.2 0.1 28.9 42.1 0.3 32.5 5.7 10.5 28.9 29.1 29.4 29.5 8.3 35.7 14 26 30.4 25.3 18.4 3.3 6.4 13.7 3.5 20.9 | 25 1.1 0.5 12.1 29.6 29.1 34.1 17.4 11.2 14.3 31.4 4.8 | 1 29.8 32.6 3.6 27.2 0.3 | 13 24.1 12.7 40.4 36.4 29.5 11.5 33.1 30.2 4.5 30.2 42.3 27.7 3.1 21.1 21.9 27.6 27.7 16.3 1.6 35.7 0.8 12.8 2.5 19.7 13.2 |  |

| de el | 31.6 10.4 24.3 21.3 8.8 7.2 37.3 23.6 28.4 29.7 11 32.8 18.7 38.8 - 25.8 8.6 21.7 19.9 8.2 3 32.5 21.5 23.9 25.1 11.2 27.1 16.7 30.8 29 - | 36.5 24.4 | 25 23.1 0.1 22.8 35.9 0.3 27.8 4.3 8.3 25.3 24.7 25.1 24.4 8.2 28.8 12.3 20.2 26.3 23.4 15.7 3.2 4.7 12.4 | 29.4 44.7 27.2 30.5 24.9 0.1 28.6 42 0.3 32.9 6.8 9.8 28.1 28.4 29.5 30.4 8.6 33.4 13.4 24.4 30.2 25.4 16.9 3.9 8.5 13.2 3.6 20.6 26.3 1.2 1.1 12.3 30.1 17.1 33.2 18.5 10.7 15.3 29.3 4.2 1.2 30.1 29.7 3.1 26.1 0.5 13.7 25.5 12.4 39.3 35.7 30.3 11.3 33.9 31.6 3.5 29.4 37.9 26.8 3.4 20.4 21.9 26.9 28.2 15.8 1.5 34.2 0.8 12.2 2.1 20.3 12.5 | 3 18.5 19.9 0.6 0.4 10.1 25.7 27.6 29.8 15.2 10.1 13 25.8 4.1 0.2 24.7 23.3 2.5 23.8 0.4 11.9 21.5 11.1 33.6 29.3 25.7 10.4 28.3 26.5 3.3 25.3 29.4 21.6 2.4 18.5 16.7 23.4 24.5 14.4 1.3 30.1 0.6 10.8 1.9 18.2 11.6 |  |  |  |
|---|---|---|---|---|---|---|---|---|
| en | 43.4 13.1 29.8 31.5 10.4 13 46.2 28.7 33.3 36.6 20.5 38.1 25.5 47.6 42.1 33.7 - | 30.3 34.8 | 30 0.1 31.6 51.4 0.3 39.4 9.9 12.2 35.3 35.9 33.4 34.7 10.4 45.4 | 15 26.8 33.6 29.1 23.1 3.4 10.6 15.6 4.7 23 28.1 1.2 0.9 14.4 34 36.9 40.4 20.7 | 12 18.5 39.7 6.9 2.9 32.3 37.2 4.2 37.9 0.5 15.2 26.5 14.3 51.8 43.5 35.3 13.4 39.4 34.4 5.2 | 35 47.1 35.1 | 3 23.4 26.6 32.8 33.7 19.1 2.1 40.9 0.8 15.6 3.3 23.3 | 15 |
| es | 23.1 8.9 20.4 18.6 7.8 3.1 30.6 20.9 21.8 23.4 10.3 25 15.9 29.2 27.1 26.3 31.1 - |  | 24.1 20.9 0.1 22.6 32.8 0.3 27.8 4.2 8.5 23.3 23.9 22.5 24.6 | 7 26.9 12.4 19.4 27.1 22.1 14.7 2.2 5.3 11.9 2.6 17.4 18.9 1.1 | 1 9.7 24.5 25.7 28.2 14.4 9.2 11.6 23.3 3.8 0.6 23.5 23.1 2.8 23.8 0.4 9.6 21.6 10.5 30.1 29.2 24.2 9.8 25.9 24.6 3.2 23.3 27.5 14.3 1.6 17.8 14 22.3 22.4 12.9 1.4 28.3 0.7 10.4 1.8 16.8 11.3 |  |  |  |
| et | 27.6 8.8 21.8 20.5 8.4 5.2 34.3 23.2 26.6 25.9 10.1 29.6 16.8 32.8 31.5 26.6 38.3 25.5 - |  | 23.8 0.1 26.1 36.9 0.3 28.6 6.9 9.2 25.9 26.8 27.5 25.9 8.4 30.7 12.8 22.2 27.1 24.2 16.8 4.3 4.4 12.9 3.8 19.2 20.9 | 1 0.4 12.3 29.5 30.9 30.7 | 17 10.5 14.4 27.3 4.9 0.9 26.6 25.4 2.7 23.3 0.4 13.2 23.7 11.9 34.9 32.3 28.6 | 11 30.9 28.9 4.1 27.2 32.9 23.6 2.9 19.7 19.1 24.7 26.6 15.2 1.6 32.6 0.6 |  | 11 2.4 19.4 12 |

| fa |  |  | 24.4 8.9 22.3 14.5 6.9 2.6 29.8 21.7 22.3 23.7 6.4 25.3 16.2 27.4 26.2 25 33.7 22.3 24.1 - |  | 0 21.2 | 33 0.3 25.6 1.9 8.7 23.6 25.8 23.3 23.6 8.3 |  | 28 12.4 19.2 24.4 21.8 14.7 3.7 2.9 12.4 1.6 17.9 16.4 0.7 0.5 |  | 10 24.1 26.3 27.3 11.8 10.6 8.9 24.8 3.1 0.3 22.3 21.8 2.1 20.7 0.5 10.6 19.7 12.2 31.7 28.2 23.4 | 11 25.9 24.3 2.9 22.3 27 20.7 2.8 18.5 8.6 22.4 22 14.7 1.2 30.4 0.7 10.5 1.6 17.5 11.3 |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ff | 3.3 1.5 1.2 1.4 1.3 | 2 3.7 | 3 2.4 3.2 2.4 2.7 2.6 2.9 2.8 3.2 3.9 | 3 2.9 2.1 - | 2.1 3.6 0.3 | 3 1.7 2.5 2.3 3.3 2.4 3.2 1.2 2.7 2.1 2.6 | 3 3.3 2.3 1.1 | 1.9 | 2 0.9 2.5 2.2 0.8 0.4 1.3 2.5 2.2 3.5 2.5 1.4 1.9 2.6 | 1 1.7 2.9 2.6 1.4 3.4 0.2 1.3 2.8 1.5 3.3 2.9 2.6 | 1 2.7 2.8 0.8 2.3 2.7 2.5 1.5 2.3 2.6 2.2 2.8 1.3 0.9 2.9 0.6 2.1 | 1 2.5 2.2 |
| fi | 25.4 8.3 20.4 17.7 | 8 | 3 31.2 21.4 23.3 24.5 7.9 27.4 15.5 | 31 29 25.3 35.3 24.3 26.3 21.4 0.1 - | 35.3 0.3 26.4 | 5.2 8.4 24.1 24.8 24.9 25.5 7.6 28.4 12.2 21.2 | 25.8 23.8 15 3.5 | 4.3 11.8 | 3.1 18.7 19.7 0.9 0.5 11 27 28.5 28.7 15.1 9.8 11.7 24.5 3.7 | 0.5 25.6 23.5 2.8 20.5 0.4 11.4 22.2 10.7 32.4 29.7 26.8 | 9.8 | 28 26.9 2.8 23.9 31.6 20.1 2.4 19 13.1 23.1 23.8 14.1 1.4 30.7 0.6 10.1 1.8 18.4 11.3 |
| fr |  |  | 30.9 9.7 24.6 23.8 8.8 4.4 36.7 23.4 27.9 31.3 13.5 31.6 19.3 36.9 34.5 30.4 45.5 28.2 29.2 24.9 0.1 27.3 - |  |  | 0.3 30.3 5.8 10.1 28.4 28.4 28.6 29.1 8.5 34.5 13.3 23.2 32.2 25.2 17.9 2.9 6.7 13.6 3.1 |  |  |  | 20 23.9 1.2 1.1 12.1 30 30.9 32.9 17.7 10.9 14.2 29.8 4.8 0.8 28.2 28.1 3.3 30.8 0.4 12.7 24.9 12.3 41.6 36.6 30.1 11.3 32.8 29.7 4.4 28.7 36.2 22.5 2.4 20.2 19.7 26.6 27.8 15.9 1.5 34.1 0.9 12.4 2.4 19.7 12.9 |  |  |

| ga | 1.2 0.2 0.2 0.2 0.3 0.5 | 1 0.4 0.3 0.9 0.6 0.8 1.4 1.1 | 1.1 0.5 2.6 | 1 | 1 0.2 0.1 | 1 1.5 - | 1.2 0.4 | 1 0.6 0.6 0.6 1.1 0.3 0.8 1.8 0.8 0.9 0.8 1.2 | 0 0.3 0.2 0.1 0.2 1.4 0.5 0.4 0.4 0.8 0.1 0.9 0.3 0.5 0.6 0.5 | 0 0.3 | 1 | 1 0.7 | 1 0.1 0.4 0.8 | 1 1.2 0.8 0.5 0.5 0.7 | 1 0.5 0.6 0.9 0.8 0.3 | 1 1.4 0.4 0.7 0.2 0.3 0.9 0.7 1.2 0.4 0.5 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gl | 28.9 9.3 22.9 26.9 8.5 9.3 | 35 22.5 25.5 29.2 14.7 29.2 17.7 33.9 | 32 28.6 41.5 28.2 | 26.5 | 23.8 0.2 24.6 | 34.4 0.4 | - | 5.4 9.2 27.1 26.7 27 26.2 7.9 32.7 | 13 21.9 29.8 23.5 17.2 2.6 8.8 12.8 2.8 19.2 22.5 1.2 0.5 | 12 27.4 29 32.2 16.3 9.9 14.3 28.5 5.4 1.3 26.4 | 26.7 | 3.4 25.7 | 0.5 12.3 23.5 11.1 | 36.8 34.2 28.3 10.9 30.2 28.7 | 4.1 27.6 33.7 | 25 2.3 19.5 20.9 25.2 25.9 14.8 1.6 33.3 0.8 11.3 2.6 18.5 11.9 |

| gu 1.9 ha he hi hr hu hy id ig is it ja jv ka kk kn ko | 1.9 1 24.9 11.2 20.3 15.3 24.1 7.4 12.3 5.2 5.9 4.3 3.3 11.9 6.1 3.5 km 15.8 6.4 5.3 7 1.8 0.9 | 0 0.4 0.7 0.5 1.4 1.5 0.8 1.2 0.6 1 2.3 1.8 1.2 0.7 1 1.2 1.2 0.7 0.1 1 1.9 0.3 0.5 - 0.7 0.5 1.5 1.2 1.2 0.4 0.9 1.9 1.3 1.6 0.6 1.3 0.5 0.6 0.7 0.6 0.5 1.9 0.2 13.2 4.9 9.2 6.1 4.6 7.2 14.3 11.2 10.1 12.5 8.8 10.8 11 13.6 11.2 11.6 17.7 11.9 11.7 10.8 0.2 9.8 15.9 0.5 13.7 7 - 11.3 13.5 10.8 11.3 7.2 15.1 9.5 10.7 11.6 10.6 11.5 5.6 6.4 8.4 4.6 28.5 9.2 23.5 18.9 6.7 2.5 34.5 22.4 24.9 26.9 6.8 29.3 17.9 32.7 30 27.5 40.1 24.1 27.3 24.6 0.2 24.7 38.4 0.3 28.9 3.6 9.1 - 27.4 26.1 26 9.3 31.2 12.7 8 6.7 29.8 25.5 21.3 22.9 8 24.2 17.2 27.9 25.4 23.2 37.1 20.9 23.4 22.7 0.1 20.4 32.4 0.3 25 7.2 9.4 22.3 - 21.7 22.9 9.5 27.3 12.3 18.7 22.5 22.4 14.5 5.2 28 10.4 22.4 19.3 9.5 4.6 35.7 23 31.6 27.2 10.9 31.6 17.7 33.4 31.6 27.7 38.3 25.8 28.8 23.3 0.2 25.6 37.8 0.3 30.1 6.8 9.6 26.6 26.7 - 26.5 9.1 21.5 18.9 8.8 4.4 32.5 22.4 24.8 25.7 10.1 29.5 16.5 31.3 30.9 26.1 36.6 24.9 26.1 22.5 0.1 24.7 36.3 0.3 28.1 6.2 9 24.7 25.1 26.2 - 16.5 5.9 8.2 5.4 6.1 7.5 21.6 14.8 13.5 16.8 2.2 16.5 12.2 17.9 17.4 16.9 22.2 15.2 17.6 14.3 0 14.3 23.4 0.3 18.3 5.6 5.3 14.1 18.6 14.8 15.5 - 29.1 10.5 22.8 19.7 5.7 4.2 33.7 23.9 25.3 27.3 12.2 28.4 20 33.3 30 26.6 42.7 24.5 27.4 24 0.3 24.3 37.3 0.3 29.1 3.8 11 26.2 28.1 26.8 26.7 8.6 - 11.8 4.7 8.6 6.3 4.9 6.6 13.3 10.4 9.2 11.2 8.8 10.5 9.8 12.6 10.7 10.9 16.3 11 10.1 9.8 0.2 8.6 15 0.7 12.2 7 5.3 10 12.3 9.5 10.3 6.6 13.4 - 10 11.1 18 15.3 6.8 6.5 27.3 19 20.5 22.7 10.1 23.4 16.2 29.2 25.2 22 31.7 20.9 23.4 18.9 0.1 21.4 31 0.3 24.3 5.1 9.1 20.4 22.2 22.1 21.7 7.2 26.2 12.7 - 25.3 10.2 21.7 19 8.6 5.8 33.4 22.2 23.9 26.9 12.7 27.3 16.5 31.2 30.1 27.2 35 27 26.1 22.5 0.1 24.2 37.8 0.3 30.3 6.1 8.9 24.7 25.8 25.1 26.9 7.8 29.2 12.5 20.9 - 18.9 8.3 17.3 12.8 7.3 4.5 24.2 20 17.1 19.1 7.8 21 12.6 22.6 21.4 21.1 26.1 19.4 20.9 18.2 0.1 18.7 26.6 0.5 20.6 5.6 6.6 19.1 22.5 18.6 20.2 6.8 23 10.1 16.6 20.3 - 20.2 7.8 12.4 11.5 7.2 9.6 21.9 17.3 16.3 17.9 14.8 18.8 15.3 19.9 18.5 18.2 28.7 17.3 17.3 15.6 0.3 15.6 24.3 0.3 20.5 8.3 8.5 15.8 19.3 16.2 17.6 7.2 27.5 11.4 15.4 17.1 6 16.2 13.4 9.5 13.1 2.2 12.5 10.3 13.4 12.8 13.9 15 13.2 14.6 11.8 0 11.4 18 0.3 14.6 5.1 3.8 12.1 15.1 11.6 13.3 5.7 12.1 8.8 9.8 13.7 9.5 7.1 - 5 4.6 6.1 14.5 11.3 9.9 12.1 5.1 9.9 10.5 11.4 11.3 11.3 2.6 11.8 13.3 9.8 0.2 10.6 16 0.3 13.5 5.3 4.6 7.8 13.7 10.5 11.9 6.3 9 4 7.4 18.5 14.4 12.9 16.2 6.8 14.1 12.7 18.2 14.6 15.7 17.5 14.6 15.8 13.3 0.3 13.4 21.6 0.3 16.7 6.3 7.2 12 16.2 14.2 15.5 6.5 15.7 11.1 13.2 15.5 9.6 10.7 4.8 4.7 - 0 0.4 0.5 0.4 1.7 1.4 1 1.4 0.6 0.8 1.9 1.4 1.2 0.9 0.8 1.2 1.2 0.7 0 1.1 1.9 0.3 0.5 0.5 0.6 0.5 1.6 1.2 1.3 0.2 0.7 1.9 1.5 1.5 0.8 19.8 8.6 17.8 12.8 8.2 5.6 25.6 20.5 18.2 20.2 6.9 21.4 13.2 23.8 22.5 21.5 27.9 19.1 21.7 19.4 0.1 19.5 27.1 0.3 21.5 5.9 7.7 19.3 23.2 19.6 21.2 7.2 24.9 10.9 17.4 20.4 25.3 12.5 4.2 | 11 10.3 22.5 20.7 13.9 3.3 6.5 11.9 16 - 8 10 12.5 7.6 8.2 3.9 - | 8 29.4 12.4 21.1 27.3 23.7 15.6 3.9 5.2 12.8 4.1 19.2 20.7 1.1 0.5 10.8 14 22.2 25.9 24.7 21.8 4.3 3.3 14.4 2.9 19.8 19.9 1.3 0.8 13.1 12 3.5 6.1 11.1 6.1 8.7 11.7 4.5 1 0.3 0.6 0.7 - 6 11.2 3.8 - | 27.8 8.7 31.4 12.8 22.1 28 23.8 16.2 4.2 8.7 13.5 3.8 18.8 21.4 1.1 0.4 12.3 29.3 14.2 34.6 16.9 10.8 14.3 27.5 5.1 16.7 10 12.9 16.2 12.4 8.4 6.2 5.6 7.3 2.6 10.1 11.7 0.3 0.4 6.7 17.6 19.2 19.7 9.1 8.5 9.4 14.6 0.5 4.4 3 16.1 17.6 1.1 0.7 11.3 23.7 15.4 2.6 8.2 12.7 2.9 18.4 20.9 1.2 1.3 11.8 26.4 3.1 7.1 12.3 | 12 15.4 1.4 0.7 9.9 17 18.1 20.8 12.9 8.2 11.2 23.6 1.6 6.7 2.4 7.5 11 0.4 0.3 6.3 14.4 15.6 16.2 7.9 7.9 2.8 5.8 10.3 0.5 0.5 5.4 14.2 13.5 15.7 7.3 8.7 7.7 0.6 1.6 0.2 0.1 0.7 1.5 15 0.7 0.7 8.9 21.9 11.5 23.5 15.5 9.8 13.2 21.8 | 0 0.7 1.9 1.2 1.8 1.2 1.1 1.5 1.1 0.4 0.5 0.9 1.2 0.7 1.4 0.3 2.8 21 26.2 22.8 16.1 3.5 1.9 13.1 2.8 18.9 19.5 0.7 0.4 11.4 27 29 31.5 11.9 10.6 9.1 27.7 3.8 0.5 24.9 25.5 2.1 23.6 0.4 11.1 22.1 11.9 36.1 32.5 27.4 10.8 29.4 5 13 4.4 17.5 16.9 0.7 0.5 11.4 23.4 25.1 26.8 18.4 10.6 18 25.7 5.2 2.4 21.5 21.9 2.5 22.1 0.4 17.3 19.1 12.6 27 26.8 29.8 16.5 10.5 13.6 26.4 27 29.2 31.4 12.8 11.1 7.8 33.2 5.2 0.6 26.4 26.3 3.5 5 5.7 8.3 3.9 7.2 9.5 1.2 0.6 5.3 9.4 9.8 11.7 8.6 4.3 7.2 12.6 3.7 6.8 10.2 10.1 3.2 11.7 0.6 23 20.6 25.1 14.2 14 30.2 17.2 10.1 14 25.5 3 20.2 14.6 0.6 0.5 10.5 21.3 13.7 22.5 14.7 9.3 12.8 20.6 3.6 1.9 20 17.8 | 7 11.1 1.3 0.6 4.7 10.4 11.2 13.1 9.3 4.8 7.6 14.4 4.1 7.8 10.8 11.5 2.9 11.9 0.5 5.4 10.2 6.2 15.3 13.8 11.5 6.6 11.6 11.2 3.9 11.1 13.2 14.1 7.4 9.1 13.8 10.6 1 27.6 26.2 3.5 25.1 0.4 13.4 24.2 11.5 35.4 32.6 28.9 5 0.9 26.1 24.6 2.7 23 0.4 12.1 22.5 11.7 34.3 30.8 26.9 25 0.4 12.9 21.8 12.6 36.4 32.5 26.4 9 12.1 23.3 4.1 1.2 22 22.5 3.2 20 0.4 10.1 18.5 10.2 29 26.7 21.6 9.6 24.4 23.3 3.8 21.2 27.4 22.2 3.2 16.4 17.8 20.8 5 0.6 25.7 24.9 3.3 23.9 0.4 12 22.9 10.7 34.4 7 7.8 10.7 0.4 3.1 11.1 10.9 1.7 13.6 0.2 7.7 10.7 7.9 16.6 15.3 10.3 7.3 13.3 12.9 1.5 12.7 13.3 11.4 3.4 10.7 9 1.7 5.4 7.9 9.7 1.9 1 1.6 1.1 1.3 1.1 0.9 0.4 0.2 0.7 1.1 0.6 1.3 0.2 2.1 1.1 1.4 1.5 1.3 0.3 0.9 0.9 1.2 0.5 1.2 1.2 3 2.2 20 18.2 2.1 17.9 0.5 11 17.3 | 1 1.3 1.3 1.5 0.6 1.2 1.1 1.4 0.6 1.4 1.3 1.4 0.6 1.8 1.6 0.7 27 3.7 26.9 31.8 31 27.9 22.9 13.4 24.8 23.5 3.6 22.8 26.9 21.8 5.8 18.2 12.2 22.7 14 13.4 1.4 15.8 0.3 9.2 13.3 9.5 21.7 19.8 16.7 8.6 17.5 16.9 1.6 16.3 18.1 16.1 4.2 12.1 10.9 14.1 2 17.5 0.5 11.7 17.6 9.3 24.7 23.1 20.1 9.4 20.7 19.6 2.3 18 21.7 19.3 5 4.5 16.5 16.7 3.8 19.7 0.5 8.4 14.7 9.5 24.4 22 17 9.1 18.9 17.3 5.3 17.8 20.5 20.8 7.1 14.9 18.3 16.5 17.1 10.1 1.8 24.4 11 0.5 7.5 9.7 7.8 15.6 13.7 9.9 5.5 12 12.1 2.4 11.8 1 0.5 9.6 15.6 15.8 18 8.6 7.4 7.8 15.2 2.1 6.3 12.4 14.2 2.7 14.2 0.4 9.4 12.8 8.5 19.9 18.7 13.1 7.3 15.5 15.7 3.3 14.5 16.1 16.7 4.1 14.6 15.5 9.5 15.2 9.4 0.8 22.1 0.7 9.6 1.8 10.4 10.2 | 1 1.2 0.1 2.2 0.6 1.1 0.3 0.6 1.1 11 7.5 1.3 15.3 0.7 7.9 2.6 8.4 8.8 23 2.5 19.1 9.2 24.4 25.3 14.2 1.3 32.2 0.7 11.2 1.8 17.8 11.9 22 18.8 1.4 29.8 0.7 11 1.8 16.8 11.5 11 32.3 31.9 4.2 26.2 32.8 25.6 2.7 19.5 20.2 24.8 27.8 15 1.8 32.5 0.8 11.8 2.3 19.4 12.3 10 29.4 27.7 3.9 25.1 31 21.5 2.8 19.1 16.3 24.3 24.3 14 1.8 31.2 0.7 10.9 2.2 19.4 11.6 17 10.8 0.6 21.4 0.5 8.6 1 9.8 9.2 11 29.2 27.3 5.1 25.7 32.5 25.3 3.1 20.9 13 25.4 25.5 15.3 1.9 35.8 1.1 12.8 2.7 20 13.3 5 8.8 5.4 13.8 12.4 10.3 6.2 10.4 10.1 3.5 10.4 12 12.6 6.9 8.8 12.3 10.1 9.8 6.9 1.3 14 0.7 7.2 2.5 8.4 7.4 21 12.5 1.3 28.1 0.8 10.9 2.4 15.4 11.3 32 26.3 10.4 28.2 27.2 3.7 24.9 30.4 24.1 2.5 18.5 19.6 23.8 24.5 14.2 1.5 31.2 0.7 11.1 2.4 18.2 11.2 3 17.2 15.2 18.6 17.5 12.9 1.1 25.7 0.6 8.4 1.5 17.5 8.8 1 10.1 2.7 12.6 10.8 8 9.3 13.1 8.9 0.6 16.2 0.5 6.9 1 7.7 7.3 11 10.5 3.5 10.9 8.4 6.6 13.8 7.8 1.1 15.7 0.7 7.1 1.4 8.3 7.9 1 0.5 1.7 1.1 0.6 0.8 1 0.1 1.5 0.6 1.1 0.3 0.7 1.2 10 25.8 23.8 20.2 9.7 22.1 21.2 2.2 18.9 23.2 19.9 3.6 17.3 14.2 20.5 19 12.9 1.1 27.1 0.6 8.9 1.7 17.2 9.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| lb | 27.4 8.7 14.9 15.4 | 8 12.8 29.3 19.3 21.8 24.2 15.7 24.5 16.5 31 33.9 23 37.3 21.9 24.7 18.1 0.2 21.2 33.3 0.3 27.8 6.7 8.5 19.8 21.4 21.9 23.4 7.5 24.9 11.7 19.7 24.4 17.4 | 16 | 5 9.8 12.1 3.9 | 14 - | 1.2 0.7 11.1 22.7 13.9 | 28 14.2 8.7 12 21.8 | 5 4.6 24.3 24.6 3.2 24.6 0.4 10.5 18.9 10.3 31.7 28.9 | 24 9.9 27.4 24 4.4 | 25 29.7 22.6 5.7 17.2 19.9 18.1 22.9 11.8 1.7 26.7 0.8 10.6 2.7 15.3 11.2 |
| lg | 3.7 1.9 2.1 2.1 | 2 2.7 5.4 4.4 3.3 4.6 3.5 4.2 3.7 4.4 4.4 4.5 5.3 4.7 3.9 3.1 0.4 3.1 5.9 0.4 4.2 2.5 2.2 3.7 4.3 3.4 4.2 1.6 4.5 2.9 3.7 4.8 5.6 3.1 | 1.5 | 2.9 2.3 1.4 3.2 | 3.3 | - | 0.7 1.7 3.5 3.1 4.9 2.8 1.9 2.2 4.4 1.3 2.1 4.6 4.4 2.3 4.2 0.3 | 2 3.8 1.8 5.2 4.5 3.8 1.5 4.1 | 4 1.6 3.5 4.5 3.9 1.5 3.3 4.3 3.2 3.8 1.8 0.8 4.4 0.9 3.2 1.3 | 4 3.6 |

**ln 4.6 2.2 3 2.5 2.3 3.6 6.3 5.1 4 5.8 3.8 5 4.7 5.4 5.3 6.5 6.7 6.2 5 3.6 0.4 4.7 7.6 0.3 5.6 2.8 2.9 5.2 5.6 4.5 5.9 2.5 6.2 3.4 5.2 6.1 5.9 3.9 2 3 3.3 1.7 3.9 4.2 1.3 - 2.2 4.9 4.5 6.2 3.7 2.5 3.3 5.5 1.7 3.1 6 5.2 2.6 5.7 0.3 2.7 4.9 2.2 7.4 5.4 5.2 1.9 5 5.2 1.9 4.6 5.7 5.2 1.8 4.5 5 3.5 4.8 2.3 1.1 5.8 0.9 3.4 1.4 5.1 3.5 lo 11.3 3.8 12.1 3.3 3.7 6.5 13.8 13.5 6.8 12.6 5.2 10.7 8.8 10.4 9.4 11.9 17.9 10.8 12.3 11.9 0 10.8 16.4 0.4 13.9 8.8 2.5 12.5 14.8 10 11.4 7.2 14.9 6.2 7.9 12.5 10.5 9.5 4.7 5.7 10.5 5.4 7.4 10.5 0.7 0.5 - 10.6 9 12.6 9.9 5.1 6.9 13.4 3.4 8.1 9 9.1 1.9 8.9 0.4 4.8 9.8 5.7 16.1 15.2 11.4 5.5 9.3 8.6 2.3 9.5 9.2 13 7.1 14.3 12.4 12.2 11.3 8.2 1 16.3 0.6 5.4 1.6 8.8 7.4 lt 25.4 8.7 21.2 17.3 8.8 7.5 32.2 20.5 24.5 24.4 7.9 27.2 16 29.9 28.1 25.2 34.6 23.8 27.4 21.8 0.1 23.6 35.1 0.3 27.3 5.8 9 24.3 24.8 26 24.6 8.2 28.5 12.2 20.6 26.3 23 15.3 3.8 4.9 12.2 3.8 18.2 19.7 1 0.4 12 - 27.9 29.4 15.2 10.6 13.1 24.9 4.6 0.6 24.9 23.2 2.7 21.6 0.3 12.5 23.3 11.3 32.5 29.7 27.5 10.2 28.2 27.2 3.8 25.2 28.9 21 3.3 18.6 15.1 23.1 25.6 13.7 1.8 30.6 0.7 10.6 2 17.9 11.4**

| lv | 25 9.4 21.5 7.8 9.2 3.7 16.8 22.1 5.9 25.2 8.7 19.1 15.4 29.9 18.7 26.6 36.7 24.6 | 29 22.8 | 0 26.3 36.5 0.3 28.4 4.5 7.4 25 25.9 11.5 25 7.9 29.6 12.5 21.3 18.6 19.5 15.4 3.6 5.9 12.3 1.9 7.4 18 0.7 0.4 10.3 28.6 - | 29 15.8 10.9 13.6 25.8 2.9 0.4 12.1 23.7 | 2 21.6 0.4 12.7 21.2 11.6 33.8 14.8 15.9 10.5 13.6 11.4 | 3 11.4 17.9 22.9 2.5 18.9 17.8 24.3 18.3 14.8 1.5 18.3 0.6 10.4 1.8 11 11 |
|---|---|---|---|---|---|---|
| mk 29.5 10.6 24 20.9 9.8 8.5 | 39 23.7 30.4 28.5 11.1 30.3 18.9 34.8 31.8 29.6 42.1 25.9 29.1 25 0.1 25.8 39.1 0.3 32 6.7 9.7 28.5 | 28 30.7 27.6 9.8 33.3 13.1 | 23 28.1 24.8 17.1 4.4 10.2 13.4 4.2 20.2 22.5 0.6 0.5 11.9 29.5 23.1 - | 17.6 11.3 15.3 28.7 4.8 1.4 27.1 26.9 2.4 26.4 0.4 13.1 23.8 12 37.8 34.2 29.5 | 12 32.2 30.7 | 4 31.4 33.2 26.6 3.7 20.5 19.5 25.7 28.5 15.8 1.6 33.9 0.6 12.3 2.3 19.6 12.5 |

| ml mn 16.5 mr my ne nl no | 19.5 9.1 15.6 11.4 6.6 7.4 22.8 20.3 15.4 17.9 8.9 18.5 13.9 21.7 19.2 18.3 26.8 16.9 18.2 17.2 0.2 15.7 25 0.3 19.3 8.7 7.3 16.8 22.8 16.3 8 14.3 7.8 7.9 10.5 19.8 17.1 13.9 15.3 8.7 16 12.2 17.9 15.9 16.3 20.9 14.9 16.2 16.4 0.1 18.4 8.4 14.2 9 6.9 5.9 21.3 20.8 15.1 17.1 9.3 16.9 13.1 19.4 18.4 17.5 26 16 17 16.2 0.2 14.6 22.7 0.3 18.1 9 7.3 15.6 24.5 15.6 16.1 7.7 19.8 10.7 14.5 17.5 17.7 ms 28.9 10 22.4 19.3 7.6 8.7 32.8 23.1 24.3 26.1 10.6 27.8 20.1 32 29 25.9 42.8 23.5 25.3 23.2 0.2 22.8 35.9 0.3 28.2 4.9 11 24.7 27.6 25 25.4 8.4 36.2 13.4 21.2 25.6 24.1 8.8 4.2 2.4 2.9 1.8 2.9 9.9 10.8 6.2 11.1 2.8 6.9 8.9 9.7 7.6 10.5 8 9.3 9.7 8.1 0.1 14.8 5.5 7.2 5.6 3.6 6 15.8 15.3 11.3 12.5 7.3 12.5 10.4 13.6 14.1 13.7 19 11.7 12.7 9.1 0.1 11.8 16.5 0.3 24.9 9.4 19.5 17.5 8.3 9.7 30.5 20.2 22 24.3 10.6 25 15.9 30 27.9 24.2 33.7 23.3 24.4 19.9 0.1 23.1 33.2 0.3 26.6 6.1 8.9 22.5 23.7 23.5 24.6 7.7 26.9 11.9 30.2 7.5 22.7 18.4 6.6 4 34.3 22.2 25.9 27.5 10.9 28.6 18.8 35.9 32 27.1 45.1 25.6 27.3 23.4 0.1 25.5 37.6 0.3 29.9 4.9 9.8 26.3 27.1 | 17 7.8 20.8 10.5 15.2 17.9 17.5 11.9 5.3 6 11.3 6.2 13.8 14 20.8 0.4 17.5 11.7 4.5 15.8 18.6 15.1 15.8 10.9 19.2 8.7 13.8 15.9 17.3 11.4 9.3 10.5 11.3 7.6 12.4 13.9 0.5 0.4 6.3 15.7 17.8 18.6 14.8 - 20 5.7 4.7 14.2 3.5 18.8 19.4 1.3 0.6 13.3 8 13.6 0.3 11.2 4.3 4.1 8.6 11.7 7.3 9.3 2.7 7.2 8.3 8.1 10.3 6.4 5.9 2.4 1.8 7.8 2.2 14 7.2 5.3 10.6 20.1 11.4 12.3 4.8 13.7 7.7 10.7 13.1 11.6 9.2 3.8 3.8 8.3 3.4 8.6 11.4 0.8 0.3 8.2 12.8 12.9 15.5 11.2 7.5 12.1 12.5 2.4 - 20 25 21.9 14.5 3.7 7.2 12 3.7 17.3 19.2 | 15 0.4 0.4 10.7 12 4.7 5.5 10.6 5.8 13.7 14.4 0.6 0.2 10.3 17.2 18 19.9 16.6 1 1.2 11.2 24.7 26 26.6 7.9 32.7 12.8 23.1 27.7 23.2 15.9 3.1 3.5 12.9 2.5 19.4 21.7 1.1 0.8 11.6 26.7 26.7 31.5 14.5 10.5 13.2 28.5 3.8 0.6 27.8 - | 18 19.8 20.4 - 11.9 17.7 5.7 11.5 14.6 14.7 9 - 18.6 4 2.7 25 26.3 29.4 14.9 10.7 13.2 - 4 7.8 0.4 0.4 5.6 9.8 8.2 12.1 5.6 6.1 6.4 8.7 - 14 27.7 16.2 9.5 13.3 23.7 4.7 1.1 - | 9.3 14.3 18.6 5.1 1.9 17.3 16.8 2.2 17.9 0.4 12.9 14.2 10.1 23.2 20.9 2 15.8 0.5 7.4 13.3 16 15.4 2.2 16.5 0.5 14.6 3.9 1.2 24.9 25.5 3.8 23.6 0.4 11.7 20.8 12.2 35.5 31.4 25.8 4.8 7.2 7.3 1.4 9.4 0.3 8.2 24.2 3.1 21.6 0.5 3.2 24.9 0.4 11.3 22.5 11.8 | 7 7.1 12.7 11.1 5.8 4.5 8.6 11.7 10.9 1.9 13.1 0.5 11.8 10.7 8.2 16.6 15.3 11.8 7.9 13.6 12.6 2.3 12.7 14.6 13.1 12 20.6 | 17 10.7 18.8 18.3 8 20.2 18.5 15.5 9.9 15.9 15.9 3.4 11 27.7 25.7 4.2 37 32.8 27.2 10.4 30.2 27.4 |  | 3 17.8 20.5 19 7.3 14.3 13.7 17.5 16.3 13.7 17 17.3 16.9 10.7 13.2 15.1 16.1 15.5 11.1 1.4 21.1 0.6 7.3 1.7 14 10.3 22.6 20.5 16.5 11.5 17.6 16.8 3.2 16.5 19.9 17.5 6.8 13.8 12.9 15.6 15.9 14.3 1.2 22.8 0.6 9.2 1.9 12.6 9.5 25 31.5 27.2 3.9 19.9 14.8 24.4 24.3 14.9 1.8 9 1.4 7.8 8.3 9.7 2.5 9.5 10.6 4.6 8.9 6.2 0.4 14.4 0.6 5 11.7 11.8 11 30.8 28 24 9.7 25.9 25 3.2 22.6 28.8 22.2 2.7 17.5 17.6 22.4 22.6 13.6 1.5 28.9 0.8 10.7 2.2 17.4 11.2 | 9 12.2 10.1 4 25.6 36.5 25.9 2.6 19.1 19.1 25 25.2 14.6 1.4 33.1 0.7 11.8 2.4 18.9 12.2 | 1 23.2 0.6 8.9 1.7 12.8 9.8 34 0.9 6 1 16.2 0.5 | 12 8.5 13 2.5 18.1 13.6 1 5.9 7.2 7 1.5 8.3 7.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ns | 5 2.5 2.6 2.6 2.5 3.8 7.5 5.9 4.7 6.4 5.5 5.5 5.7 6.7 5.8 7.5 8.8 6.6 5.5 4.2 0.5 5.4 8.2 0.3 6.1 3.6 3.6 5.6 6.4 5.2 5.9 2.8 | 7 4.2 5.2 6.9 6.6 4.9 2.1 3.5 3.5 2.2 3.7 | 5 1.8 0.9 2.7 | 6 5.2 6.8 3.8 2.8 3.6 6.9 2.1 2.8 6.5 6.3 - | 6 0.4 | 3 5.6 2.8 7.7 6.6 | 6 2.7 5.6 5.7 1.6 5.6 | 6 | 6 2.4 5.5 6.6 4.1 5.6 | 3 1.2 | 7 1.1 4.6 1.8 | 5 4.9 |
| oc | 29 8.8 22.5 22.1 8.6 6.7 34.8 22 23.6 29 15 28.7 18.1 32.7 30.8 27.9 47.5 26.1 26.2 22.2 0.4 24.1 41.3 0.3 28.6 5.2 9.8 24.1 26.7 24.7 25.7 6.6 33.3 12.9 19.9 27.9 23.2 18.4 2.8 8.1 |  |  | 13 2.9 18.7 22.5 1.5 0.7 12.7 25.6 26.7 30.7 15.1 10.1 12.7 26.2 5.2 0.9 25.2 25.5 3.3 - | 0.4 12.7 21.8 |  | 11 40.1 33.2 26.4 10.2 29.3 27.4 4.4 |  |  | 26 32.9 24.8 2.2 19.2 21 24.2 25 13.4 1.7 32.2 0.8 11.7 2.6 17.7 12.2 |  |  |
| or | 0.6 0.2 0.2 0.6 0.5 0.4 0.9 0.6 0.5 0.7 0.3 0.3 1.1 0.7 0.6 0.3 0.5 0.7 0.5 0.2 0.1 | 0.5 0.8 0.3 0.3 0.5 0.5 0.1 0.6 0.4 0.5 0.2 0.4 1.5 0.4 0.6 0.3 0.4 0.3 0.2 0.4 0.3 0.2 0.9 0.3 0.5 0.4 0.7 0.6 0.6 0.3 0.3 0.4 0.5 0.2 0.3 0.4 0.5 0.8 |  |  | 1 - |  |  | 0.7 0.6 0.7 0.6 0.6 0.5 0.9 0.5 0.5 0.4 0.4 0.4 0.4 0.3 0.4 0.9 0.2 0.5 0.5 0.4 0.4 0.6 0.6 0.3 0.2 0.6 |  |  |  |  |

| pa pl | 12.9 4.7 13.8 2.8 4.4 7.2 16.9 18.9 8.2 14.2 3.5 11.7 9.2 12.7 11.7 14.2 21.6 12.2 14.1 14.2 23 9.2 20 18 8.5 2.6 31.2 20.9 22.3 24.1 9.3 27.1 15.3 28.1 27.3 24.9 31.5 23.8 25.4 20.7 0.1 23.1 32.8 0.3 26.7 | 0 12.6 19.1 0.3 15.5 15.1 1.3 14 24.4 11.9 12.7 9.1 15.4 5.4 8.1 | 6 8.6 22.8 23.5 23.4 24.8 7.4 26.7 11.9 19.7 25.5 22 14.7 |  | 14 12 7.8 5.9 | 3 5.1 | 6 8.8 8.5 8.6 12.5 0.2 0.3 3.3 12.7 13.2 15.3 13.7 5.9 12 13.2 3.9 13.4 9.6 10.9 1.4 9.9 0.4 - 12 3.2 17.4 18.4 0.9 0.7 10.3 27 21.2 27.5 15.4 9.9 12.9 23.3 4.3 0.6 24.4 21.7 2.8 22 0.4 11.6 - |  | 10.3 7.8 19.1 17.1 13.1 | 9 11.2 10.7 1.7 12.5 11 13.6 9.3 12 14.6 13.6 13.8 13.1 0.8 15.1 0.5 5.2 1.1 9 6.5 10.5 30.4 28.8 25.3 9.7 27.2 25.9 2.9 21.9 27.1 21.7 2.4 17.6 14.6 21.2 23.6 13.7 1.4 29.1 0.7 10.4 2.2 17.7 11.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| ps | 17.6 8.8 13.7 11.5 7.1 10 20.2 17.7 14.4 16.3 12.1 17.1 12.8 19.3 17.7 | 16 22.6 15.2 16.4 16.5 0.2 13.8 22.4 0.5 17.8 13.3 6.1 14.7 19.8 | 15 16.1 | 11 18.7 10.3 13.6 15.8 15.2 | 12 | 9 9.6 | 11 8.3 11.4 14.4 0.7 0.3 8.4 15.8 17.1 18.4 14.9 7.4 12.1 17.4 6.3 12.2 15.1 14.9 2.3 17.6 0.6 10.7 | 13 - | 21.1 19.1 15.3 11.4 16.5 16.3 | 4 16 17.7 17 10.6 12.8 16.2 16.3 15.5 13.5 1.5 21.3 0.6 8.6 1.9 11.2 9.3 |
| pt | 32.7 10.8 25.2 27.6 9.6 5.7 38.5 24.2 29.2 32.1 13.8 33 20.9 38.3 35.3 30.6 49.8 29.1 30.4 | 26 0.1 27.5 44.9 0.3 34.9 6.3 10.3 | 30 29.3 29.5 29.5 | 9.4 36.5 13.9 23.8 31.7 25.6 | 19.2 | 3.7 7.4 | 14.1 3.7 21.4 23.9 1.2 | 1 13.5 29.7 31.1 34.6 16.6 10.9 14.4 31.4 5.4 0.7 28.6 29.9 3.6 31.9 0.4 13.8 24.5 12.7 - | 38.3 30.5 11.8 33.3 30.4 4.9 30.9 37.3 | 24 2.7 20.7 20.6 28.5 29.4 16.2 1.8 35.7 0.8 12.9 2.4 20.1 13.2 |
| ro |  |  |  | 30.9 11.4 24.6 22.9 10.1 8.6 38.5 24.4 29.2 30.5 14.1 33.3 18.8 37 34.5 30.3 44.6 28.3 29.9 25.7 0.1 27.5 43.8 0.3 34.4 7.3 9.8 29.2 29 29.9 29.7 8.9 35.2 13.4 23.5 31.9 25.6 18.7 3.5 9.5 14.1 4.1 20.6 24 1.2 0.8 13.3 30.9 17.4 34.8 18.2 11.3 15.9 30.4 5.7 1.3 28 29.1 3.3 28.7 0.4 13.6 25.6 12.4 41.3 - |  |  |  |  |  | 31.1 11.8 33.5 30.9 4.4 29.4 36.3 28 3.3 20.5 20.6 27.7 29.3 16.2 1.7 35.4 0.7 12.7 2.6 20.6 13.2 |

**ru 25.4 9.9 21.7 18.4 7.9 7 35 22.3 25.1 25.4 9.3 29.3 16.2 31 29.9 27.2 35.7 24.9 28.3 23.1 0 24.8 36.1 0.3 28.9 6.4 8.2 25.4 25.6 26.6 26.2 8.9 29 12.3 20.9 26.8 23.8 15.1 3.9 6 12.9 2.9 18.9 20 0.6 0.8 11.2 29.5 19 30.7 16.8 10.7 14.2 25.5 4.2 1.3 24.8 23.9 2.2 23.3 0.5 12.7 23.5 10.9 34.6 31 - 10.9 29.4 28.6 3.2 26.3 30.3 23.8 2.8 18.9 17.6 22 33 14.9 1.6 31.5 0.6 10.7 1.9 18.7 11.6 sd 4.8 2.2 0.8 1.9 1.6 1.5 4.7 4.4 2.8 4.9 1.5 3.8 4.8 4.8 4.1 3 6 4.2 5.4 3.4 0.1 4.6 6.1 0.3 4.3 2.1 2.9 1.6 6.4 3.1 4.5 1.2 2.9 5.5 3.8 4.1 2 2.1 1 1.1 3.5 0.9 2 2.7 0.5 0.6 2.6 5.4 6 4.5 2.7 3.3 3.1 2.5 0.8 1.1 3.7 3.5 1.2 3.7 0.3 7.1 3.3 6.2 5.9 4.8 3.5 - 4.7 4.6 1.3 3.8 4.7 4.7 1.5 6.4 3.8 2.8 3.9 5 0.4 6.4 0.6 3.9 0.6 1.6 4.2 sk 28.4 9.8 22.8 19 9.5 3.2 35.8 23.2 28.6 27.8 11.8 36.8 17.7 34.7 32.9 27.7 40.4 26.2 30.3 24.2 0.1 27.3 38.2 0.3 31 7.1 9.4 27 26.9 29.8 28.3 8.8 32.3 12.9 23.1 28.8 24.1 16.7 4.1 6.7 13.4 3.7 19.5 22.7 1.1 0.6 12.4 29.8 16.1 32.9 17.4 10.4 14.8 28 5 0.9 28 26.9 3.1 25.4 0.5 12.9 25.4 12.1 36.7 33.5 29.8 11.3 - 30.5 4.5 28.5 32.9 25.9 2.9 20 19.1 25.7 28.1 15.2 1.7 33.3 0.7 11.6 2.5 18.8 11.8 sl 27.1 9.4 21.7 18.3 8.8 3.5 33.6 22.4 27.4 26.1 11.1 29.4 17 32 30.5 26.4 36.7 24.7 28.1 23.1 0.1 25.5 36 0.3 28.9 6.6 8.7 25 25.7 29.2 26.6 8.4 30 12.9 21.4 27.8 23.5 16.1 4.1 7.3 13 4.3 18.8 21.6 1.1 0.6 11.9 28.4 12.3 32.1 16.3 10.2 14.3 26 4.9 1 26.9 24.3 2.9 23.8 0.4 13 23.4 11.4 34.3 31 28.1 10.8 30 - 3.9 26.7 30.2 24.6 3.2 18.8 18.4 24.5 26.9 14.6 1.7 31.9 0.6 11.1 2.2 18.2 11.6 so 4.3 2.1 0.3 1.9 0.7 1.8 3.8 3.8 2.2 4.7 3 2.7 6.6 4.2 2.7 2.6 4.2 3.1 4.8 1.3 0.3 2.4 5.8 0.3 3.7 2.3 3.4 1.2 3.4 2 3.1 0.7 2.1 5.5 3 3 2.6 4.7 1 1.9 3.9 1.2 1.8 3.9 0.9 0.4 1.6 5.2 3.7 4.6 1.4 3.4 1.7 1.6 0.7 2.7 2.2 2.7 2.4 4.1 0.3 3.6 3.1 4.3 3.3 4.5 1.4 2.5 3.5 3.7 - 2.7 3.8 3.8 1.9 5.3 5 1.6 2.7 2.4 0.6 2 0.7 5.4 0.9 2.1 6.2 sr 30.1 11 24.1 21.7 9.7 4.9 39.3 24.6 32.9 28.1 13.7 32.3 18.7 35.7 33.2 29.9 41.9 26.2 29.9 26 0.2 26.2 40.4 0.3 32.3 7.9 9.4 28.5 28 32.1 28.3 10.2 33.2 13.4 22.8 28.1 25.4 17.6 5 7.3 14.3 4.2 19.7 22.9 0.5 0.5 13 30.3 14.1 37.7 19.3 11.6 15.9 28.7 5.1 1.5 26.6 26.9 2.7 27.8 0.5 13.5 24.9 12.4 38.3 34.6 30.7 12.4 33.4 32 4.5 - 34.4 27.5 4.6 20.3 21.1 26.5 30.3 16.1 1.5 34.2 0.6 11.7 2.3 19.6 12.8 sv 33.6 11 23.8 21.6 9.7 11.6 38.4 24.7 29 30.1 14.5 32.7 19.7 42.9 36.9 29.2 48.7 27.5 31.1 25.4 0.2 30 42.5 0.3 33 9 10.5 28.5 29 30.1 30 9.4 35.6 13.6 25.4 30 25.1 18.9 4.7 9.2 13.9 4.9 20.5 24.9 1.2 1.5 13.6 30.1 18.4 34.6 18.4 10.9 15.9 31.1 5.9 1.6 30.1 33.4 3.2 27.5 0.5 13.8 24.4 12.3 40.6 35.8 30.3 11.7 33 30.4 4.5 30.4 - 27.9 3.6 20.3 21.2 27.2 29.2 15.9 1.8 35.2 0.9 12.6 2.6 20.4 13.1 sw 25.4 8.8 18.4 13.3 7.1 11.6 29.3 20.9 21.4 23.1 10.1 24.8 17.2 27.8 24.6 22.5 37.5 20.2 22.7 20.8 0.1 20 32 0.3 24.4 6.5 10.5 21.7 24.6 21.9 21.1 9.1 29.2 14.4 19.2 22.2 20.8 16.7 4.9 6.9 12.9 4.5 15.5 17.7 1.2 0.9 11.9 22.8 24.2 26 15.3 10.1 13.1 26.3 5.4 2.6 21.3 21.5 3.4 22 0.4 12 18.2 11.1 30.7 27.7 22.8 11.1 25 23.7 2.3 23.4 27.5 - 4.9 17.3 19 20.9 21.7 13.8 1.4 29.3 0.8 12.8 2.1 15.8 13.7 ta 9.8 4.9 3 3.3 3.2 3.4 12 10.4 6.6 9.9 2.8 8.4 8.3 9.3 9.7 10.4 13.1 9 10.1 8.2 0.1 8.7 13.4 0.3 9.7 4.3 4.6 6.7 12.2 7.9 9.6 3.6 8.9 8.3 7.8 9.4 6.1 5.2 2.7 1.9 6.5 2.8 5.8 6.1 0.4 0.3 5.5 10.3 11.1 10.5 8.1 6.2 7 8.6 1.6 1.3 8.2 7.3 1.6 7.6 0.3 8 7.5 7 12.6 11.4 8.4 5.7 10.1 9.3 1.6 9.1 10.3 10 - 8.9 5.3 7 8.9 7.3 0.5 11.7 0.6 6.4 0.8 5.3 7.1 th 20.3 9.5 18.4 14.9 8.9 12.9 25.9 21.1 18.5 20.4 13 22.3 14.1 24.2 22.4 21.3 27.4 19.7 21.1 19.6 0.1 19.9 28.5 0.7 23.1 12.3 7.5 19.9 23.1 19.8 20.7 11.6 26.1 11.6 18.1 21.2 21.6 14.9 10.5 10.9 13.7 7.7 16.7 16.6 0.8 0.4 12.9 20.9 23 23.5 17 9 14 22.6 6.7 13.4 20.2 19.3 2.7 20.4 0.6 10.9 17.7 10.1 26.2 24.8 21.1 10.9 22.3 21.1 4.2 21.3 23.5 20.6 11.5 - 18.5 20.5 19.9 12.9 1.5 28.1 0.6 9.5 2.2 16.7 10.3 tl 24.2 8.7 16.1 13.7 6.8 10.2 26.4 19.3 19.4 22.3 19.2 22 17.8 25.2 23.7 20.7 34.1 20.5 21.3 18.8 0.1 17.7 30.2 0.3 24.6 5.6 9.7 20.2 23 20.1 20.1 8.4 28.4 13.3 17.6 21.4 18.9 17.7 5.5 7.2 12.6 4.1 14.3 18.6 1.4 0.8 10.7 20.1 22 24.5 14.9 9.3 12.5 25 5 2.3 19.7 20.6 3.8 23 0.4 9.5 16.9 10.9 29.7 25.6 20.6 10.6 22.1 20.7 5 21.7 24.5 23.1 5.9 15.9 - 20 20.8 13.3 1.6 27.2 0.8 12.2 3 14.4 13 tr 26.5 10.5 21.8 15.3 9.6 1.8 32.2 22.8 23 25.2 6.6 26.7 16.6 31 28.4 25.9 36.9 24 26.8 23.6 0.1 23.5 35.1 0.3 28 2.6 9 24.7 26.3 24.8 25.8 9.6 30.2 12.7 20.8 26.2 24 15.7 5.6 1 12.5 2.6 18.4 17.6 1.2 0.8 10 26.1 26.9 28.7 13.4 11 9 26.9 2.1 0.3 25.1 23.8 2.9 22.9 0.4 9.9 21.1 11.7 34.2 30.8 23.5 10.2 28.1 26.1 3.9 24 30 21.9 3 19.4 9.6 - 23.9 14.5 1.8 31.7 0.8 11.1 2.2 18.4 11.7 uk 27.6 10.1 22.9 19.7 10.1 10.6 36.4 22.9 26.9 26.5 11.9 30.8 17.5 32.4 31 27.5 38.5 25.2 28.2 24 0.1 25.4 37.3 0.3 29.7 7.5 9.3 26.6 26.7 28.2 27 9.9 31.3 12.5 21.6 27.4 23.9 16.4 4.6 10.3 13.5 4.2 18.9 21.6 0.7 0.5 12.7 29.9 16.9 33.1 17.3 11.5 15.3 26.6 4.7 1.4 25.7 24.9 2.5 26.3 0.5 13.3 23.9 11.6 35.7 33.1 33.9 11.6 31.2 29.1 4.5 29.3 31.6 25.5 3.2 18.8 19.5 24.8 - 15.2 1.6 32.4 0.7 11.3 2.2 18.6 11.8 ur 16.5 6.9 11.6 7.3 6 5.8 20 18 13.6 16.6 5 17 12.2 18.2 17.3 16.3 23.8 15 15.9 15.6 0 14.3 22 0.3 16.5 7.3 6.5 14.8 22.9 14.7 15.5 7.7 18.1 9.9 13.4 16 14.8 9.5 4.8 3.7 9.9 4 12.3 11 0.6 0.4 8.7 16.1 17 18.1 13.3 8 12.2 15.9 3.1 1.7 15 14.3 1.8 15.3 0.3 14.7 12.8 10.9 21 19.2 15.3 12.7 17 16.6 2 14.8 18.6 15.4 5.7 13.1 10.2 14.5 14.4 - 0.9 21.6 0.5 8 1 11 8.5**

| uz | 3.1 1.5 1.2 1.6 1.8 1.3 4.2 3.2 2.2 3.4 | 2 2.3 3.1 3.1 3.2 3.2 2.3 3.7 3.7 1.9 0.1 3.1 4.7 0.3 4.4 1.7 1.7 1.6 3.3 2.4 3.5 0.8 2.3 2.5 3.3 3.3 | 3 2.2 | 1 2.4 | 2 0.9 1.9 3.3 0.6 0.6 1.6 3.4 | 4 4.5 2.2 2.1 2.2 | 2 0.8 0.8 2.5 2.7 1.5 4.1 0.4 2.8 2.5 1.9 4.1 3.1 3.6 1.3 | 3 | 3 1.2 2.9 2.7 2.5 0.9 3.6 3.1 | 1.8 2.8 1.9 - | 3 0.7 2.1 0.6 2.6 2.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vi | 25.4 9.2 21.5 17.7 9.1 10.4 31.5 21.9 23.2 24.4 9.5 26 17.1 30.1 27.2 25.1 36.1 22.8 25.1 22.2 0.1 22.8 33.6 0.3 27.2 5.3 9.5 23.7 25.2 24.1 24.2 8.5 32.2 | 13 21.3 25.2 23.9 | 18 4.2 | 7.3 13.5 | 4.2 19.5 18.5 1.2 0.8 13 25.4 13.5 | 28.5 14.7 10.3 12.5 27.2 | 4.5 1.5 23.9 24 3.3 22.8 0.4 12.6 21.2 11.3 32.9 29.5 25.9 10.4 26.8 | 25.5 | 3.8 22.8 | 29 22.3 2.6 19.7 11.9 24 24.3 14.4 1.9 - | 0.8 11.7 2.4 19.3 12.2 |

| wo | 5.2 2.3 3.2 2.2 | 2 3.5 6.4 4.8 4.2 5.6 4.1 4.7 4.1 | 5 | 5 6.2 6.6 | 6 4.7 3.7 0.4 4.3 6.9 0.3 5.7 3.1 2.7 5.1 5.9 4.6 5.7 2.5 | 6 | 3 4.7 5.6 | 5 | 4 2.3 2.6 2.9 1.8 4.3 4.1 1.3 0.5 1.6 4.9 4.4 6.1 | 3.2 2.4 3.1 5.4 1.3 | 3 5.7 4.9 2.4 5.6 0.3 2.8 | 5 1.9 6.8 5.5 4.9 1.7 | 5 4.7 1.4 4.3 5.2 4.8 1.8 4.5 4.5 3.6 4.5 2.6 1.1 | 6 - | 3.4 1.2 5.1 3.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xh |  |  | 17.6 8.4 13.3 11.5 6.5 9.2 19.6 15.9 14.1 16.5 12.4 15.5 14.6 18.6 15.9 15.4 25.3 15 15.1 14.3 0.3 12.7 22.3 1.4 17.9 |  |  | 11 7.5 14.6 17.7 14.1 14.3 9.3 20 12 13.2 15.3 14.9 13.9 |  |  | 8 | 9 10.7 6.6 10.5 13.9 1.3 0.7 8.4 15.1 15.4 17.8 13.5 7.6 10.8 19.5 5.9 10.4 14.7 14.8 3.5 17.3 0.7 7.9 12.5 8.7 21.4 18.3 15.1 9.1 15.7 15.3 4.9 |  |  | 15 17.1 18.5 9.4 11.5 17.6 15.4 14.8 10.5 1.9 20.8 0.8 - |  | 2.9 10.9 13 |

**yo 5.1 1.3 1 2 1.6 2.5 5.7 3.9 3.5 4.4 4.1 3.3 4.6 4.2 3.6 4.3 4 4.5 4.3 2.8 0.3 3.2 5.3 0.3 4.6 2.2 3.2 1.6 3.8 3.8 3.8 1 4.6 4.1 4.2 3.7 3.5 4 1 1.8 2.7 0.9 2.2 4.4 1.2 0.6 1.7 3.6 3.1 5.7 2.1 2 1.9 3.3 1 2.8 3.7 3.8 2.3 5.4 0.3 2.3 3.7 2.7 4.5 4.7 4 1.8 4.2 4.3 1.7 3.6 4 3.9 1.3 3.9 5.5 2.1 3.9 2.3 0.7 4.5 0.7 3.7 - 3.4 4.4 zh 20.3 8.8 18.9 12.5 8.2 8.2 26 20.2 19 20.8 6.4 22.2 13.2 24.2 22.1 22 26.8 20.2 21.6 19.6 0 20.4 29 0.3 23.3 5.6 7.1 19.9 22.4 20.4 22.3 8 24.8 11.3 17.6 21.6 22.9 13.3 4.8 6.5 12.1 4 18.4 14.8 0.7 0.6 9.7 22.8 11.3 23.8 14.8 9.5 12.8 22.1 3.9 1.6 20.4 19.2 2.2 19 0.5 10.1 18.4 10 26.8 24.7 21.5 9.9 22.4 21.5 2.3 18.3 23.6 20.4 2.6 18.5 14.3 20 19.6 13 1.2 29.1 0.7 8.8 1.6 - 9.9 zu 17.9 8.3 13 10.6 6.7 9.2 19.6 15.3 14.3 16.3 12.4 15.9 14.9 19.1 16.4 15.5 24.8 14.6 15.9 14.6 0.1 12.7 22 1.2 17.8 11.1 7.4 15.1 17.8 14.1 14.5 9.4 20.6 12.1 14.1 15.1 15.3 13.9 7.7 9.6 10.9 6.9 10.5 14.5 1.2 0.5 8.9 15 15.7 17.5 13.4 7.4 10.8 20 5.7 10.6 14.5 15 3.2 17.6 0.6 8.1 12.8 8.3 21.6 19.2 15.4 9.5 15.6 15.5 5.1 15.6 17.6 19.9 9.9 11.5 18 14.9 15 10.8 1.6 21.1 0.8 12.7 2.9 10.9 -**

Figure 9: Evaluation results of 12B M2M-100 on a subset of FLORES-101 devtest set. The i-th row is the source language, while j-th column is the target language. There are 87 languages and 7,482 directions.

-----

**af am ar ast az be bg bn bs ca ceb cs cy da de el en es et fa ff fi fr ga gl gu ha he hi hr hu hy id ig is it ja jv ka kk km kn ko lb lg ln lo lt lv mk ml mn mr ms my ne nl no ns oc or pa pl ps pt ro ru sd sk sl so sr sv sw ta th tl tr uk ur uz vi wo xh yo zh zu af - 22.5 28.1 20.7 16 17.5 38.9 29.3 31.5 30.8 25.5 33 44.1 41.5 36.7 29.7 60.5 27.2 30.8 27.6 0.2 27.5 44 33.7 34.2 20.3 15.3 34 33.4 31.8 29.6 29.4 38.3 16 25.2 29.7 27.5 16.6 25.5 15.7 9.7 14 21.6 11 10.7 18 8.1 30.1 12.2 36.3 28.1 17 23 36.7 9.6 3.5 29.7 32.2 3.9 31.1 10.5 22 25.1 11.8 43.4 37.4 30.5 10.7 33.6 31.2 6.5 34.9 39.2 31.4 16.4 22.7 28.1 31.8 31.5 19.5 25.6 36.5 2.8 21.5 2.6 24.3 17.4 am 23.6 - 20.2 12.4 12.4 9.7 27 23.9 20.2 21.8 19.2 23 27.8 26.7 23.7 21.7 32.5 19.9 21.9 22.3 0.5 18.7 29 22.5 23.2 16.5 12.5 22.5 25.5 21.4 21.7 22.5 25.9 14.5 18.3 20.8 21.7 12.8 19.7 11.9 8.6 13.3 17.3 6.2 9 15.4 7.8 21.7 16.2 25.7 22.9 13.1 19.1 24 9.3 9.8 20.8 21.5 2.5 19.9 9 18.1 17.6 9.5 28.8 25.7 20.8 7.6 23.3 22.2 5 23.8 25.4 24.6 21.3 16.9 21.8 22.8 21.5 17.6 20.5 27 2.4 16.3 1.9 18.7 13.9**

**ar 29.5 22.6 - 14.3 15.3 11.1 34.9 27 27.5 27.6 22.3 29.9 35.9 34.4 31.2 28.3 44.2 25.5 28 28.5 0.8 25.1 38.3 29.7 30 17.1 13.2 30.8 29.9 28.1 27.3 26.2 33.3 14.1 24.6 27.9 26.8 14.9 24.4 10.8 8.7 13.5 21.5 6.9 9.5 16.8 8.4 28.6 13.6 14.4 9.6 14.9 6.3 30.6 10.5 0.1 26.1 26.2 2.6 26.6 9.8 20.7 22.6 9.3 36.6 33.1 27.9 8.5 30.6 28.8 5.5 29.5 33 24.8 4.3 21.5 10.2 27.5 27.9 14.9 23 33.5 2.6 18.3 2 23.3 16.6**

**ast 28.9 6.7 24.3 - 13.6 14.9 32.9 25.4 24.6 27.7 18.6 28 31.1 32.8 30.3 27.2 39.8 24.4 27.3 25 0.4 23.7 37.8 23 31.9 16.3 12.1 27.9 27.1 26.3 26.2 24.4 30.8 12 22.5 28 25.5 13.2 22.1 12.6 7.5 12 19.8 8.5 8.7 16.6 8.4 27.2 11.8 31.7 4.4 13.1 11.8 28.7 8.3 3.7 25.6 25.6 3.4 27.8 8.5 17.2 22.9 8.6 36.6 31.9 26.4 8.4 28.6 27.5 5.2 29.4 31.8 18.3 5.7 18.5 23.1 26 26.8 13.9 18.8 31.4 1.9 15.7 1.8 21.8 8.5 az 18.5 16.7 11.6 11.5 - 4.8 24.1 19.9 17.5 19.1 15.8 18.9 23.5 22 20.3 20.6 20.7 18.6 20 19.6 0.2 18 26.1 19.6 19.4 13.3 9.3 18.5 21.1 18.3 20 12.2 20 10.9 14.8 20.1 14.2 7.1 21.5 6.5 5 9.5 15.1 5.8 6.5 13.4 10.1 20.2 9.1 22.8 17.5 10.8 10.9 16.9 7.8 2.2 18.5 18.2 2.3 10.5 7 14.8 16.7 6.9 24 22.5 19.5 5.6 20.3 20.6 3.8 19.6 20.8 15.7 10.9 15.9 9.9 20.2 18.8 6.2 16.9 20.2 1.8 13.1 1.3 16.2 6.4 be 17 15.4 9.7 12 8.8 - 16.6 15.8 15.4 18.1 14.9 10.7 22.5 20.2 18.8 19.1 22.1 18.2 19.1 12.2 0.1 15.6 23 19.3 20.5 13.5 8.4 13.4 17.9 17 17.1 16.2 15.5 10 14.6 19.6 14.2 7.9 21.7 8.4 6.3 9.6 11.4 5.7 6.5 12.1 4.6 20.9 0.1 21.8 12.8 10.2 11.6 16 5.8 3.2 17.6 15.8 2.4 14.8 5.7 14.2 10.9 7.3 21 19.2 18 6 11.2 12.5 3.8 14.6 18.5 15.6 9.1 14.3 11.7 3.3 20 11.9 16.1 19.7 1.4 11.8 1.3 13.6 7.2 bg 31.1 24.3 27.6 17.1 16.8 4.3 - 28.2 30.9 28.8 22.9 32.1 37.5 36.4 34.3 30 45 27 31 28.4 0.9 26.9 40 31.3 32.3 19 13.5 31.9 31 30.5 29.9 29.9 33.8 14.8 25.5 29.9 27.8 15.6 28.4 17.3 9.4 14.1 22.2 8.3 9.6 17.4 7.4 30.6 17.1 37.5 27.9 16.9 21.3 30.7 11.8 3.6 28 28.2 3.1 27.9 11.5 22.1 25.8 11 38.1 34.9 31.4 9.7 32.4 30.9 5.7 34.3 35.7 27.5 14.7 21.5 25.2 29.5 31.5 19.8 26.1 34.1 2.9 19.3 2.4 24.8 16.9 bn 25.4 21.3 22.5 13.8 14.2 7.3 30.1 - 23 23.9 20.3 24.9 30.9 29.2 26.5 23.3 38.8 21.8 24.7 23.9 0.5 21.3 32.6 26.1 25.3 20 12.1 24.8 30.2 24 23.5 22.4 29.3 13.6 20.7 23.7 24.7 12.8 22.7 11.2 8.3 15.1 19.9 7.2 8.3 15.3 7.1 23.9 9.2 28.2 27.5 13.9 22.8 27.3 9.4 13.8 23.4 22.8 2.5 20.8 11.4 21.7 19.6 9.7 30.9 28.5 23.8 9.6 25.9 24.2 4.9 25.9 28 23.3 22.8 18.7 21 24.4 23.5 18.2 21.2 29.4 2 17 2 20.6 13.7 bs 32.8 23.9 28.3 17.6 17 12.2 40.3 28.4 - 29.8 24.2 35.1 38.3 38.8 35.4 31 47.9 28.1 32.4 29.4 0.4 29.1 42.3 32.3 33.7 18.6 14.4 33.3 31.8 35.5 30.9 30 36 15.3 26.3 31 28.9 17 28.1 16 9.8 13.4 23.3 9.1 10 18.1 9.3 32.3 16.3 39.2 26 16.3 14.1 32.7 12 1.1 29.5 29.4 3.5 27.5 10.8 21 27.4 11.1 40.3 36.5 31.9 8.6 36.1 35 6.3 36.7 37 29.1 13.3 23.1 25.7 31.4 32.4 19.9 26.1 35.7 3 20 2.5 25.8 16.8 ca 30 19.1 25.7 20.8 14.7 16.7 35.7 26.1 28.1 - 22.8 29.2 36.4 35 32.5 28.1 46.2 26.9 27.7 25.3 0.5 24.7 41.6 28.6 35 18.7 13.2 29.6 28.7 28.6 27 25.7 32.9 14.9 23.1 30.6 26.5 14.8 23 13.9 8.5 13.5 20.6 9 9.5 17.2 8.1 27.2 13.1 33.1 24.5 14.7 19.7 30 8.8 1.8 26.2 26.9 4 31.9 10.3 20.4 23.4 10.2 39.7 34.9 28.2 9.2 29.7 28.7 5.9 31.2 34.3 26.1 11.1 19.8 25.3 28.2 28.3 17.2 22.4 32.4 2.7 18.2 2.3 23.5 14.2 ceb 28.7 20.2 21.7 16.2 12.9 13 31.7 23 23.4 25.8 - 26.6 32.4 31.6 28.4 25.4 41.6 23.2 25.6 23 0.4 22.8 35.6 25.2 27.8 17.9 13.7 25.6 25.6 25.3 24.7 23.9 32.5 14.4 21.3 25.9 22 12 22.3 11.9 8.2 12.2 19.6 7.9 9.4 16.8 4.5 24.8 2.5 30 16.1 13.5 16.3 30 8 9.5 23.7 24.4 4.8 20.7 9.1 18.8 20.6 9.1 35 29.6 24.7 7.3 27.2 26.5 5.9 27.8 30.1 25.9 8.6 20 28.5 25.3 25 15 20.9 31.6 3 17.8 2.5 20.2 12.7**

| cs | 30.6 22.7 26.3 16.9 15.8 3.1 36.9 27.8 30.8 28.2 23.1 - |  |  | 35.9 35.8 34.4 28.9 44.2 26.8 31.1 27.3 0.8 27.6 40.3 30.3 31.8 17.8 13.7 30.9 30.4 31.8 30.2 27.9 33.6 14.4 25.3 29.6 28.1 15.7 26.5 16 9.4 13.8 22.1 8.7 9.6 17.7 8.8 31.3 16.5 35.1 26.6 15.8 20.6 31.1 11.2 1.2 27.5 27.6 3.7 26.4 10.8 20.4 26.7 10.9 38.1 34.9 30.4 9.4 36.2 32.5 6.1 32.6 35.1 27.4 11.8 21.4 25.9 29.5 30.7 19.4 25.3 34 | 3 19 2.4 24.8 16.7 |
|---|---|---|---|---|---|
| cy | 37.1 6.4 29.3 19.9 16.6 8.8 39.4 22.4 30.7 31.6 24.8 33.3 - | 41.3 36.2 29.9 59.7 | 28 32.6 28.8 0.3 29.3 45.1 31.9 34.4 19.7 15.4 | 34 33.1 31.4 30.8 28.5 40.6 15.8 27.3 30.9 28.3 17.7 22 14.9 8.7 13.8 21.8 9.4 10.2 17.8 7.9 30.8 21.6 36.7 2.8 16.3 7.7 37.6 10.2 0.8 28.7 31.9 4.4 31.9 9.6 21.9 25.1 10.3 43.5 38.6 30.3 9.5 34.3 32.3 | 6 35.5 40.1 33.9 9.9 21.1 31.5 32.8 31.6 17.2 25.9 37.4 3.6 20.1 2.8 26.1 17.9 |
| da | 35.4 24.8 28.8 19.2 16.8 16.5 | 40 29.6 31.7 31.1 25.3 33.9 41.7 - | 38.1 30.3 52.7 28.5 33.1 28.8 0.8 30.9 44.4 34.2 34.4 20 14.8 33.9 33.1 32.2 31.3 29.7 37.8 15.3 28.6 31.5 29.5 17.2 26.4 16.5 9.7 | 14 23.2 10.1 10.3 18.8 9.3 | 31 17.5 36.2 29.5 16.9 23.4 34.5 11.8 4.1 30.8 32.9 4.3 29.7 12 22.4 26.1 12.1 42.4 37.8 30.7 10.3 34.8 32.5 6.7 35 43 29.7 12.7 23.1 28.3 33.4 32.4 20.6 26.8 36.1 3.5 20.8 2.7 25.6 19.1 |

**de 33.4 23.8 28.1 19.3 16.8 15.2 39 28.8 30.7 30.7 24.3 34 39.1 40.3 - 30.4 48 28.3 32.6 29 0.9 29.1 43.4 32.5 34.2 20 14.3 32.9 31.9 32.3 31.2 28.8 35.9 15.2 27.2 32.2 29.7 16.8 27.8 17.4 9.5 15.8 22.9 11.2 9.9 17.5 7.8 31.5 16.9 35.8 28.5 16.8 23.3 32.6 11.2 1.9 31 30.2 4 28.8 11.9 22.1 27.2 11.4 40.9 37 31.3 9.6 35.1 33.6 6.1 34.3 39.2 28.5 15.5 23.2 27.1 32 32 20.9 26.8 35.1 3.2 19.5 2.6 25.9 18 el 27.7 21.5 25.4 15.7 15.1 9.3 34.4 25.4 27.1 26.5 22.1 28.7 33.9 32.6 30.7 - 40 25.6 27.8 25.9 0.6 25.1 37.7 28.3 29.8 15.9 13.1 28.8 27.4 27.5 26.3 24.7 31.1 14 22.9 27.5 26.7 14.9 24.6 9.5 8.6 9.9 20.6 7.9 9 17.2 8 27.8 13.5 31.9 21.3 14.9 9.5 28.5 7.5 0.3 25.2 24.9 2.8 26 9.6 16.6 23 9.7 35.3 31.8 26.8 8.4 29 28.2 5.7 30 31.3 22 3.6 20.4 19.8 26.9 27.2 13.3 22.6 31.7 2.2 17.4 1.6 23.3 15.1 en 47.6 30.2 35.4 24.8 17.8 19.1 48.9 36.4 36.4 37.1 31.6 39.8 58.9 48.8 46 35 - 32.2 36.7 33 0.6 34.8 54.7 45 40.4 19.3 16.8 43.2 38.9 37.8 35 31.7 49.5 16.3 27.4 36.6 35.2 18.2 28.5 18.2 9.8 14.9 27.2 12.8 12 21.3 12.2 35.2 40.3 44.5 30.7 18.3 32.3 45.3 15 18.2 32.3 36.5 6.7 38.5 15.3 22.4 29.9 15.3 53.2 44.4 35.9 14.9 42.7 37.5 7.6 43.3 50.3 34.8 35.3 23.9 34.5 39.5 37.8 19.5 32.8 41.8 3.6 25.9 3.3 32.5 24.1 es 25.6 18.9 23.5 17.2 14.9 9.6 32.1 25.7 24.3 23.6 20.6 26.9 32.4 30.7 29.2 27.3 35.2 - 26.1 24.6 1.3 24.3 35.2 27.6 27.9 17.2 13 26.5 26.4 24.8 26.2 25.5 28.7 13.8 22.3 28.9 26.5 14.5 24.3 14.5 8.5 11.1 19.8 8.1 9 16 7.1 26.2 13.6 30.6 2.2 15 9.8 27 12 1.1 24.7 24.5 4.1 25.2 10.8 19.3 23.6 10.1 32.2 30.9 25.4 9 27.5 26.8 5.7 27.3 29.6 16.1 2.6 19.6 19 26.2 25.3 10.4 22.7 30.1 3.1 16.4 2.5 22.1 16.1 et 29.9 22.4 25.4 16.7 15.7 18.5 35.6 27.8 29.1 27.1 22 31.1 34.9 35.2 33 27.5 43.2 26.2 - 27.2 0.8 29.5 38.9 29.7 30.8 18.7 13.4 29.6 30.4 29.6 29.7 27.7 32.4 14 25.3 29.1 27.5 15.6 26.7 15.9 9.2 13.6 21.9 7.7 9.6 17.2 7.6 31.2 15.1 32.8 26.5 15.5 20.8 30.2 10.2 1.5 27.6 26.8 3.5 24.9 10 20.3 24.9 10.4 36.7 33.7 29.7 8.8 31.7 30.6 5.8 30.9 34.5 24.2 15.1 21.7 25.2 29.8 29.8 19.6 25.1 33.4 2.6 18.1 2.3 23.6 15.5 fa 27.1 21.8 26 14.6 15 11.1 33 26.3 26 26.2 21.9 28.4 33.8 32.5 29.4 27.6 41 24.6 27.7 - 0.8 24.1 36.7 27.5 28.2 17.1 12.4 28.4 29.5 26.6 26.4 24 31.3 13.7 22.6 26.9 25.9 14 23.6 9.7 8.7 14.1 20.6 6.3 8.9 16.8 8.5 27.4 13.1 31.3 11.8 14.8 7.8 28.2 10.7 0.2 25.2 25.1 2.5 23.3 10 20.2 21.8 9.1 35.2 31.4 26.2 8.5 29.4 27.9 5.3 28.7 31.1 22.6 4.9 20.6 10.3 27.6 26.7 15.4 22.6 32.5 2.2 17.6 2 23 15 ff 3.8 1.5 0.8 2 1.4 0.6 3.2 3.8 2 2.5 2.4 1.7 2.1 2.2 2.2 2.1 4.6 2.3 1.8 2.4 - 1.6 2.6 2.3 2.4 3.2 1.9 1.7 2.9 2 1.8 3 2.4 1.8 2.3 2.2 2.4 1.5 3.1 1.3 1.5 2.7 1.4 1.4 1.4 2.2 1.3 1.7 1.1 2.5 2.8 1.6 3.3 2.2 0.8 0.2 2 2.1 0.6 2 1.1 2.5 1.9 1.5 2.6 2.2 1.7 1.5 1.9 2.2 1.1 1.6 2 2.1 2.4 2.4 2.4 1.9 2.2 1.9 2 1.9 0.4 1.7 0.5 1.9 0.8**

| fi fr ga | 26.9 20.7 23.7 14.6 14.5 15.9 33.1 25.2 26.1 25.7 21.2 29.2 32.6 33.3 30.4 26.9 39.4 25.1 30.3 24.9 0.9 - 33 23.1 28.5 19.5 16.9 12.8 38.5 28.3 31 32.1 24.4 33.4 39.9 38.8 36.2 30.6 49.9 28.8 30.7 28.5 1.1 28.4 - 30.1 8.6 24.5 17 15 16.3 33.4 27.4 26.1 27.2 22.6 27.4 35.1 34.1 29.9 25.7 47.9 25.5 27.8 24.5 0.3 24.5 | 38 - | 36.7 28.3 28.5 16.3 11.7 27.8 27.3 27.4 27.4 25.2 | 30 12.7 24.2 27.9 27 13.6 24.3 14.3 8.5 12.1 21.3 7 8.4 16.4 32.7 35.6 19.2 14.8 33.3 31.7 31.6 30.5 28.9 36.4 15.3 26.1 33.3 29.6 17.4 26.6 17.4 9.7 14.5 22.5 9.9 10.4 18.1 8.3 30.6 14.8 35.7 28.3 16.2 18 33.1 12.5 0.5 29.3 29.4 4.4 34.9 29 16.8 13.6 27.8 28.9 26.9 26.8 25.5 32.8 13.9 22.8 27.1 25.7 14.6 20.3 13.9 7.8 13.2 18.8 8.3 9.3 16.8 5.6 26.7 22.8 31.2 | 8 28.7 | 14 30.9 20.8 14 12.6 27.4 9.7 0.9 26.3 25.6 3.2 22.8 9.8 18.1 23.8 8.1 34.2 30.9 27.3 7.5 29.9 28.9 5.5 | 2 14.6 15.8 31.4 7.9 3.3 26.3 26.7 3.9 27.1 7.4 20 22.3 9.2 35.6 32.4 25.7 8.2 29.7 28.4 5.6 29.3 32.6 27.8 16.7 19.3 26.6 28.5 26.6 16.3 23.2 32.2 2.9 16.7 2.4 22.8 14.2 | 28 32.9 19.6 3.8 20.5 19.7 27.1 27.2 13.5 22.5 31.4 2.9 12 22.3 26 11.1 42.6 37.9 31.3 10.2 33.4 31.6 6.4 33.2 37.8 22.1 3.4 22.9 27.5 31.6 31.6 18.7 26.3 35.4 3.5 20.2 2.8 25.8 | 17 2.2 23.8 14.4 19 |
|---|---|---|---|---|---|---|---|---|---|
| gl | 32.1 21.1 26.7 21.5 16.4 19.2 37.1 27.7 29.2 31.1 22.8 31.6 38.2 37 35 30 45 28.7 30.3 27.4 0.4 27.8 42.8 | 31 - | 19.3 14 31.6 30.7 30.3 | 30 27.8 35.3 14.9 25.2 32.2 27.9 15.9 25.8 15.8 9.2 12.2 22.1 9.5 9.4 17.8 8.6 | 30 12 35.5 19.4 15.8 20.1 32.1 11.8 | 4 28.3 28.6 4.1 30.5 11.2 21.4 25.6 10.8 39.7 36.3 30.4 9.4 32.9 | 31 | 6 33.6 36.1 25.1 6.5 21.8 25.8 30.6 30.3 15.3 | 25 34.8 3.2 19.1 2.5 25.2 16.4 |
| gu | 23 18.3 13.9 12.4 11.1 5.9 26.7 25.5 20.6 21 18.2 21.8 28.7 26.5 23 22.1 | 25 19.4 22 20 0.2 19.2 28.8 23.5 22.8 - | 10 13.5 29.7 20.8 | 20.9 17.6 20.9 11.6 16.8 21.5 16.5 8.5 19.9 5.9 3.7 8.4 18.7 6.3 7.5 14.5 7.4 19.7 | 7.8 25.4 24.2 9.6 18.7 23.3 5.6 6.1 | 20.4 20.1 | 2 13.9 9.3 19.4 17.1 8.3 26.8 24.6 | 20 7.9 22.7 22.3 3.9 23.5 25.4 16.6 | 17 17.1 13.3 14.7 19.5 11.4 15.4 19.6 2.1 14.6 1.1 17.7 8.3 |
| ha | 16.3 9.8 10.9 9.1 8.3 5.4 19.1 15 12.6 15.5 12.3 14.4 14 18.9 16.3 15.8 14.5 14.8 14.8 15.2 0.5 12.9 20.9 7.3 16.1 9.4 - |  |  | 14 17.3 14.8 14.1 12.5 18.9 10.7 12.8 14.6 8.3 6.1 12.7 4.1 2.8 6.3 11.6 4.4 7.3 13.7 7.2 |  | 15 13.1 17.5 8.9 6.6 7.1 17.1 4.2 1.7 14.5 14.9 2.9 11.2 4.5 10.4 12.5 4.7 19.8 17.7 14.6 4.7 15.9 15.9 4.3 16.4 17.8 14.2 5.5 10.9 12.8 14.4 14.2 4.8 9.5 17.2 2.1 11.2 1.6 12.7 3.9 |  |  |  |

**he 31.9 23.1 28.3 16.6 15.1 15.3 37.8 26.6 29.8 29 22.9 31.9 37.4 37 33.8 29.8 47.8 26.4 30.2 28.3 0.9 27.1 41.7 30.9 32 17.6 13.4 - 30 30.8 29.1 27.1 34.8 14.3 25.6 28.8 26.5 15.6 25.7 12.9 8.8 12.4 22.2 7.6 9.4 17.3 8.8 29.8 15.6 35.7 13.1 15.6 6 31.4 9 0.3 27.8 29.2 2.7 26.5 9.2 19.6 24.7 8.7 38.9 35.5 29.9 8.7 32.3 30.4 5.4 33.2 36.4 24.3 5.1 21.1 10 29.4 30.7 14.3 24.1 34.3 2.5 19.1 2.1 23.2 15.6 hi 28.8 23.5 24.7 15.4 14.7 8.9 32.7 29.8 25.8 25.8 22.1 27.5 34.3 32.3 29.6 25.2 45 23.6 26.4 26.4 0.6 22.6 36 28.6 27.8 22.1 12.9 26.7 - 26 25.5 23.9 30.8 14.5 21.7 25.6 26 14.3 23.6 11.1 8.7 16.6 20 7.6 9.6 16.4 8.1 26.2 9.8 30.5 28.4 14.8 25.2 29.5 10.5 19.5 25 25.3 3 21.9 13 24.7 21.5 10.6 34.4 30.8 25.8 10.9 28.4 26.9 5.2 28.8 30.9 22.7 22.1 20.3 20.8 27.3 26.3 24.1 22.6 31.9 2.4 18.7 2.1 22 16 hr 29.6 22.3 25.6 17.3 15.9 15 37 27.7 32.9 28 22.7 33 35.4 35.5 33.1 28.9 42.7 26.6 30.4 27 0.5 26.7 39.5 30.4 32 17.7 13.3 30.9 29.8 - 29.5 28.3 33.8 14.5 25.5 30 28.4 15.7 26.4 15.5 9.5 12.8 22.2 8.6 9.6 17.2 8.9 30.7 15.1 36.4 24.8 15.5 13.8 30.4 11.5 1.1 28.4 27.5 3.5 25.8 10.8 19.7 25.9 10.3 37.3 34.3 30.2 8.2 33 33.2 6.1 32.9 34.7 26.7 13.1 21.7 23.9 29.5 30.7 18.5 23.9 33.3 2.8 19 2.5 24.6 15.7 hu 28.5 21.7 25.3 16.3 15.7 16.6 34.9 26.2 27.6 26.8 22.2 30.6 33.6 34.2 32 27.4 42 26 29.6 26 0.7 26.5 38.1 29 30.3 18 12.9 28.7 28.5 28.8 - 25.8 31.9 14.1 24 28.9 27.7 14.5 26 15.6 9.2 14 21.4 7.6 9.2 16.6 8.3 28.6 14.5 32.1 24 15.7 11.7 29.1 10.6 0.7 27.4 26.5 3.4 24.3 10.2 19.9 24.1 9.9 36.4 33 27.8 8.1 30.3 29.6 5.8 29.7 32.8 21.8 4.3 21.2 22.6 28.9 27.8 15.1 24.4 32.7 3 17.7 2.4 24 15.8 hy 26 19.9 15.9 14.3 8.8 9.8 32.2 24 24.5 23.8 19.7 27 31.4 30.9 27.8 25.1 35.1 22.3 26.8 22.1 0.2 23 32.5 25.4 25.6 17.2 11.7 23.8 27.7 24.8 24.9 - 27.7 13.7 19.5 24.6 18.1 11.8 24.8 3 7.8 12.4 18.3 6.8 8.5 15.7 7 25.1 6.5 30.1 21.5 13.6 13.4 24.2 8.3 2.1 23.6 24 2 13.9 7 18.3 21.2 9 32.8 29.5 25.9 6.5 26.8 26.2 4.5 28.3 29.9 19.9 11.1 18.9 14.2 20.6 25.7 12.1 18.9 23.9 2.4 16.6 1.4 18.9 10.8**

| id | 31.7 23.6 | 27 17.9 15.4 15.6 35.7 28.6 28.5 28.1 25.6 30.7 38.5 36.1 32.6 27.6 48.8 25.5 29.2 27.5 1.1 25.8 39.8 31.2 31.7 | 19 14.8 30.8 31.1 | 30 28.8 26 - | 15.4 25.1 28.4 28.7 20.3 25.6 14.9 9.9 14.4 22.1 8.5 10.3 18.6 9.9 | 29 13.3 34.2 11.2 17.3 | 6 34.9 11.7 0.1 27.2 28 | 4 27.2 11.6 20.6 24.1 10.5 38.9 34.5 28.2 9.4 31.3 29.5 6.5 31.2 34.9 25.1 4.4 23.2 14.5 30.7 29.2 14.6 25.8 37.2 3.3 19.9 2.8 25.6 18.4 |
|---|---|---|---|---|---|---|---|---|
| ig | 14.8 8.5 | 10 7.9 7.2 4.7 17.2 14.7 12.7 13.7 12.5 13 15.2 16.9 14.3 14.8 17.9 14.1 13.7 11.4 0.5 11.9 19.2 11.8 14.7 10.6 | 9.2 11.9 15.9 12.8 | 12.3 13.6 16.7 | - | 12.6 13.9 12.1 4.4 12.6 5.2 3.6 7.1 9.9 4.3 6.3 12.3 5.1 13.4 11.9 16.4 11.4 7.2 10.5 15.3 4.1 2.1 12.4 13.5 3.2 12.8 4.2 10.9 11.6 4.8 | 18 15.8 13.1 4.5 13.6 13.6 3.8 14.7 15.7 14.6 6.9 11.2 13.4 11.7 | 13 6.5 8.9 16.2 1.8 10.5 1.5 12.2 4.2 |

**is 25.2 18.8 22.5 14.7 12.6 14.7 29.9 23.1 23.9 24.7 19.8 26.2 30.2 31.6 27.9 23.7 37.6 22.7 25.7 22.9 0.5 24.2 33.8 25.5 26.3 15.3 12.3 24.8 25.7 25.4 24.4 21.4 29.6 13 - 25 24 13.1 21.2 9.7 8.3 11.2 19.1 6.9 8.3 16.9 8.7 25.5 13.4 28.7 20.1 12.4 15.7 25.7 7.9 0.8 23.7 24.8 3.5 20.2 7.5 16.8 20.7 8.7 31.1 28.9 24.3 7 26.4 26.1 5.6 26.1 30.6 22.5 9.7 18.6 20.7 25.2 24.2 14.9 18.8 29.5 2.7 16.6 2.1 21.2 12 it 27.8 19.5 24.9 17.4 15.7 10.4 34.5 27 27.1 27.9 22.5 28.9 34 33 31.7 28.2 38.5 28.2 27.9 25.7 0.9 25.1 38.8 29.3 31.7 18 13.4 28.5 28.6 28 27.8 25.8 31.2 14.5 23.8 - 27.6 15.2 25.5 15.4 9 12.7 20.8 8.6 9.3 16.7 8.1 27.4 14.7 32.6 15 15.6 18 28.7 10.3 1 26.6 26 4.2 26.4 11.5 20.6 24.9 10.5 35.7 33.4 27.6 9.1 29.3 28.9 6 28.5 31.8 25.5 9.3 20.9 24.7 28.4 27.1 18.1 24 32.2 3 17.3 2.6 23.8 16.6**

| ja | 22 19 21.5 12.5 | 13 8.9 27.3 24.1 | 21 22.2 18.5 23.4 27.4 26.6 24.7 23.3 32.7 21.5 22.9 21.8 0.9 21.6 29.9 23.9 23.9 15.3 10.5 | 23 25.3 22.2 22.9 21.6 26.2 12.4 19.9 23.3 - | 12.1 22.4 11.2 8.1 13.2 22.8 5.7 7.2 14.1 | 7 23.8 8.4 25.7 22.4 13.7 16.8 24.4 10.9 2.1 22.7 21.4 2.5 19.5 9.2 17.7 19.5 8.6 28.2 26.2 22.4 6.6 23.5 23.3 4.3 22.9 25.3 21.6 11.7 19 19.9 23.2 22.2 16.5 20.4 28.5 2.2 14.2 1.7 22.1 12.3 |
|---|---|---|---|---|---|---|
| jv |  | 23.1 13.9 14.9 13 9.8 10.6 25.2 20.2 18.8 20.3 17.6 21.3 25.9 25.6 22.2 | 21 35 18.8 20.5 19.6 0.3 18.1 27.6 | 19 22.7 10.9 10.7 19.7 22.4 20.4 20.8 15.4 23.3 10.4 17.8 19.8 15.6 - | 17.9 6.9 | 6 7.4 16.8 5.9 7.1 14.8 8.6 20.1 5.4 24.3 11.1 9.5 8.3 24.1 4.7 0.6 19.6 20.7 2.6 18.8 5.5 12.2 17.2 6.2 28.1 24 19.5 6.1 22.2 21.4 4.6 22.1 24.2 18.7 5.2 16.1 14.5 20.3 19.6 8.8 14.1 27.5 1.2 13.5 1.5 16.7 4.1 |

| ka | 23.6 18.8 21.8 14.5 15.2 10.2 30.6 24.9 22.9 22.6 18.4 25.2 28.6 27.3 25.9 24.3 29.9 23.1 25.3 23.4 0.2 21.6 31.9 24.5 25.1 16.4 11.1 24.8 25.6 23.5 24.2 24.7 26.2 12.2 20 24.3 23.1 11.8 - |  |  | 12.6 6.1 8.1 18.9 7.5 7.7 14.8 7.4 25.9 14.3 28.2 24 13.3 19.1 24.2 7.3 4.8 22.8 21.9 2.5 20.1 8.3 17.8 21.1 8.8 29.5 26.9 |  | 24 7.5 25.2 23.8 4.5 26.6 26.2 21.5 12.8 18.2 19.7 24 24.5 16.3 22.3 28.1 1.8 14.9 1.6 19.7 11.3 |  |  |
|---|---|---|---|---|---|---|---|---|
| kk | 16.2 12.9 5.7 9.5 5.2 7.1 21.3 13.4 14.2 16.2 14.2 14.8 21.5 20.6 16.9 15.1 21.3 14.6 16.8 10.5 0.4 14.5 19.8 16.3 16.3 12.8 7.6 11 17.3 15.5 15.4 6.5 13 8.6 11.9 16.5 10.2 5.3 18.5 - | 5 7.6 8.8 | 5 5.8 | 12 4.8 16.3 0.9 20.7 12.5 8.1 8.1 12.6 | 5 2.6 15.2 15.2 1.9 11.5 5.6 12.4 | 13 6.3 20.2 18.3 17.5 5.7 16.3 16.8 3.3 17.8 18.5 9.5 8.7 12.4 7.9 8.4 16.3 9.2 11.2 7.1 1.4 | 11 1.1 | 9 3.6 |
| km 20.4 15.2 17.4 11 8.9 4.3 25.2 19.6 19.3 19.6 17.2 21.6 24.4 24.9 22.1 20.8 27.5 18.6 20.6 16.2 0.6 17.8 27.1 20.8 | 21 14.6 11.1 15 22.5 19.8 18.4 16.1 22.8 12.7 17.2 19.9 19.6 6.2 18.4 6.1 - | 9.4 17.1 6.2 | 8 15 | 7.9 19.9 3.9 23.8 17.6 11 12.3 23.9 7.5 | 3.3 19.2 19.1 | 3 16 6.3 15.2 17.2 7.1 23.9 24 20 5.3 21.8 21.1 4.8 22.1 23.7 18.4 10 18.2 12.6 16.7 19.7 9.2 13.4 26.6 2.5 14.3 | 1.6 18.4 | 8.5 |

| kn | 20.7 16.9 15.8 11 10.7 4.7 24.8 22.8 18.6 18.5 17.4 20.4 25.1 24.1 21.5 21.4 13.9 19.3 20.6 | 19 0.3 | 18 26.4 21 20.6 | 15 9.7 17.3 25 19.5 19.8 17.1 21.8 10.3 16.8 19.8 18.4 8.4 18.3 8.3 1.8 - | 17.7 5.1 6.9 14 | 8 20 5.8 23.4 16.7 8.7 18.8 21.5 5.5 6.8 19.4 18.5 2.2 15.6 7.9 15.5 16.6 7.7 25.4 22.5 18.3 6.5 | 21 20.7 3.9 21.5 22.7 17.3 21.1 15.9 14.6 18.9 18.2 15.3 13.7 21.1 1.7 14.3 | 1 17.6 6.8 |
|---|---|---|---|---|---|---|---|---|
| ko | 22.3 19.6 22.8 13.1 13.1 7.2 28.2 24.4 21.9 22.1 19.1 24.1 28.7 28.5 25.8 23.9 34.4 21.4 | 24 23.1 0.9 22.3 | 30.7 24.8 24.1 15.4 | 11.5 23.3 26.2 23 23.8 20.5 28.9 12.3 20.9 23.8 28.1 12.6 22.6 10.1 8.8 13.6 | - | 6.4 8.1 15.4 8.2 24.3 10.9 26.4 22.4 13.6 13.8 25.4 10.1 | 1.1 22.4 22 2.6 18.9 8.8 17.6 19.9 8.4 29.6 27.6 22.1 7.5 25.4 24.7 4.8 23.3 26.9 22.1 13.2 19.8 18.8 24.6 22.8 17.1 19.8 29.9 2.3 15.6 | 2 23 12.8 |

**lb 21.3 1 13.3 9.2 8.2 7 24.9 18.6 14.4 19.4 9 19.4 12.8 24.8 25.3 19.2 26.2 17.8 18.8 17.7 0.7 16.5 26.1 8.8 20.6 9.5 5.2 19.1 19.5 18.1 18.1 15.9 21.1 5 15.4 20.1 15.8 6.5 13.2 7.6 4.4 8.1 12.4 - 3.9 7.8 5.4 18.7 8.1 22.8 7.5 7.3 6.7 19.1 3.1 3.7 17.4 19.8 1.8 14.5 3.9 8.2 15.2 4.4 24.9 22 18.6 5.2 21.5 19.8 2.7 20.6 23.3 14.4 9.5 12.9 16.5 17.4 18.7 4 11.5 21.2 1.3 6.4 1.2 12.2 2.7 lg 13.5 6.9 8.2 7.4 6.3 5.6 15.5 12.9 10.7 11.8 11.6 12.2 12.6 15.1 13.1 13.2 11.2 11.9 11.3 12 0.8 10.2 16.4 8.8 13.4 7.6 8.4 11.6 14.5 11.5 11.7 11.4 15.1 9 10.7 11.9 10.1 6.1 11.4 6 2.3 6 10.3 4 - 11.8 6 11.7 11.3 14.8 9 5.9 7.9 13.8 3.8 4.7 11.5 12.1 3 11.4 4.2 8.5 10.1 4 16 13.9 11.8 3.9 12.7 12.4 3.7 12.9 14.2 12.9 8 9.5 13.7 10.6 11.6 7.1 6.7 14.9 2 10 1.4 10.7 4**

| ln lo lt | 21.9 15.4 19.4 11.6 10.7 9.1 25.5 20.5 19.4 18.4 | 16.5 13.8 15.4 9.7 9.4 7.2 20.3 16.8 15.7 15.6 15.3 16.8 18.1 20.2 17.2 17.6 17.5 16.2 16.5 16.7 0.6 14.5 22.7 14.5 18.1 11 11 16.4 18.2 16.7 16.3 15.7 20.7 11.9 14.4 16.7 18 21.6 24.3 25.2 22.4 19.8 25.6 18.5 20.1 20.6 1.1 18.2 26.3 20.2 20.7 13.9 11.8 26.6 21.4 24.2 15.2 15.5 16.3 33.5 24.6 26.5 25.9 21.1 29.5 31.6 31.9 29.8 27 38.6 25 29 25 0.8 25.6 36.6 28.2 28.8 |  |  | 17 12.4 26.8 27.5 28.1 27.3 25.8 30.3 13.5 22.8 27.9 26.3 13.8 25.6 | 14 9.5 15 7.5 3.2 20 22.6 20 20 18.3 26.7 12.9 18.3 19.1 18.9 10.7 18.4 8.5 9.2 10.1 17.6 6.4 8.6 15.4 - 15 8.3 12.3 21 7.3 8.8 16.8 8.1 - | 8 14.3 4.9 7.8 - | 8.1 16.6 15.9 | 19 14.4 8.6 12.4 19.4 5.7 6.3 15.9 15.7 3.5 15.7 5.7 12.2 14.3 5.6 21.6 18.3 16.7 5.1 17.3 17.5 4.4 17.5 18.3 16.9 8.5 13.4 17.4 15.8 16.2 11.6 10.5 21.3 2.5 20.9 7.6 23.5 18.6 10.6 15.2 24.5 6.9 7.4 19.3 19.5 3.6 17.5 6.8 14.8 17.1 7.1 26 24 20.1 5.2 21.7 21.3 5.3 21.9 23.7 20.9 16.3 19.8 19.6 19.9 19.2 12.9 14.1 28.2 3.4 14.5 2.4 17.4 10.6 16.5 31.1 21.7 14.1 12.9 27.1 10.1 1.1 25.8 24.3 3.2 22.7 9.6 19.4 24.5 8.8 34.5 31 28.3 |  | 8 29.8 29.2 5.4 28.8 30.5 20.3 4.8 19.9 22.5 26.7 28.1 13.6 22.5 30.8 2.9 17.7 2.2 23.2 14.6 | 13 1.7 15.2 7.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lv | 4.8 11.3 7.2 3.3 4.4 0.1 16.7 2.1 7.5 10.4 | 10 8.5 21.8 15.7 15.6 9.2 40.3 13 16.9 10.1 | 0 11 19.9 11.8 7.1 1.5 3.6 | 6 | 9 8.5 5.3 1.1 11.1 2.6 | 4 15.6 10.5 0.4 7.8 0.2 1.3 1.7 3.5 0.1 | 2 7.7 | 2 18.5 - | 12.8 2.1 3.9 0.5 5.1 0.4 0.1 12.4 9.9 1.1 0.2 0.7 2.8 6.9 1.3 18.5 5.5 19.3 2.9 10.3 9.6 3.8 7.6 12.5 23.1 0.3 | 7 | 1 12.7 15.4 1.2 4.9 11.8 0.3 6.8 | 0 12.9 0.8 |
|  | mk 31.5 23.7 27.8 18.9 16.6 15.8 40.3 28 33 29.5 23.9 32.6 38.1 36.8 34.5 30.7 47.7 27.4 30.7 |  |  |  | 29 0.7 27.4 41.2 31.3 33.1 18.7 13.9 33.1 31 32.5 29.2 29.8 35.5 15 25.9 30.2 28.6 15.9 27.8 16.3 9.4 13.1 22.9 8.6 9.6 17.5 8.6 31.2 14.8 - |  |  |  | 26.7 16.1 21.3 32.2 12.2 3.9 27.9 28.6 3.2 27.8 11.2 20.9 25.2 11.2 39.6 35.1 30.8 9.3 33.6 32.2 5.7 37.1 35.6 28.4 17.2 21.8 25.6 30.2 31.4 20.3 24.6 34.2 2.6 19.7 2.3 25.3 17.2 |  |  |  |

| ml | 25.4 21.7 20.8 14.1 13.8 9.6 | 30 27.6 22.6 23.3 20.6 24.3 31.1 28.6 25.9 23.2 37.8 21.1 24.2 22.5 0.3 20.5 32.8 26.2 25.4 20.2 | 12 23 29.2 23.2 22.7 20.6 | 27 13.3 20.1 23.5 24.1 11.9 22.7 10.6 8.1 15.4 19.1 | 7 8.4 | 15 6.9 | 23 7.5 28.4 - | 14 21.3 25.5 9.5 9.8 23 22.6 2.4 20.6 10.8 21.4 18.8 9.9 | 30 27.7 22.6 9.4 25.5 24.4 4.8 26.2 28.2 22.3 26.1 18.6 14.3 24 23.1 19.2 22.1 28.2 1.9 17.2 1.9 | 19 13.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mn 21.1 | 17.8 19.3 10.7 12.3 9.2 25.3 | 22.1 18.6 19.3 16.9 20.9 24.9 24.3 21.7 21.7 23.9 18.8 21.3 20.7 0.5 18.8 26.6 21.8 21.4 14.5 | 10.2 20.5 23.1 20.4 21.1 18.2 | 24.9 11.6 17.6 19.6 21.5 10.3 20.4 9.3 6.8 9.1 18.6 5.6 | 7.1 | 14.4 | 6.4 | 21 10 24.2 20.5 - | 14.6 22.4 8.1 4.8 19.4 18.8 2.5 16.1 7.2 15.7 17.7 7.2 26.4 23.8 19.8 5.9 21.5 20.8 4.2 22.4 22.7 18.5 12.6 17.2 16.5 21 20.1 10.2 17 25.9 2.3 13.9 1.5 18.5 | 10 |
|  |  | mr 24.7 21.2 17.6 13.9 12.3 8.5 28.4 27.4 21.7 22.2 19.9 22.7 30.7 28.4 25.4 21.9 39.3 19.1 23.2 18.7 0.3 17.9 26.3 25.7 23.8 21.7 11.8 18.4 30.8 22.1 20.3 19 22.1 13.7 18.7 22.8 22.6 12.6 22.2 8.7 7.8 15.4 18.5 6.9 8.6 15.5 7.8 20.7 8.4 27.1 |  |  |  |  | 27 13.7 - | 24.1 9.9 | 14 21.9 21.9 2.5 15.3 11.1 22.2 18.3 9.9 28.8 27.1 20.5 9.2 23.8 23.3 4.7 25.4 26.8 18.2 19.8 18.3 16.9 19.3 21.2 18.2 21.1 25.2 1.9 | 17 1.8 18.4 12.7 |
| ms 31.9 | 24 26 17.8 15.2 16.9 34.6 | 27.8 27.9 27.5 24.9 29.6 38.5 35.3 | 32 27.2 49 25.4 28.2 26.4 0.7 24.2 38.6 31.1 30.4 19.4 15.1 29.3 30.8 28.5 27.6 24.9 31.5 | 15.8 23.6 27.3 27.7 18.9 23.9 14.5 10.1 13.8 20.6 8.8 | 10.7 | 18.3 | 9 27.1 10.7 32.5 22.3 16.4 8.6 - | 9.5 0.1 26.3 27.3 4.1 | 27 11 21 23.4 10.4 38 33.6 27.5 9.5 30 28.5 6.7 30.2 33.5 27.8 11.4 22.2 | 22 29.1 27.9 16.6 25.6 35.7 2.8 20.7 2.9 23.9 17.4 |

| my ne nl no | 11 14.4 11.5 7.9 5.7 0.7 15.6 17.6 2.4 11.1 7.8 25.7 20.8 22.6 16.1 14.3 13.9 |  | 20 15.1 32 23.2 26.4 17.3 15.6 16.8 35.9 26.9 28.4 27.9 23.2 30.2 38.6 36.9 33.5 27.9 47.6 26.6 29.3 26.6 0.7 27.4 40.4 30.5 30.8 18.5 13.6 31.1 30.6 28.7 28.4 26.8 33.9 14.6 25.8 28.9 |  | 15 14.4 13.7 15 20.7 18.3 17.3 15.6 20.1 15.3 14.4 14.1 0.3 13.4 20.7 16.9 14.5 12.2 6.9 12.1 17.6 13.6 14.4 9.9 17.6 8.9 11.6 16.3 | 16 2.3 15.9 4 5.6 6.5 14.4 3.9 5.3 12.5 7 13.7 0.9 17.9 14.4 8.4 8.5 14.3 - 3 17.5 20.4 12.8 14.9 18.2 9.3 28.4 18.7 15.9 8.1 32 5.5 13.5 3.7 0.2 6.8 5.2 16 13.3 17.8 10.4 3.7 26.8 9.8 4.1 12.5 4.2 12.5 10.1 15.1 8.6 3.1 19.7 4.9 6 12.6 9.1 5.7 7.9 14 11.1 7.4 0.7 16.9 15.5 11 15.8 10.3 7 - 31 24.5 24.5 24.7 20.7 26.4 31.9 30.9 29.1 24.9 36.7 24.1 25.9 23.1 0.8 24.3 35.3 27.5 27.2 17.4 12.2 26.1 26.2 25.7 25.8 23.7 28.6 13.5 22.2 26.3 25.9 13.6 23.2 14.5 8.7 13.8 19.2 8.5 9.1 16.3 8.4 25.7 12.6 28.8 24.5 14.4 17.4 26.3 11.2 1 - 27 16.2 25.1 15.1 9.7 13.6 21 9.1 9.9 17.5 9.5 28.3 15.5 33.8 26.6 15.8 20.4 31.4 11.3 2.3 28.6 - | 24.8 4 26.4 | 1.4 15.9 14.1 1.2 10.2 3.3 11.8 13 4.7 20.1 17.7 4 23.8 11.3 18.7 22 9.9 | 15 4.6 16.4 15.1 2.8 16.3 18.7 8.8 10.1 13.6 4.9 13.9 8.6 9.4 9.6 18.7 0.6 11.5 0.6 13.9 2.6 13 13.9 2.4 8.9 9.4 19.9 9.2 7.9 8.1 14.8 9.8 6.9 12.9 13.9 4.2 14.4 15.7 10.1 9.2 11.7 7.1 2.3 10 13 11.5 5 1.9 14.7 1.2 9 6.4 32 29.1 24.9 8.8 27.3 26.2 5.6 25.8 29.7 23 10.1 19.5 22.2 25.8 25.3 16.9 21.8 29.1 3 17.1 2.5 22 15.3 11 21 23.7 11.3 38.1 33.8 28.2 9.7 31.5 29.4 6.1 30.8 36.9 27.6 12.9 21.2 25.5 29.5 29 18.9 24.6 33.6 3.1 19.4 2.5 24.6 17.5 |
|---|---|---|---|---|---|---|---|---|---|
| ns | 6.4 2.4 3.7 3.9 | 3 | 3 7.4 7.4 4.4 7.2 6.3 5.5 5.7 6.9 6.7 7.2 7.8 6.7 4.9 | 5 | 1 | 5 8.6 3.8 5.8 5.2 3.5 6.3 7.5 4.5 4.9 6.4 5.6 3.6 4.8 6.6 7.4 2.4 7.1 2.8 2.2 3.9 6.1 2.4 3.1 5.7 2.3 5.2 4.1 6.6 1.7 3.2 4.6 4.4 0.8 3.2 6.1 | 6.2 - | 5.9 1.9 4.2 5.7 | 2 8.1 6.4 5.9 2.5 5.6 5.5 2.2 5.5 6.2 4.8 5.6 5.1 5.2 6.3 6.1 3.6 3.6 6.1 1.2 4.6 1.2 4.3 2.1 |
| oc | 35.2 | 19.1 29.2 20.7 14.5 | 18.1 | 38.8 27.4 30.4 32.4 23.5 33.2 40.4 39.2 | 36 31 54.4 28 30.1 28.6 0.4 28.2 45.4 31.9 35.5 19.6 | 14 33.5 31.4 31 29.9 27.3 38.4 14.3 25.5 31.4 28.2 15.1 25.9 | 13 8.6 13.2 23.4 9.4 9.2 17.6 8.1 30.6 2.1 37.1 23.7 14.1 20.6 33.7 9.4 6.1 28.9 29.9 4.3 - | 9 20 25.2 9.8 44.3 36.4 30.3 8.6 | 33 31.8 5.9 35 39 28.3 12.8 21.8 27.5 31.4 31.1 18.7 21.1 36.1 2.7 19.7 1.9 25.1 11 |

| or pa | 24.4 18.4 |  | 13.3 11.6 3.8 6.5 4.9 2.2 16.6 20.6 12.8 13.5 12.3 9.9 19.9 15.3 13.8 11.4 20.8 11.7 12.9 7.7 0.5 9.7 14.7 13.6 12.7 15 6.2 4.9 21.5 10.5 10.2 7.8 9.3 8.1 8.1 13.9 7.1 0.8 14.7 3.8 4.2 17 12.7 12.2 6.2 28.5 26.6 21.3 21.9 19.1 23.2 29.2 28 25 23.3 31.9 20.9 23.1 22.4 0.3 20.5 30.5 24.4 24.2 19.7 11.4 14.7 31.9 21.6 21.9 21.3 21.8 12.7 18.5 22.4 18.1 8.6 | 21 | 9 7.1 4.4 4.7 10.3 4.2 8 6.5 11.9 18.4 6.7 8 14.7 10.1 22.4 6 26.9 24.3 11.1 21.6 24.4 | 10 0.6 15.6 11.4 6.9 12.2 10.6 1.6 4.2 11.9 10.9 1.2 9.8 - | 7 15.4 21.4 21.6 2.2 17.8 10.8 - 18.1 9.1 | 14.3 9.1 5.5 14 14.2 10.7 6.3 11.8 11.5 2.2 10.7 13.9 7.3 11.5 11.1 6.5 | 28 25.8 21.5 8.6 23.7 23.5 4.1 24.9 26.4 21.3 21.5 17.8 18.7 18.3 20.8 20.8 16.5 25.3 2.2 15.5 1.5 18.2 10.5 | 5 6.1 12.2 10 6.8 0.8 9.3 0.8 8.2 2.2 |  |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pl | 25.1 20 22.5 15.1 14.5 2.5 32.4 24.7 24.7 | 25 20.2 27.6 30.3 30.2 28.7 25.9 34.6 24.6 27 23.5 | 1 24.4 34.7 27.1 27.5 16.9 12.3 26.5 26.4 25.9 | 26 24.7 28.8 13.6 22.2 27 25.7 13.3 24.4 | 14 8.7 13.5 19.5 7.5 8.6 16.2 7.9 28.1 14.5 29.9 23.5 14.4 12.8 | 26 10.8 0.9 25.5 | 23 3.9 22.7 10 18.5 - | 9.8 31.4 30.1 26.5 7.7 28.9 27.6 5.6 | 26 29.1 23.6 10 20.2 20.7 25.1 25.7 16.6 | 22 29.8 3.2 16.6 2.5 22.6 | 15 |
| ps | 19 14.3 16.7 9.4 9.8 7.7 22.3 19.2 16.4 18.3 |  | 15 18.9 23.2 22 19.7 18.1 24.4 17.3 18.5 18.9 0.3 16.1 25.3 17.9 18.4 13.4 8.7 17.9 21.4 17.4 17.8 17.1 21.8 |  | 10 15.3 18.6 18.6 4.3 17.3 8.1 5.8 8.6 13.7 4.6 6.1 12.6 4.8 18.5 5.5 20.1 15.3 9.5 13.4 19.6 3.7 6.5 |  | 18 16.6 1.8 15.5 6.1 14.8 14.6 - | 23.5 20.5 | 17 7.2 19.3 19.1 3.3 19.4 20.6 16.3 13.9 13.9 15.9 19.2 17.8 15.7 15.6 23.7 1.1 13.1 1.1 16.3 5.5 |  |  |

| pt | 34.7 24.9 29.1 21.2 16.9 14.5 39.8 29.4 31.9 32.6 26.1 34.3 41.6 40.2 36.7 30.7 53.7 29.5 31.5 29.6 | 1 29.1 46.4 33.6 35.8 20.3 15.2 33.9 32.8 31.9 30.6 29.3 38.5 16 26.5 33 29.6 18.4 26.6 17.2 9.9 | 13 23.5 9.7 10.4 18.6 8.8 31.3 16.6 36.6 25 17.4 11.5 35.2 12.8 0.6 29.7 30.5 4.3 34.3 12.8 22.6 | 26 11.4 - | 39.4 31.7 10.3 34.4 32.3 6.9 34.7 38.7 24.5 3.5 22.9 23 33 32.7 15 27.4 36.2 3.5 20.9 2.9 | 26 20.5 |
|---|---|---|---|---|---|---|
| ro | 33.1 23.7 28.6 18.7 16.7 11 39.6 28.7 31.8 31.4 25 34.1 39.5 39 36 31 48 29.2 31.8 29.3 0.8 28.9 45 33.1 35 19.8 14.4 34.1 | 32 32.8 31.3 28.4 37.4 15.4 26.4 32.6 | 30 17.6 27.4 13.8 9.7 13.6 22.7 8.5 10.3 17.9 8.8 31.8 17.1 36.7 27.2 16.8 20.6 33.7 9.3 0.5 29.4 30.1 | 4 29.1 11.3 21 27 11.1 42.3 - | 31.5 9.6 34.7 32.7 6.5 33.9 37.8 29.2 11.6 23.1 25.2 32.4 32.3 19.2 25.7 36.2 3.2 20 2.5 25.8 18.2 |  |

**ru 27.9 21.8 24.9 15.9 15.9 5.6 35.5 26.4 27.6 27 21.2 29.7 33 32.8 31.1 27.8 39.5 25.7 29.7 26.1 0.8 25.5 38 28.9 30 17.2 12.3 29.5 28.3 28.6 27.4 28.4 30.8 13.8 23.5 28.4 26.9 14.5 28 16.3 8.8 12.5 20.6 7.6 9 16.8 7.9 30.4 20.8 32.8 25.2 15 20.3 28.4 11.9 6.3 26.1 25 3.1 25.2 10.7 19.2 25 10.2 36.1 32.3 - 8.3 30.2 30 5.1 30.7 31.6 25.7 10.6 20.2 23.8 25.1 33.4 18.8 24.7 32.3 2.6 17.4 2.1 23.8 15.5 sd 16.4 12 6.8 7.8 6.4 3.4 18.3 16.9 13.6 15.6 13.1 13.5 21 18.5 16.3 14.4 23.5 13.5 13.5 14.5 0.4 11.8 20.6 15.7 15.1 14.3 7.1 9.8 22.4 13.3 11.1 8.4 13.6 8.9 10.6 15.3 8.6 3.3 14.4 4.1 4.9 8.6 9.4 4.3 5.4 11.2 5.6 12.9 2.3 16.4 12.6 8.2 9.9 14.4 3.7 2.8 13.9 14.2 1.6 12.2 6.4 16 11 6.9 19.7 16.9 12.1 - 15 15.5 2.7 16.1 17.1 9.2 10.5 11.6 7.6 10.4 9 15.2 11.6 10.1 0.8 10.7 0.8 9.8 3.4 sk 30.3 22.9 26.1 17.3 16.1 3.1 37 27.6 30.8 28.2 22.8 36.2 35.6 36.9 34.8 28.7 44.9 27.4 31.4 27.9 0.8 27.7 40.9 30.4 32 18.2 14 30.8 30.4 31.9 30.4 27.8 34.1 14.7 25.3 30.1 27.5 15.9 26.7 16.1 9.3 13.1 22 9 10 17.4 8.3 30.8 14.6 33.9 26.6 15.9 19.3 30.7 11.5 2 28.7 28.1 3.8 26.3 10.8 20.4 26.5 10.7 38.3 34.9 30.6 9.3 - 32.5 6.2 32.8 34.9 26.8 14.1 21.1 25.1 30.8 30.5 19.2 25.1 33.8 2.9 18.8 2.4 24.7 16.6 sl 28.7 21 25.3 16.5 16.2 3.8 35.3 26.5 29.9 26.9 22.1 31.3 33.3 34.2 32.1 27.6 39.9 26.3 29.6 26.2 0.6 26.9 38 29 30.8 16.5 12.6 28.7 29.3 31.7 28.6 26.5 31.9 13.8 24.2 29.6 27.3 14.7 25.3 14.8 8.7 11.4 20.8 8.1 8.8 17.2 8.9 29.6 11.7 33.6 25.6 15.1 14 28.9 9.6 1.1 27.5 26.7 3.2 24.1 10 19.5 24.9 10 35.9 32.7 29.2 8.8 32.1 - 5.9 31.3 33.2 26.1 14.4 20.7 24.7 28.6 28.9 18.3 24.6 32.1 2.8 17.9 2.3 23.2 14.8 so 12.7 5 4 7 5.4 3.5 13.9 9 10 12 11.2 9.6 12.5 13.6 11.2 11.6 15.1 11.2 10.9 9 0.8 9.7 15.7 8.9 12.2 8.1 6.8 7.9 10.9 10.6 10.8 10 12 7.5 8.6 11.9 5.8 3.6 10.6 4.4 3.8 5.6 6.2 3.9 5.2 10.7 3.8 10.8 9 12.5 3.7 6 5.2 10.8 2.7 0.7 10.6 10.4 2.6 8.6 3.3 8.1 9.1 4.1 14.9 12.7 9.6 3.5 11.2 11 - 11 12.1 8.6 5.1 8.6 8.9 6.6 8.6 6.2 8.1 13.8 2 9 1.4 6.3 3.4**

| sr sv ta th tl tr uk | 23.6 18.7 28.4 24 26.1 16.2 18.5 30 22.4 26.3 16.9 16.3 10.6 | 33.4 23.8 28.2 17.6 16.9 5.3 41.1 28.8 35.4 29.9 24.4 34.5 38.7 38.9 36.2 31.2 48.1 27.9 sw 28.3 22 23 14.8 12.9 13.9 31.7 24.5 25.1 25 21.5 26.6 34.2 31.4 28.4 15.6 17 3.3 9.8 6.6 2.6 15.6 17.5 11.4 11.3 16.8 9.6 24.8 13.9 30.9 20.6 19.9 18 12.4 13.6 34.6 24.9 26.8 27.9 27.8 28.7 38.2 35.4 30.9 26.2 50.4 24.9 | 32 29.3 0.7 27.9 43 32.1 34 18.8 14 33.3 31.8 33.4 30.7 30.2 35.9 15.3 26 30.8 29.2 16.9 28.1 16.9 9.3 13.6 22.9 35.4 23.9 27.7 18.8 16.4 17.2 39.5 28.9 31.6 30.6 24.5 33.5 40.2 43.1 38.1 30 51.6 28 32.2 28.7 0.9 31.6 43.8 32.7 33.6 20.2 14.5 33.4 32 32.6 31.2 28.5 37 15.4 28.4 31.2 29 18.1 25.7 16.5 9.6 14.1 23.1 9.8 10.5 18.3 9.6 30.9 18.5 35.7 28.5 16.6 20.4 34.1 11.6 1.4 30.2 33.8 3.9 29.2 11.6 24 37.7 22 24.8 23.6 0.6 21.6 34.5 26.8 26.5 17.9 14.5 25.1 27.9 25 23.2 22.9 31.8 15.1 21.7 24.2 23.6 14.4 21.4 9.7 7.7 10.4 18.3 7 10.3 17.8 5.8 24.6 25.4 29.6 19.6 13.4 15.7 29.3 8.4 11 8.6 37 7.2 11.1 8.5 0.4 6.8 9.5 18.8 11.5 17.7 8.4 5.9 22.1 9.5 7.1 7.3 9.2 11.1 8.6 10.7 5.3 5.3 16.8 3.4 5.6 13.2 7.2 4.6 6.6 11.8 5.8 6.1 3.4 13.1 12.4 9.6 9.3 9.5 6.6 1.9 10.2 10.2 2.3 4.4 8.1 18.4 8.5 6.7 10.6 12.7 8.7 6.5 11.8 11.6 3.5 12.3 13.5 9.1 - 22 13.4 13.2 11.1 28.9 24.9 22.8 22.9 19.7 25.1 29.1 28.6 26.1 23.6 35.4 21.8 23.8 23.3 0.7 22.5 32.3 24.4 25.5 15.2 11.8 23.7 26.7 24.1 23.7 20.8 30.1 12.6 21.1 23.9 25.1 14.4 21.4 10.2 8.9 12.1 19.7 6.7 8.4 15.7 8.3 24.6 12 27.3 22 13.7 11.7 26.8 8.6 0.5 22.8 21.8 2.6 20.9 7.6 28 22.4 0.4 24.3 38.6 29.9 30.3 19.2 14.3 25.1 27.3 27.4 26.3 21.3 34.7 15.5 22.7 27.6 21.7 15.1 23.1 9.8 8.5 11.6 2 33.8 27.4 26.6 27.3 23.2 29.3 35.1 35.2 31.9 27.2 44.2 25.9 28.9 27.7 1.2 25.6 38.2 29.7 29.3 18 13.4 29 30 28.2 28.3 25.2 33.4 14.4 24.4 29 27.5 15.1 25.2 14.7 8.7 14.6 21.5 7.3 9.3 17.1 8 28.2 11.1 32 16.3 15.9 7.7 38 27.4 29.8 28 22.5 32.2 35.5 35.2 33.5 28.7 43.2 26.6 30.4 27.6 0.6 26.5 40.2 30.2 32 18.1 13.3 31.1 30.4 30.4 29 29.4 33.5 14 24.5 29.5 28.3 15.9 28.4 16.7 9 14.3 21.4 7.9 9.2 17.2 7.9 31.7 17.1 35 26.7 16.2 21.7 30.2 11 | 8 9.8 17.4 9.3 32.7 12.7 39.9 28.8 16.7 19.2 33.1 11.7 1.5 29.2 29.2 2.9 28.8 11.7 21.8 26.9 11.6 40.8 36.8 31.9 9.7 35.4 34.3 6.1 - 2 23.7 24.3 3.6 20.3 8.8 18.8 20.4 9.1 33.3 30.7 25.1 7.9 27.4 18 20.5 9.1 30.4 28.1 23.6 19 8.4 10.1 17 5.6 26.5 4.5 32.2 14.1 14.5 15 32.4 7.5 3.3 26.1 26.6 4.4 21.6 9.2 20.7 22.2 9.5 37.3 33.3 25.6 7.9 29.4 28.2 5.9 30.3 33.2 28.4 6.4 20.8 - 30 10.1 0.1 27.2 26.6 3.5 24.2 11 21 23.4 9.4 36.9 33.1 24.6 8.4 30.6 29.1 6 27.2 26.8 3.1 27.3 10.5 21.3 25.8 10.5 38 34.4 34.1 | 22 25.8 11.7 41.4 37.2 31 10.1 34.1 32.5 6.6 34.6 - | 26 5.6 27.6 30.2 - 7 25.9 25.3 5.1 25.7 27.4 21.4 10 - 6 29 32.8 21.5 5.5 21.6 11.1 - 9 32.5 31.1 5.8 33.5 33.8 27.4 15.6 | 36.9 30 16.9 22.3 27.1 31.5 33.4 20.8 27.6 35.8 2.4 19.5 2.3 24.4 18.2 29.3 13.8 22.4 26.9 32.1 31.9 20.1 26.2 36.1 3.2 20.4 2.7 25.8 18.9 14.7 24.3 24.3 16.3 21 25.8 28.6 - | 10.2 19 18.1 24.9 24.9 16.5 20.8 30.9 2.5 20.1 2.1 19.9 11.8 13.1 3.5 8.3 7.1 12.2 13.1 12.5 1.5 14.8 1.3 6.5 8.3 20 31.4 2.1 16 1.9 21.8 13 25.4 27.4 15.5 22.1 32.3 2.7 19.8 2.4 20.4 14.2 26.3 14.5 25.2 33.3 2.5 18.8 2.4 23.9 16.5 19.8 25.4 33.6 2.6 18.4 2.3 24.4 16.2 |
|---|---|---|---|---|---|---|---|---|
| ur | 22.7 19.6 19.4 12.8 12.2 | 5 27.1 25.6 21.1 | 22 19.1 23.1 29.9 27.8 24.9 22.9 38.5 19.9 22.9 21.5 0.2 20.2 30.8 25.6 22.4 18.3 10.5 19.5 28.6 21.5 21.1 16.2 24.6 12.4 17.8 22.6 19.9 6.2 21.5 5.6 7.7 12.7 17.7 5.5 7.7 15.1 7.5 21.2 6.2 24.9 22.8 12.3 | 14 23.3 8.4 5.6 21.8 20.6 2.1 14 9.1 21.3 17.4 8.5 28.8 25.9 20.8 | 10 24.2 22.8 | 4 23.3 25.9 15.8 15.4 17.7 | 9 21.5 20.3 - | 14.8 27 1.3 16.1 1.2 18.6 11 |
| uz | 24.5 16.1 21.7 12.4 16.1 15.6 31.4 24.6 22.4 23.2 17.4 25.6 27.1 28.9 26.4 23.4 33.3 22.7 25.3 24.2 0.6 22.2 33.6 20.6 25.2 16 9.9 25.1 26.6 23.7 24.2 24.8 28.3 11.3 19.8 23.8 24.3 12.8 23.3 15.5 6.7 11.3 18.1 | 6 | 7 14 5.4 25.4 22.2 28.7 17.4 14.5 18.5 25.6 5.3 | 7 23.2 22.6 1.9 18.9 6.6 17.8 20.1 8.5 31.1 28.1 23.9 6.8 26.2 25.3 | 4 27.2 | 27 21.1 22.4 17.7 22.8 26.2 | 25 17.6 - | 29.5 1.3 14.2 1.1 20.4 6.9 |

| vi wo xh | 9.3 2.3 4.1 4.4 4.1 24.6 16.4 19.6 12.7 |  | 27.1 20.8 24.7 15.5 14.7 13.7 32.7 26.5 25.8 25.5 22.9 27.7 33.1 32 29 25.7 41.2 23.8 26.4 25.2 1.2 4 10.7 8.8 6.5 9.3 7.4 7.1 4.6 8.9 7.9 9.7 9.2 | 9 6.8 7.7 0.6 6.6 11.8 3.7 8.7 | 6 12 13.5 27.1 22.6 20.4 21.9 20.4 22.6 27.3 27.8 23.8 21.8 34.5 19.7 22.2 20.9 0.9 19.3 30.3 21.3 23.3 15.2 13.8 21.7 24.8 21.6 21.2 20.9 27.8 14.1 18.7 21.3 21.5 12.2 19.3 10.4 6.7 11.6 17.5 6.4 10.1 17.2 | 4 7.7 9.6 7 6.6 |  | 24 35.9 28 28.2 17.2 13.7 27.3 28.2 26.5 26.3 23.6 34.4 15.2 23 26.6 27.2 16.7 23.3 11.6 9.2 13.8 20.9 7.4 9.6 17.7 8.9 26.6 11.4 30.8 16.6 15.4 7.1 30.4 11.5 0.6 24.9 8 8.7 4.4 6.5 8.6 8.2 2.8 7.6 3.7 2.7 4.7 6.9 2.6 3.3 6.8 2.4 7.7 5.6 9.8 |  | 3 3.9 4.8 7.2 1.3 1.4 7.8 8 1.5 9 21.9 22.1 26 11.8 11.2 18 26.6 5.4 7.8 20.2 20.4 3.6 21.4 | 25 4.1 23.9 10.7 19.8 22.5 9.3 34.1 30.8 26.5 8.2 28.2 26.8 6.2 27.4 30.5 21.6 6 2.1 5.3 7.1 2.6 10.6 8.6 7.8 2.5 7.9 | 8 2.4 8.3 7.7 5.8 6.3 7 16.8 18.3 7.4 28.9 25.9 21.3 7.6 22.8 22.5 5.3 24.2 25.1 25.2 20.3 15.6 23.7 22.2 21.7 16 16.8 27.4 2.5 - | 4 21.4 11.2 27.3 26.9 13.1 23.3 - 6 8.2 8.1 7.9 4.1 4.2 9.7 - | 3.3 18.2 2.8 24.3 16.8 4.5 1.1 4.9 1.6 1.8 18.5 7.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yo | 5.2 1.4 1.8 3.1 | 3 1.2 7.3 6.1 4.9 | 7 4.9 4.8 4.5 6.7 | 6 6.3 7.3 6.3 4.8 4.3 0.3 4.4 7.9 3.1 | 6.1 | 5 2.6 3.8 6.8 4.4 4.1 5.9 | 6 2.8 4.1 6.5 3.7 | 1 6.4 2.4 1.6 3.9 | 4 1.9 2.2 4.4 2.1 4.4 2.5 6.4 2.6 2.7 | 4 4.1 1.5 0.3 5.3 6.1 | 1.5 4.1 1.6 3.8 4.4 1.8 7.6 5.9 4.5 2.3 5.3 5.1 | 1.6 5.6 5.7 3.6 3.8 | 5 4.6 3.3 3.9 3.1 3.1 5.4 0.5 3.4 - | 4.3 1.2 |
| zh | 23 18.2 23.1 13.4 | 14 8.7 | 29 24.7 23.2 23.1 19.6 25.4 28.7 28 26.2 | 24.1 34.6 22.7 24.5 23.7 0.9 23.1 32.6 24.9 25.4 | 15.9 | 11.2 24.2 25.8 | 24 24.6 22.1 28.9 12.8 20.7 25.1 27 13.3 22.6 11.8 | 9 13.1 21.5 6.5 | 8 15.6 8.6 25.9 11.6 27.5 22.2 14.1 | 17 26.3 11.5 1.9 24.1 22.3 | 2.8 19.9 9.5 18.4 21.1 8.9 30.4 27.5 23.9 7.1 25.4 | 24.4 | 5 23.9 27.9 22.6 8.7 20.3 20.1 24.2 23.7 17.2 20.2 31.5 2.5 15.3 2.1 - | 13.5 |
| zu |  |  | 23.1 14.8 18.3 11.7 11.3 12.7 25.6 21.6 19.4 21.3 18.8 21.7 25.7 26.1 23.1 20.2 30.8 18.5 20.3 19.7 0.9 17.3 28.8 19.4 21.7 14.5 |  |  | 13 21.1 23.7 20.1 19.8 20.6 26.7 13.8 |  |  | 18 20.3 20.3 10.1 17.7 9.5 5.8 9.7 15.5 5.9 9.1 15.8 6.1 20.5 21.2 24.4 13.3 11.3 16.9 25.6 |  |  | 2 5.3 18.8 20.1 2.8 19 5.1 15.6 16.6 6.8 28.3 25.2 20.6 6.7 21.7 21.7 4.9 22.9 24.4 25.2 19.5 14.5 22.7 21.3 20.9 15.5 17.1 27.2 1.7 19.5 1.3 17.5 - |  |  |

Figure 10: Evaluation results of 3.2B DEEPNET on a subset of FLORES-101 devtest set. The i-th row is the source language, while j-th column is the target language. There are 87 languages and 7,482 directions.
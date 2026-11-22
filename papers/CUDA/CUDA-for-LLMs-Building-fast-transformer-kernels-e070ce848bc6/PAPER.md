(UD

LL

#### Bu i ld ing fast transformer kerne ls

### E l l iot Ar ledge

###### MA N N I NG

-----

# CUDA for LLMs

#### Building fast transformer kernels

##### Elliot Arledge

MANNING Shelter Island

-----

For online information and ordering of this and other Manning books, please visit www.manning.com. The publisher offers discounts on this book when ordered in quantity. For more information, please contact

Special Sales Department Manning Publications Co. 20 Baldwin Road PO Box 761 Shelter Island, NY 11964 Email: orders@manning.com

© 2027 Manning Publications Co. All rights reserved.

No part of this publication may be reproduced, stored in a retrieval system, or transmitted, in any form or by means electronic, mechanical, photocopying, or otherwise, without prior written permission of the publisher.

Many of the designations used by manufacturers and sellers to distinguish their products are claimed as trademarks. Where those designations appear in the book, and Manning Publications was aware of a trademark claim, the designations have been printed in initial caps or all caps.

∞ Recognizing the importance of preserving what has been written, it is Manning's policy to have the books

we publish printed on acid-free paper, and we exert our best efforts to that end. Recognizing also our responsibility to conserve the resources of our planet, Manning books are printed on paper that is at least 15% recycled and processed without the use of elemental chlorine. The author and publisher have made every effort to ensure that the information in this book was correct at press time. The author and publisher do not assume and hereby disclaim any liability to any party for any loss, damage, or disruption caused by errors or omissions, whether such errors or omissions result from negligence, accident, or any other cause, or from any usage of the information herein.

Manning Publications Co. 20 Baldwin Road PO Box 761 Shelter Island, NY 11964

ISBN 9781633434899 Printed in the United States of America

Development editor: Connor O'Brien Technical editor: Ruud Gijsen Review editor: Kishor Rit Production editor: Kathy Rossland Copy editor: Kari Lucke Proofreader: Katie Tennant Technical proofreader: Mohamed Maher

Typesetter: Tamara ŠveliÊ SabljiÊ Cover designer: Marija Tudor

-----

*For every curious CUDA learner who, like me, wished this resource existed at the beginning*

-----

```text

```

-----

# contents

---

*preface xii acknowledgments xiv about this book xvi about the author xix about the cover illustration xx*

*1* ***When PyTorch just isn't enough 1***

1.1 What is CUDA? 2 1.2 When do you need custom CUDA? 2 1.3 CUDA basics 4

*Host and device: Two worlds working together 4* *Kernels: Functions that run on thousands ofthreads 5* 1.4 Recognizing parallel opportunities 7

*The memory hierarchy: Your performance bottleneck 8* *When parallelism doesn't help 10* 1.5 Deep learning through the CUDA lens 11

*The nature ofdeep learning computation 11 Why GPUs excel*

*at deep learning 12* 1.6 The optimization stack 13

*Starting point: Naive kernels 13 Optimization layers 13*

*The compounding effect 14 Scaling to multiple GPUs 14*

*Why this matters 14* 1.7 Getting ready to build 14

**v**

-----

**vi contents**

*2* ***Your first CUDA program 17***

2.1 The "why": Throughput vs. latency 18 2.2 Your first run 18 2.3 Anatomy of a CUDA program: The vecadd.cu file 20 2.4 Where the parallelism comes in 23 2.5 Scaling up: Grids, blocks, and boundary checks 25 2.6 Expanding to 3D arrays 29

*3D tensor addition: CPU implementation 29 The CUDA kernel*

*for3D tensor addition 29 3D kernel launch configuration 30*

*Real-world example: Batch image processing 31* 2.7 The universal pattern: From nD to 1D 32

*3* ***Naive kernels 34***

3.1 From 1D to 3D element-wise addition 35

*From loops to parallel threads 37 Kernel 1, matrix transpose:*

*When data needs to change shape 38* 3.2 Core neural network kernels 41

*Kernel 2, GEMM: The heart of neural networks 41* *Kernel 3, softmax: The final activation 43* 3.3 Convolution and pooling toolkit 45

*Kernel 4, 1D convolution: Processing signals and sequences 45 Kernel 5, 2D convolution: The engine ofcomputer vision 46 Kernel 6, 2D max pooling: Downsampling and feature detection 49*

*4* ***Training and verifying neural nets in raw CUDA 53***

4.1 Level 1: The PyTorch approach 53

*PyTorch implementation: Setup, training, and benchmarking 54* 4.2 Level 2: Backpropagation theory 57

*Building intuition: From slopes to gradients 59 Scaling to vectors*

*and matrices 61 Layer-by-layer gradient derivations 62*

*Common pitfalls and debugging tips 63 Vectorized*

*implementation for batches 63* 4.3 Level 3: The NumPy approach 64

*Manual implementation in NumPy 64 Verification technique:*

*Isolate and conquer 67 Performance 67*

-----

**contents vii**

4.4 Level 4: The C approach 68

*Setup: Headers, constants, and data structures 68 Memory*

*management and initialization 69 The forward pass in C 70*

*The backward pass in C 71 Training loop and performance 72*

4.5 Level 5: The naive CUDA approach 74

*CUDA setup and memory management 74 From C loops to CUDA*

*grids: The matmul kernel 76 Forward and backward pass kernels 77*

*The training loop and performance 79 A note on profiling:*

*Wall-clock vs. GPU time 82* 4.6 Level 6: The cuBLAS approach 83

*Integrating cuBLAS and optimizing the training loop 84 Dissecting*

*the cublasSgemm API call 86 Debugging gradient instability 87*

*Performance and final verification 88 A necessary lie and*

*the path forward 88*

*5* ***Transformer inference in CUDA 91***

5.1 Getting started: Transformer architecture and setup 92

*Establishing the baseline scenario 92* 5.2 Inference: Theory and optimization 105

*Optimizing autoregressive generation 105 Deploying the*

*optimized stack 114* 5.3 Inference kernels 117

*Memory layout and contiguity 117 GEMV: The inference*

*workhorse 118*

*6* ***Optimizing our transformer kernels 126***

6.1 Preparing to optimize 127

*Profiling CUDA kernels 127* 6.2 GEMV 127

*Understanding the memory access problem 129 Achieving coalesced*

*access 131 Profiling the coalescing improvement 134 Scaling*

*up with block-level reduction 135 Maximizing bandwidth with*

*vectorization 138 Final GEMV performance analysis 139*

*GEMV optimization summary 140* 6.3 Softmax 141

*Online softmax: A better sequential algorithm 143 Parallel softmax*

*with warp shuffles 145 Fused softmax with shared memory 147*

*Vectorized softmax 149 Profiling softmax optimizations 150*

*Softmax optimization summary 151*

-----

**viii contents**

6.4 Layer normalization 151

*Fused LayerNorm with warp operations 155 Fused LayerNorm*

*kernel 157 LayerNorm optimization summary 159*

*LayerNorm profiling results 159* 6.5 Top-k 160

*A better sequential algorithm: min-heap 162 Parallel top-k with*

*block reduction 164 Top-k optimization summary 165*

*Top-k profiling analysis 165* 6.6 General matrix-matrix multiplication 166

*Understanding the GEMM memory access problem 168 Improving coalescing through thread remapping 169 GEMM optimization with tiling and shared memory 169 1D blocktiling: Computing multiple outputs per thread 174 2D blocktiling: Maximum register reuse 175 Vectorizing*

*memory access for tiling 177 GEMM profiling: Understanding*

*the bottleneck 179 GEMM optimization summary 180*

*7* ***Tensor cores 183***

7.1 Tensor core fundamentals 184

*What are tensor cores? 184 When to use tensor cores 185*

*A brief history oftensor core instructions 185 Tensor core*

*performance across GPU generations 186* 7.2 WMMA: The high-level tensor core API 187

*WMMA fragments and tiling strategy 188 Loading data*

*with WMMA 190 Matrix multiply with WMMA 191*

*Storing results 193* 7.3 WGMMA: Asynchronous warp group operations 193

*Basic WGMMA: The first asynchronous tensor cores 194 Understanding PTX inline assembly 195 WGMMA with*

*larger tiles 198 WGMMA with asynchronous memory*

*operations 199 WGMMA with maximum tiles 201*

7.4 Performance analysis: CUDA cores vs. tensor cores 202 7.5 Performance progression analysis 203

*Decision framework: When to use tensor cores 204* 7.6 From CUDA cores to tensor cores: Pattern mapping 205

*Manual tiling becomes hardware tiling 205 Vectorized loads*

*become TMA 206 Shared memory becomes implicit 207*

*What you still optimize manually 208*

-----

7.7

7.8

*8*

8.1

8.2

8.3

8.4

*9*

9.1

9.2

9.3

9.4

**contents ix**

TCGen05: Blackwell's tensor cores 208

*TCGen05 overview 209 Challenges and considerations 209*

Looking ahead 210

***Flash Attention 212***

From compute-bound to memory-bound 213

*The HBM bottleneck in standard attention 213* Designing Flash Attention 214

*Building intuition: Why this problem is harder than it looks 214 The Flash Attention solution: Tiling and online softmax 215 Tiling: Breaking the problem down 215 Online softmax:*

*Computing global statistics iteratively 215* Implementing Flash Attention 217

*Dissecting the naive implementation 218 Understanding the work*

*distribution strategy 220 The three-level loop hierarchy 221*

*The fused Flash Attention kernel 222 Benchmarking our kernel 228*

The evolution of Flash Attention 229

*Flash Attention 1 and 2 229 Flash Attention 3: Asynchrony*

*and low precision 230 Flash Attention 4: Complexity for*

*Blackwell 231 What this evolution teaches us 231*

***Quantization 234***

Quantization building blocks 235

*Data types: The foundation ofquantization 235* *Basic quant/dequant operations 239* Deployment strategies 243

*Granularity schemes: How to apply quantization 243 Calibration: Computing quantization parameters 249 Static vs. dynamic quantization 251 Symmetric vs.*

*asymmetric quantization 254* Advanced techniques 257

*Activation-aware weight quantization 257* *Simple example: Big vs. small inputs 257* Putting it all together 259

*Real model quantization workflow 259 Accuracy validation*

*strategy 260 Common pitfalls 261 The tradeoffs in one place 262*

-----

**x**

9.5

9.6

*10*

10.1

10.2

10.3

10.4

*11* ***CUTLASS***

11.1

11.2

11.3 11.4 11.5

11.6 11.7

*12* ***Now what?***

12.1

# contents

Where we can go further 263

*GPT quantization 263 NormalFloat4 263*

*Quantization-aware training 264 Quantization kernel*

*profiling 264 The road ahead 265*

Exercises 265

***Distributed computing 268***

Understanding the hardware 269

*Lab 1: Mapping your hardware 270 Understanding NCCL's*

*primitives 270 Ring vs. tree algorithms 276*

Tensor parallelism 278

*Initialization and GPU assignment 279 Memory allocation 281*

*Local matrix multiplication 281 Combining the results*

*with AllReduce 282* Pipeline parallelism 284

*Naive pipeline parallelism implementation: Sequential processing 285 Optimized implementation: CUDA*

*streams 287 Why streams work 289*

Evaluating scaling strategies 289

*Parallelism strategies 290 When to use each strategy 293*

***296***

Why CUTLASS matters 297

*The core idea: Decomposing a GEMM 298* Building our first CUTLASS kernel 299

*Defining the kernel's shape 299 Composing the kernel with*

*CollectiveBuilder 300 Launching the kernel from the host 301*

CuTe: The language of layouts 302 Tuning for performance 303 Architecture and precision 305

*Across the generations: Ampere vs. Hopper 305 Narrow*

*precision operations: FP4 on Blackwell 307* Profiling and verification 309 Beyond a single GPU: Distributed GEMM 310

***312***

What this book intentionally left out 313

*Data center-level considerations 313 Convolution-heavy*

*architectures 314 The Blackwell TCGen05 instructions 314*

-----

12.2

12.3

12.4 12.5

12.6

12.7

12.8 12.9 12.10

*appendix A appendix B appendix C appendix D*

*appendix E*

**contents xi**

Choosing the right abstraction layer 315

*Triton: Python-level GPU programming 315 torch.compile 316*

Serving systems and end-to-end performance 316

*vLLM 316 SGLang 317 TensorRT-LLM 317 ONNX 318*

---

Compiler and generated-kernel workflows 319 High-level frameworks and languages 319

*JAX 319 Mojo 320*

CUTLASS and template libraries 320

*Python CuTe DSL 320 CUTLASS profiler and auto-tuning 321*

Expanding your toolkit 321

*GPU MODE 321 Ultra-Scale Playbook 321 LeetGPU 322*

High-level inference tricks 322 Running on edge hardware 322 Charting your next moves 322

*The meta-point: The fundamentals generalize 323 Where to go*

*from here 323 Final thoughts 323 Resources 323*

*Installation and setup 326 Running the code examples 330 Troubleshooting and development practices 340 CUDA reference: commands, hardware, templates, and bindings 347 Additional resources 371 index 378*

-----

# preface

---

I wrote this book because it is the one I wanted when I was starting out. I did not come to CUDA as someone with 10 years of performance engineering and distributed systems experience. I came to it as a curious person who wanted to understand what was happening underneath the deep learning tools I was already using.

When I first tried to learn CUDA for deep learning, I often found myself missing the connective tissue between ideas. I could find explanations of CUDA mechanics, and I could find examples of sophisticated kernels, but I still had to work out how the pieces lined up: how threads map to tensors, how memory access patterns become performance problems, how reductions show up inside ordinary model operations, and why attention changes the shape of the whole optimization problem.

That process of connecting the dots became this book. Many of these concepts were not perfectly clear to me the first time I encountered them. Writing the chapters forced me to slow down, build the examples, test the kernels, check the performance claims, and teach each idea back carefully enough that I knew I understood it. The result is a path through CUDA that follows the order in which the pieces became clear to me.

I think that perspective matters. An expert who has lived in this world for a decade can forget which steps are nonobvious at the beginning. I tried to keep those steps visible: what the GPU is doing, why the code is shaped the way it is, where the profiler is pointing, and how a small kernel-level choice connects to the larger deep learning workload.

Just as important is knowing when not to write custom CUDA. Most of the time, a high-quality framework operation, vendor library, or generated kernel is the right answer. Custom kernels are valuable when they earn their complexity: when you need fusion, unusual memory layouts, hardware-specific behavior, or performance that the

**xii**

-----

**preface xiii**

framework path cannot provide. I want readers to leave with that judgment, not just with syntax.

The book moves from CUDA foundations through the kernels that matter most for transformers and modern inference: GEMV and GEMM, softmax, layer normalization, top-k, tensor cores, Flash Attention, quantization, and distributed communication. It then steps up to CUTLASS and production concerns, where the question becomes not only "can we write this kernel?" but "should we, and how do we maintain it?"

If this book saves you some of the searching, false starts, and disconnected explanations that I went through, it will have done its job. I learned much of this the slow way, and I wrote down the version I wish someone had handed me at the beginning.

-----

# acknowledgments

---

Thank you, development editor Connor O'Brien, for guiding this book with patience, clarity, and a steady sense of what the reader needed next. Connor helped turn a long sequence of experiments, diagrams, revisions, and structure questions into a manuscript that a reader can actually follow.

Thank you, Ruud Gijsen, for the technical review work that made the manuscript sharper. Ruud is a senior software engineer with expertise in high-performance computing and simulation engines. A book about CUDA and performance has to be precise, and the review process caught places where a sentence, diagram, benchmark, or claim needed to be more careful.

Thank you, technical proofreader Mohamed Maher, for checking the final manuscript with care and helping ensure that the code, terminology, and technical explanations remained accurate.

Thank you, the Manning team and the reviewers who read drafts along the way: Abhijeet Rajwade, Adarsh Mittal, Aditya Lad, Alejandro Cuevas Rivero, Alexander Davies-Moore, Amber Whitehead, Angelo Liseno, Chancellor T. Pascale, Chandrani Mukherjee, Chrilly Donninger, Clyde Kallahan, Deep Bodra, Deep Shah, Divakar Yadav, Éric Platon, Francesco Basile, George Ehrhardt, Hongyu Cai, James Melvin Ebenezer, Jenny Vega,Jiabao Chen, Karthikeyan Manivannan, Krzysztof Kamyczek, Lars Nyland, Less Wright, Maicon Melo Alves, Michael Seewald, Onur Akdemir, Peng DU, Pengzhan Zhao, Pooja Walia, Randy J. Ray, Ravi Teja Reddy Mandala, Recep Erol, Roman Pavlov, Rutvij Karkhanis, Saloni Garg, Sean Settle, Seema Phalke, Sergey Polyashov, Shujian Qian, Sixie Fang, Smit Kadvani, Srujith Poondla, Stefan Turalski, Stéphane Daigle, Suryansh Singh, Trevor Smith, and Yuan Lin.Your questions and suggestions improved the pacing, figures, explanations, and reader focus throughout the manuscript. A

**xiv**

-----

**acknowledgments xv**

finished technical book hides a large amount of editorial and production work, and I am grateful for the people who did that work here.

I thank Voltage Park for helping make the compute-heavy parts of this book possible. Many of the later chapters benefit from being tested against real hardware rather than only described in the abstract.

I thank FR8 for giving me the space, time, and cohort energy to keep writing when the book needed sustained focus. Having room to think about the book as a real project, not just something squeezed into the margins, mattered more than I can neatly summarize.

And I thank my family, who listened to me walk circles around the kitchen island thinking out loud about kernels, diagrams, performance results, and how to teach one more hard thing clearly. This book exists because I had room to go through that process, and you made that possible in the most practical sense.

-----

# about this book

---

*CUDA for LLMs teaches CUDA programming through the deep learning workloads* where GPU performance matters most. Rather than presenting CUDA as a set of isolated language features, the book builds a practical mental model of how GPU hardware executes neural network kernels, moves data through memory, and exposes performance tradeoffs that high-level frameworks often hide.

The book starts with the foundations: kernels, threads, blocks, memory transfers, indexing, and simple neural network operations. From there it moves into increasingly realistic deep learning kernels: transformer inference, GEMV and GEMM, softmax, layer normalization, top-k, tensor cores, Flash Attention, quantization, distributed communication, and CUTLASS. Each topic is tied back to the same question: what is the GPU doing, and how should that change the code we write?

Throughout the book, you will profile kernels, reason about memory bandwidth and occupancy, compare implementation strategies, and learn when a custom CUDA kernel is worth its complexity. The point is not to replace frameworks for the sake of it. The point is to understand the layer beneath PyTorch, JAX, TensorRT, and other high-level tools well enough to make better performance decisions.

###### Who should read this book

This book is for engineers, researchers, and technically curious builders who use deep learning systems and want to understand what happens below the framework layer: AI and machine learning engineers who want lower latency or cheaper inference than the default framework path provides, systems and performance engineers moving into GPU work, and researchers who need to verify, debug, or extend the kernels their models depend on.

**xvi**

-----

**about this book xvii**

Readers should be comfortable reading Python and C or C++ code. They should know basic C concepts (pointers, arrays, functions, and loops), and experience with NumPy or basic PyTorch is enough on the Python side. Linear algebra at the level of matrix multiplication and dot products helps with the deep learning applications, but the book focuses on computational patterns rather than mathematical theory. Prior CUDA experience is not required: the early chapters introduce CUDA from first principles, and later chapters show how those fundamentals scale into modern GPU programming patterns.

## How this book is organized: A road map

The book has 12 chapters and 5 appendices:

¡ Chapter 1 makes the case for learning CUDA in a deep learning world: what

GPUs are good at, how the optimization stack fits together, and what hardware you need to follow along. ¡ Chapter 2 has you write and run a CUDA kernel immediately, then deconstructs

it to build the fundamental concepts: GPU architecture, the memory hierarchy, and warps. ¡ Chapter 3 implements the essential building blocks of neural networks from

scratch: matrix operations, activations, convolutions, and more. ¡ Chapter 4 completes a full neural network (an MLP for MNIST) from the ground

up, moving from PyTorch to backpropagation theory to NumPy to naive C to naive CUDA to cuBLAS, teaching the end-to-end logistics of a training pipeline. ¡ Chapter 5 shifts from training to inference. After covering the transformer architecture, it implements naive CUDA kernels for autoregressive generation: prefill, decode, key-value caching, and mixture-of-experts routing. ¡ Chapter 6 introduces profiling and systematically optimizes the earlier kernels

through better memory access patterns and more efficient algorithms. ¡ Chapter 7 covers tensor cores, the specialized hardware units that accelerate

matrix operations, across different GPU generations. ¡ Chapter 8 builds Flash Attention, which fuses multiple attention operations into

a single kernel and dramatically reduces memory usage by avoiding large intermediate matrices. ¡ Chapter 9 covers quantization: compressing models by reducing numerical precision while maintaining accuracy. ¡ Chapter 10 covers multi-GPU execution on a single node. ¡ Chapter 11 uses CUTLASS, NVIDIA's open template library for high-performance

matrix multiplication, to build efficient GEMM kernels without writing every tiling detail by hand. ¡ Chapter 12 steps back and looks at the wider ecosystem: how to choose the right

abstraction layer, how serving systems fit together, and how the CUDA skills in this book stay relevant as the field evolves.

-----

**xviii about this book**

The appendices collect the practical reference material: appendix A covers installation and setup; appendix B documents how to run each chapter's code examples; appendix C is a troubleshooting and development-practice guide; appendix D gathers command references, CUDA templates, and Python kernel bindings; and appendix E lists additional resources for continued learning.

Read chapters 1 through 6 in order; they build the programming model, the neural network workloads, and the profiling discipline everything else depends on. Chapters 7 through 11 each stand on that foundation and can be read selectively, though they are ordered so that each one deepens the performance story of the one before it.

## About the code

This book contains many examples of source code, both in numbered listings and in line with normal text. In both cases, source code is formatted in a fixed-width font

```
like this to separate it from ordinary text. Annotations accompany many of the list-
```

ings, highlighting important concepts.

You can get executable snippets of code from the liveBook (online) version of this book at https://livebook.manning.com/book/cuda-for-llms. The complete code for the examples in the book is available for download from the book's GitHub repository at [https://github.com/infatoshi/book.cu and from the Manning website at https://](https://github.com/infatoshi/book.cu) [www.manning.com/books/cuda-for-llms. The repository is organized by chapter, and](https://www.manning.com/books/cuda-for-llms) appendix B walks through building and running every example, including the exact compiler flags and the hardware each example was verified on. The kernels were validated on consumer hardware (an RTX 3090), with the tensor core and distributed material additionally verified on data center GPUs (H100). You do not need data center hardware to follow the book: appendix A includes cloud options, and every chapter's core examples run on a single consumer GPU.

###### liveBook discussion forum

Purchase of CUDA for LLMs includes free access to liveBook, Manning's online reading platform. Using liveBook's exclusive discussion features, you can attach comments to the book globally or to specific sections or paragraphs. It's a snap to make notes for yourself, ask and answer technical questions, and receive help from the author and other users. To access the forum, go to https://livebook.manning.com/book/cuda [-for-llms/discussion.](https://livebook.manning.com/book/cuda-for-llms/discussion)

Manning's commitment to our readers is to provide a venue where a meaningful dialogue between individual readers and between readers and the author can take place. It is not a commitment to any specific amount of participation on the part of the author, whose contribution to the forum remains voluntary (and unpaid). We suggest you try asking the author some challenging questions lest their interest stray! The forum and the archives of previous discussions will be accessible from the publisher's website as long as the book is in print.

-----

# about the author

---

###### Elliot Arledge created the official 12-hour CUDA

course for FreeCodeCamp. He also authored a course on large language model creation for FreeCodeCamp. He is a trainer and independent consultant on CUDA and deep learning performance optimization. He has developed a visual teaching style using common objects like rectangles, cubes, and prisms to introduce and demonstrate complex topics like softmax, tensors, embeddings, and attention.

**xix**

-----

# about the cover illustration

---

The figure on the cover of CUDA for LLMs is "Marchand d'Indostan," or "Habit of a merchant of Indostan in 1700," taken from a collection by Charles Grignion, published 1757-1772. Each illustration is finely drawn and colored by hand.

In those days, it was easy to identify where people lived and what their trade or station in life was just by their dress. Manning celebrates the inventiveness and initiative of the computer business with book covers based on the rich diversity of regional culture centuries ago, brought back to life by pictures from collections such as this one.

**xx**

-----

# 1

## When PyTorch just isn't enough

---

| This chapter covers |
|---|
| ¡ The CUDA programming model and parallel computing fundamentals |
| ¡ Why deep learning operations are suited for GPU |
| parallelization |
| ¡ When to use CUDA versus PyTorch |
| ¡ The optimization ladder from naive kernels to |
| production performance |

Large language models (LLMs) consume staggering compute budgets. The difference between a model that trains in days rather than weeks, or infers in milliseconds rather than seconds, often comes down to how effectively you use GPU hardware. That optimization happens in CUDA, the layer PyTorch calls into when it runs on an NVIDIA GPU.

Writing raw CUDA is the most direct way to control what the GPU executes. This book teaches you to recognize when you need a custom CUDA kernel instead of an existing library and how to write that kernel yourself.

-----

###### 2 Chapter 1 When PyTorch just isn't enough

**NOTE** Higher-level abstractions like OpenAI's Triton exist, but understanding raw CUDA gives you deeper control and insight. My goal is to take you from curious about GPU programming to a CUDA developer who can spot performance bottlenecks, work close to the hardware, and craft optimization strategies that actually run. To start, we'll build your mental model of CUDA from the ground up, beginning with host/device coordination and simple CUDA programs before connecting them to deep learning workloads. We'll use examples like matrix multiplication to map out parallel opportunities, then point you to the exact tools and first projects you'll use to run your own CUDA code.

## 1.1 What is CUDA?

If you've ever played around with PyTorch to train a neural network, you've probably stumbled across a line of code that looks like the following listing.

**Listing 1.1 PyTorch CUDA model transfer**

```
model = model.to("cuda")
```

###### Transfers the model and its parameters to GPU memory, enabling parallel computation

That one line moves the model to the GPU, and for many tasks that's all you need. Frameworks like PyTorch and JAX abstract away GPU programming. But what if you need something they don't offer? What if a specific operation is bottlenecking your pipeline, or you're targeting 240 FPS inference on embedded hardware and standard operations aren't fast enough? (We'll cover how to measure performance and identify bottlenecks using profiling tools in chapter 6.) That's where CUDA comes in. CUDA is NVIDIA's programming model for writing code that runs directly on the GPU. When you call model.to("cuda"), PyTorch uses CUDA under the hood. This book teaches you to write that low-level code yourself, giving you control over exactly what the GPU executes. This book shows you how to write your own high-performance GPU code and then, optionally, integrate it back into familiar frameworks like PyTorch.

## 1.2 When do you need custom CUDA?

Most of the time, you don't. The engineers at NVIDIA, Meta, Google, and OpenAI who build cuBLAS, cuDNN, and framework backends are exceptionally good at their jobs. PyTorch handles the vast majority of use cases. But there are situations where custom CUDA makes sense. Following are a few examples of engineers who write custom kernels. A contributor to a machine learning framework, for instance, might be working on the next version of an attention mechanism and discover that the current implementation doesn't efficiently handle a new variant. They would need to write a custom CUDA kernel that not

-----

###### When do you need custom CUDA? 3

only supports new patterns like sliding window attention but is also optimized for the latest hardware, such as B200 tensor cores. The kernel must also plug into the framework's automatic differentiation system (the machinery that computes gradients during training, so users can call loss.backward() without thinking about the underlying math). This

kind of work builds the foundational tools other developers rely on. Similarly, a kernel engineer at a company like xAI working on a large language model

might profile their training pipeline and discover that standard attention kernels don't scale to their massive context lengths. Their job is to write custom kernels that can efficiently handle sequences of over a million tokens (the individual units a language model reads, roughly corresponding to words or subwords). This involves developing clever tiling strategies to minimize memory usage and tuning the kernel to achieve maximum throughput on their specific cluster hardware. A custom CUDA implementation in this context could deliver a 2-3× speedup, which translates directly into millions of

dollars in savings on training and inference costs. The edge deployment engineer at a robotics company faces a different set of constraints.

They need to deploy vision models on power-constrained embedded GPUs, which require real-time inference with latencies under 10 milliseconds. Standard framework operations are often too slow or power hungry. This engineer writes custom CUDA programs to fuse multiple operations, reducing memory bandwidth, a common bottleneck. They might also implement custom int8 quantization and dequantization kernels and tune their code for the unique memory layout of edge-class GPUs to enable applications like autonomous drones and self-driving cars. An algorithms researcher developing a novel architecture might need custom CUDA

to demonstrate competitive performance. Although many new operations can be composed from existing PyTorch primitives, some genuinely require custom implementations. The FlashAttention paper (https://arxiv.org/abs/2205.14135), for example,

required custom CUDA to achieve its memory efficiency gains.

| But can you actually beat PyTorch and Hugging Face? |
|---|
| Yes, often you can, and you do not need decades of GPU experience to start. Sev- |
| eral of the kernel-optimization walk-throughs in later chapters were inspired by, and in places refactored from, work done by people who were themselves early in their ker- nel-engineering journey. They were not career CUDA programmers; they were curious engineers who picked up the right reference material, followed a careful step-by-step optimization path, and were able to beat the corresponding PyTorch baselines on spe- cific GPUs with surprisingly few hours of hand optimization. |
| Once you accept that humans without a heroic background can do this, the next step |
| is mechanical: give an LLM the same step-by-step documentation, the same refer- ence material, and the same direction, and it can do the same thing autonomously. That is what the companion benchmark suite at https://kernelbench.com does. It runs a combinatorial sweep across GPU architectures, frontier models, and kernel problems, asking LLMs to write CUDA and Triton kernels and then grading the results |

-----

###### 4 Chapter 1 When PyTorch just isn't enough

| (continued) |
|---|
| against hardware rooflines and optimized PyTorch baselines. The published tables |
| are sortable, and the results, from roofline percentages to double-digit decode speed- ups over the reference implementation, speak for themselves. |
| PyTorch is the standard library that most deep learning code is built on. Hugging Face Transformers is the library most people use on top of PyTorch to load and run popular models like Llama or GPT-style architectures. Under the hood, when you call something like a matrix multiply or an attention layer, both libraries dispatch to fast, prewritten GPU programs. These GPU functions are called kernels. Most of those prewritten kernels are good, but they are not the fastest possible, for two reasons. First, the people maintaining these libraries have to be careful: a kernel that breaks one user's workflow is a bigger problem than one that runs a little slower for every- one, so they leave a lot of speed on the table on purpose. Second, NVIDIA releases new GPUs (such as the recent H100 and B200) that ship with new hardware fea- tures, and the libraries take a while to catch up. In the meantime, somebody who knows CUDA can write a kernel that uses those new features and run circles around the default. |
| Even if a model can write a kernel that beats what most expert kernel engineers would produce, someone still has to understand that kernel before it ships. Gener- ated kernels are unusually verifiable in isolation: you can compare outputs against a reference, inspect intermediates, and benchmark on fixed shapes, and good work- flows do all of that. But a test suite only tells you that the kernel passed the checks you thought to write. A reduction reordered in a way that loses precision, or an accu- mulation done in the wrong data type, can slip through and surface later as slightly wrong activations and the wrong tokens coming out of a production model. Trusting code you cannot read is not an engineering strategy. This book gives you the building blocks to open a fast machine-generated kernel and reason about the optimizations inside it: why the tiling is shaped that way, what the memory access pattern costs, and where the arithmetic can drift. |
| None of this is an argument against PyTorch. PyTorch is the right starting point for almost all deep learning work, and you should keep using it. The point of this book is to teach you the layer underneath so that when the default is not fast enough, you have somewhere to go. |

## 1.3 CUDA basics

This section covers the fundamental concepts: host versus device, kernels, and threads. These form the foundation for everything else in the book.

### 1.3.1 Host and device: Two worlds working together

In CUDA programming, you're always working with two separate computing environments that must coordinate to accomplish your goals. The host is your CPU and its associated system memory (RAM). This is where your main program runs and where you

-----

###### CUDA basics 5

manage the overall flow of your application. The device is your GPU and its dedicated video memory (VRAM). This is where the parallel work runs.

| What is the V in VRAM? |
|---|
| VRAM stands for video RAM. As the name suggests, it was originally designed to hold |
| video and graphics data. Think of the massive textures, complex 3D models, and lighting information needed to render a realistic scene in a video game. These are all just large arrays of numbers that need to be processed in parallel to update the mil- lions of pixels on your screen many times per second. |
| It turns out that the mathematical operations used in graphics, such as multiplying matrices to transform 3D coordinates and applying filters to textures, are strikingly similar to the operations at the heart of deep learning, such as matrix multiplications in dense layers and convolutions for image processing. The hardware architecture built to accelerate graphics, with its thousands of simple cores for parallel process- ing, was perfectly suited to the massive, parallel computations required to train and run neural networks. So the "video" RAM became the workhorse for AI. |

A CPU has a few complex cores with large caches, built to run one thread of sequential, branchy code as fast as possible. A 16-core desktop CPU such as AMD's Ryzen 9

9950X3D runs at up to 5.7 GHz, and each core can do two 512-bit fused multiply-adds per clock, 32 FP32 multiply-adds in all. The RTX 3090 used in this book has 10,496 far simpler CUDA cores at a 1.7 GHz boost clock, each doing one FP32 multiply-add per clock. In total, the GPU peaks at about 35.6 TFLOPS of FP32, while the CPU stays under 6 TFLOPS even if all 16 cores held 5.7 GHz. GPU cores are built to execute the same instruction on different data simultaneously. As illustrated in figure 1.1, this tradeoff is visible in how chip real estate is allocated: GPUs devote far more silicon to arithmetic

logic units (the cores) and less to cache and control flow units than CPUs do. As we see in figure 1.1, this design enables GPUs to excel at parallel tasks where

the same operation is performed on different data simultaneously. These two worlds must communicate and coordinate. Because the host and device have separate memory spaces, you need to explicitly manage data movement between them: allocating GPU memory, copying data to the device, running your kernel, and copying results back. Your host code orchestrates this workflow. Figure 1.2 illustrates this four-step data flow

pattern that every CUDA program follows.

### 1.3.2 Kernels: Functions that run on thousands of threads

What's a kernel? Are we talking about popcorn kernels, Linux kernels, military colonels, or convolution kernels (aka convolution filters)? Nope! A CUDA kernel is a function that runs on the GPU. But unlike regular functions, which execute once, kernels are designed to be executed simultaneously by thousands of threads. Each thread gets a small piece of the overall job and works on it independently. You write the kernel as if you're programming for just one thread, describing what it should do with one piece of data. CUDA

-----

###### 6 Chapter 1 When PyTorch just isn't enough

###### Control unit Core L1 cache CPU GPU

Core Core

Control Control

L1 cache L1 cache Core Core

Control Control

L1 cache L1 cache

L2 cache L2 cache

---

L3 cache L2 cache

---

System memory VRAM

---

**Figure 1.1 A CPU dedicates more area to caches and control logic for a few powerful cores; a GPU dedicates more area to many simpler arithmetic cores.**

| HOST (CPU) |  | DEVICE (GPU) |
|---|---|---|
| RAM | PCIe Bus | VRAM |
| Host Code (main.cpp) |  | CUDA Cores (kernel.cu) |

| 1 | cudaMalloc() | Allocate GPU memory |
|---|---|---|
|  | 2 cudaMemcpy H→D | Copy data: Host → Device |
|  | 3 kernel<<<...>>>() | Execute parallel kernel |
|  | 4 cudaMemcpy D→H | Copy results: Device → Host |

**Figure 1.2 The four-step CUDA workflow: allocate GPU memory, copy data to the device, execute the kernel, and copy results back.**

automatically handles launching thousands of copies of that function across the GPU's many cores. Think of it like this: if you need to add 1 million pairs of numbers, instead of writing a loop that performs 1 million additions sequentially, you write code that

-----

###### Recognizing parallel opportunities 7

does one addition, then launch 1 million threads, each doing that one addition simultaneously. Figure 1.3 shows this kernel execution model: write code for one thread, and CUDA launches thousands of parallel copies.

---

###### Grid

Block (0,0) **Kernel Code** T0 T1 T2 Block (0,1)

T3 T4 ...

| __global__void kernel(...) { int idx = threadIdx.x; | Launch |
|---|---|
| // process data [idx] } | <<<grid, block>>> |

**You write code for ONE thread** Block (1,0) Block (1,1)

###### ...thousands of threads

---

###### Write code for ONE thread → CUDA launches THOUSANDS of parallel copies

**Figure 1.3 Write code for one thread; CUDA launches thousands of parallel copies, each with a unique thread index.**

## 1.4 Recognizing parallel opportunities

Every time you see an operation applied independently to many pieces of data, that's a parallel opportunity.

Computer scientists call this pattern SIMD (Single Instruction, Multiple Data). It's one of the four categories in Flynn's taxonomy, the standard classification of processor architectures (https://doi.org/10.1109/TC.1972.5009071). GPUs use a closely related model called SIMT: Single Instruction, Multiple Threads. The distinction matters: in pure SIMD, every lane executes the same instruction in lockstep, with no ability to diverge. SIMT also runs thousands of threads on the same instruction, but individual threads can branch independently, and the hardware handles divergence transparently. For the data-parallel patterns we care about in deep learning, the effect is similar to SIMD, but SIMT gives us more flexibility when control flow varies across threads.

The patterns to look for are as follows: ¡ Same operation, different data → Perfect for GPU parallelization ¡ Independent computations → Can run simultaneously ¡ Large data volumes → More threads, better GPU utilization ¡ Regular access patterns → Efficient memory usage

-----

###### 8 Chapter 1 When PyTorch just isn't enough

This mental model applies everywhere in deep learning:

¡ Activation functions: Apply rectified linear unit (ReLU) to 1 million values →

launch 1 million threads ¡ Convolutions: Compute 10,000 output pixels → Launch 10,000 threads ¡ Attention: Calculate 1 million token relationships → Launch 1 million threads ¡ Gradient updates: Update 100 million parameters → Launch 100 million threads Figure 1.4 illustrates this pattern with a concrete example: applying ReLU activation to a small matrix. Each element gets its own thread, and all threads run the same max(0, x) operation in parallel.

###### Input matrix Output matrix

-2.1 3.5 -0.8 1.2 0.0 3.5 0.0 1.2 max(0, x) 0.5 -4.2 2.7 -1.9 0.5 0.0 2.7 0.0

| Thread | assignment: | one | thread | per element |
|---|---|---|---|---|
| T0 | T1 | T2 | T3 | 8 elements 8 threads |
| T4 | T5 | T6 | T7 | All run in parallel |

**Figure 1.4 Element-wise ReLU: each matrix element is processed by a dedicated thread running the same operation.**

Throughout this book, we'll follow this pattern: We'll start by describing each problem at a high level, show how frameworks like PyTorch handle it, and then progressively dive deeper into CUDA implementations. You'll see how the same mathematical logic translates across different levels of abstraction, from framework APIs down to raw GPU threads.

### 1.4.1 The memory hierarchy: Your performance bottleneck

Modern GPUs have a complex memory hierarchy, and understanding it is crucial for writing fast code. Think of it like a pyramid: the tiers below live on the device itself, while system RAM sits at the very bottom, reaching the GPU only through PCIe or NVLink links. The tiers, from largest and slowest to smallest and fastest, are as follows: ¡ *Global memory (HBM)-Large (roughly 8 GB on entry-level consumer cards to* 180 GB or more on current data center parts) but the slowest on-device tier. Data center GPUs use high-bandwidth memory (HBM), delivering roughly 2 TB/s on A100-class parts, 3.35 TB/s on the H100 SXM, and up to 8 TB/s on Blackwell.

-----

###### Recognizing parallel opportunities 9

High-end consumer GPUs use GDDR6X at roughly 1 TB/s or GDDR7 at up to 1.8 TB/s (RTX 5090), while midrange cards use GDDR6 at closer to 270 to 450 GB/s. Access latency is on the order of hundreds of clock cycles. This is where your main data lives. ¡ *L2 cache-A device-wide cache (up to ~50 MB on H100) that automatically buf-*

fers reads from and writes to global memory. An L2 hit returns in roughly 250 to 275 cycles; a miss falls through to global memory at roughly 470 to 540 cycles, about twice the L2 latency. Keeping frequently reused data resident in L2 makes the difference. ¡ *Shared memory / L1 (per SM)-Small (a 128-256 KB SRAM pool per streaming*

multiprocessor, depending on the GPU) but much faster, with around 30 to 40 cycles of latency and aggregate bandwidth in the tens of TB/s across the chip. An SM is the hardware unit that schedules your threads; chapter 2 introduces the basics, and we revisit SM architecture as optimizations demand it. CUDA lets you decide how much of this on-chip pool acts as programmer-managed shared memory versus hardware-managed L1 cache. ¡ *Registers-Tiny (up to 255 registers per thread) but effectively instantaneous, with*

single-cycle access. This is where your variables live during computation. The art of optimization often comes down to keeping your data in the fastest possible

SM SM SM

memory tier for as long as possible. Figure 1.5 shows these on-device tiers, with system memory at the bottom, on the far side of the PCIe L1 L1 L1 gap, and illustrates the tradeoffs between size and speed. If the terminology still feels

L2 cache

abstract, that's expected. Chapter 2 turns each tier into something tangible as you build your first CUDA program. GPU VRAM Consider our matrix multiplication example. A naive implementation might read each input element from global memory System memory thousands of times (once for each output it contributes to). An optimized version uses **Figure 1.5 Memory hierarchy on a GPU** shared memory to cache frequently accessed data, reducing global memory traffic by orders of magnitude. Sometimes the fastest algorithm does more operations but has a more GPU-friendly memory access pattern.

**The complete optimization journey: From naive to production**

The techniques in this book can deliver substantial performance improvements when combined systematically. Figure 1.6 shows the optimization ladder we'll climb together. These are the same techniques used in production AI systems.

-----

###### 10 Chapter 1 When PyTorch just isn't enough

As figure 1.6 shows, the optimization ladder builds fast compound improvements through systematic tech-

| Multi-GPU (final boss) |
|---|
| Quantization |
| FlashAttention 1-4 |
| GPU architecture* |
| 5-7x |
| Naive CUDA |

¡ *The compounding effect-Each* optimization 1.0x

technique multiplies the benefits of the others.

slow

Memory optimizations, tensor cores, kernel fusion, and quantization each contribute sig- **Figure 1.6 The optimization**

**ladder shows multipliers at**

| nificant | gains. When combined, | these | improve- | each level of optimization |
|---|---|---|---|---|
| ments | compound to deliver | multiple | orders | covered throughout the book. |
| of | magnitude better | performance | than naive | These represent the compound |

**effects on efficiency and**

implementations. **performance from latency,**

¡ *The hardware evolution-The* timeline shows **memory savings, occupancy,**

**throughput, and cost savings.**

why understanding modern GPU architectures matters. The tensor core capabilities that unlock roughly 2× speedups on my RTX 3090 (which can accumulate in FP32 or FP16; FP32 accumulation runs at half rate on GeForce Ampere) become even more powerful on the Hopper and Blackwell architectures. ¡ *The real-world effect-These optimizations translate directly to practical benefits:*

reduced training time, lower compute costs, and the ability to run larger models on the same hardware through techniques like not materializing intermediate matrices (as in Flash Attention) and aggressive quantization.

###### 1.4.2 When parallelism doesn't help

Recognizing parallel opportunities also means recognizing when GPU acceleration doesn't help. Choosing between a CPU and a GPU involves a tradeoff: there is overhead in transferring data, launching kernels, and managing increased programming complexity. For some problems, that overhead outweighs the benefits. The following are a few examples where a CPU is often the better choice:

-----

###### Deep learning through the CUDA lens 11

¡ *Small-scale problems-If you're working with tiny datasets or simple models, the*

overhead of moving data to the GPU might exceed the computational benefits. A 10 × 10 matrix multiplication is faster on a CPU. ¡ *Low arithmetic intensity-When the ratio of computation to memory access is low,*

you're limited by memory bandwidth regardless of compute capability. Some operations simply can't hide memory latency with computation. ¡ *Irregularworkloads-Operations with unpredictable branching or data-dependent*

control flow can't utilize GPU cores efficiently. When different threads need to execute completely different code paths, parallelism breaks down due to warp divergence. Knowing when GPUs help and when they don't is part of thinking like a performance engineer.

## 1.5 Deep learning through the CUDA lens

A brief history provides useful context for understanding CUDA's role today:

¡ *CUDA-NVIDIA released CUDA in 2007, and it immediately found adoption*

in high-performance computing (HPC) and scientific computing. Researchers in physics simulations, computational fluid dynamics, and financial modeling were among the first to use GPUs for general-purpose computation. The programming model that CUDA introduced-writing C-like code that runs on thousands of parallel threads-opened up GPU hardware to applications far beyond graphics. ¡ *Deep learning (DL)-The 2012 AlexNet paper (https://dl.acm.org/doi/10.5555/*

[2999134.2999257) demonstrated that neural networks trained on GPUs could](https://dl.acm.org/doi/10.5555/2999134.2999257) dramatically outperform CPU-trained models in image classification. AlexNet was specifically designed to run on two GPUs, splitting the model across devices. This wasn't just about raw compute: it showed that neural network architectures could be designed around GPU parallelism. The deep learning explosion that followed built directly on CUDA and the GPU programming patterns established in the HPC community. Today, CUDA underpins both scientific computing and AI workloads. The techniques in this book apply to both domains.

### 1.5.1 The nature of deep learning computation

DL is fundamentally about matrix operations-lots and lots of them. When you train a neural network, you're essentially doing the same mathematical operations repeatedly on massive arrays of numbers. The most common of these are

¡ *Matrix multiplication-General matrix multiplication, or GEMM, computes C =*

```
alpha * A @ B + beta * C: a matrix product, scaled, and accumulated into an exist-
```

ing matrix. It forms the core of every dense layer. A dense layer (also called a

-----

###### 12 Chapter 1 When PyTorch just isn't enough

*fully connected layer) connects every input neuron to every output neuron, with* connection strengths defined by a weight matrix, so its forward pass is Y = X @ W +

```
b, where the product is the GEMM and the bias add rides in the accumulate term
```

or in a fused epilogue. A single forward pass through a large transformer model involves hundreds of GEMMs, each processing millions of parameters. ¡ *Element-wise operations-After a matrix multiplication, a nonlinear activation func-* tion like ReLU (Rectified Linear Unit), GELU (Gaussian Error Linear Unit), or SiLU (Sigmoid Linear Unit) is applied independently to each value in the resulting matrix. Each activation is computed using the same logic but on different data, making this operation a prime candidate for parallelism. ¡ *Convolutions-This* is the key operation in convolutional neural networks (CNNs), often used for image processing. It involves sliding a small filter, or window, across an image or feature map and performing the same dot product computation at every spatial position. For example, a tiny 3 × 3 filter sliding over a 1024 × 1024-pixel image performs millions of identical operations. ¡ *Gradient computations-During the backpropagation phase of training, the model* calculates derivatives (gradients) to learn how to adjust its parameters. This process involves similar patterns of parallel arithmetic, computing these derivatives for millions of parameters simultaneously. ¡ *Attention mechanisms-A key innovation in modern transformers, attention com-* putes the relationships between every pair of tokens in a sequence. This requires a number of dot products that scales quadratically with sequence length, but fortunately, all of these dot products can be computed in parallel. These operations contain massive parallelism. Element-wise operations like activations are truly independent: each output depends on exactly one input. Matrix multiplications and convolutions require some coordination (each output element is a reduction over many inputs), but the sheer number of output elements still provides abundant parallel work. GPUs thrive on this pattern.

### 1.5.2 Why GPUs excel at deep learning

DL workloads involve millions of similar operations. Consider computing ReLU activation on a matrix with 1 million elements. A modern CPU can parallelize this work across its cores and use SIMD instructions to process multiple values per cycle. Even a high-end server CPU with 64+ cores and AVX-512 instructions is impressive, but a GPU approaches the same problem at an entirely different scale. An RTX 4090 has over 16,000 CUDA cores, each simpler than a CPU core but optimized for throughput. GPUs also have far higher memory bandwidth: HBM on data center GPUs like H100 provides over 3 TB/s, compared to about 460 GB/s for a server CPU with 12 channels of DDR5-4800 and under 100 GB/s for a dual-channel desktop. The GPU wins not just on raw compute but also on memory bandwidth. DL operations are often memory

-----

###### The optimization stack 13

bound: the time to load data from memory exceeds the time to compute on it. GPUs are designed around this reality. The transformer architecture that powers today's LLMs and image generators is a perfect illustration. The attention mechanism from Vaswani et al.'s "Attention Is All You Need" computes relationships between every pair of tokens in a sequence: N² independent dot products (https://arxiv.org/abs/1706.03762). Each attention head involves multiple large GEMMs for queries, keys, values, and output projections. The MLP components add more GEMMs with millions of parameters. Consider a sequence of 1,000 tokens: self-attention alone requires roughly 1 million dot products. Training LLMs requires massive GPU clusters not because the algorithms are inherently complex but because the sheer volume of parallel computation demands parallel hardware.

## 1.6 The optimization stack

With the basics covered, we can talk about what makes CUDA programming interesting: the optimization stack. Each chapter in this book adds another layer of optimization, and these layers compound.

### 1.6.1 Starting point: Naive kernels

Your first CUDA kernels will be slower than PyTorch. That's expected. PyTorch operations use highly optimized libraries written by NVIDIA engineers. Your first kernel probably reads from the slowest memory tier and misses obvious optimizations.

The rest of this book teaches you how to close that gap.

### 1.6.2 Optimization layers

Each chapter adds optimization techniques that build on each other: ¡ *Naive kernels (chapter5)-We build inference kernels for a GPT-style transformer,*

tackling the unique challenges of autoregressive generation: prefill versus decode phases, KV caching, and the shift from GEMM to GEMV operations. These naive implementations establish the baseline we optimize in chapter 6. ¡ *Memory and compute optimization (chapter 6)-Profiling reveals* bottlenecks. We learn to optimize memory access patterns and use efficient reduction algorithms, yielding significant speedups over naive implementations. ¡ *Tensor cores (chapter 7)-Modern GPUs include specialized matrix-multiply units*

that dramatically accelerate the operations we optimized manually in chapter 6. ¡ *Kernel fusion (chapter 8)-Flash Attention combines multiple operations into a* single kernel, avoiding memory bandwidth bottlenecks by never materializing large intermediate matrices. ¡ *Quantization (chapter 9)-Reducing numerical precision compresses models by* a factor of 4 to 8 while maintaining accuracy, enabling deployment on memoryconstrained hardware.

-----

###### 14 Chapter 1 When PyTorch just isn't enough

¡ *Distributed computing (chapter 10)-Multi-GPU training requires understanding*

communication patterns and parallelism strategies to scale efficiently across devices and nodes.

### 1.6.3 The compounding effect

When you combine memory optimizations, tensor cores, kernel fusion, and quantization, the improvements compound. Each technique builds on the others. A naive implementation versus a fully optimized one can differ by multiple orders of magnitude in both speed and memory efficiency. Quantization together with Flash Attention, which avoids storing large intermediate matrices, lets you run substantially larger models on the same hardware.

The optimization stack in this book addresses the following: ¡ *Throughput-Faster token generation during inference* ¡ *Memory efficiency-Fitting larger models or longer sequence contexts* ¡ *Training cost-Completing runs in less time with fewer GPUs* ¡ *Latency-Reducing time to first token for interactive applications*

### 1.6.4 Scaling to multiple GPUs

Everything covered so far focuses on single-GPU optimization. Chapter 10 extends these techniques to multi-GPU setups. Different interconnects have vastly different bandwidths. GPUs connected by NVLink communicate far faster than GPUs connected over PCIe.

Master the single-GPU techniques first. The optimization skills from chapters 5 through 9 transfer directly to distributed scenarios, where inefficiencies multiply with each GPU you add.

### 1.6.5 Why this matters

By the end of this book, you'll understand when and why custom kernels can outperform library implementations. You'll see why Flash Attention was a breakthrough and how quantization enables deployment scenarios that would otherwise be impractical. Most importantly, you'll think like a performance engineer: someone who can identify computational bottlenecks and systematically address them.

## 1.7 Getting ready to build

Before we continue, a note on terminology: "CUDA" refers to the entire ecosystem, not just the C++ programming model. This includes CUDA C++ (what we'll write), cuBLAS (optimized linear algebra), cuDNN (DL primitives), CUTLASS (template library), Thrust (parallel algorithms), and more. PyTorch's heaviest operations, such as matrix multiply and convolution, dispatch to cuBLAS and cuDNN, though PyTorch also ships many handwritten CUDA kernels of its own. This book focuses on CUDA

-----

###### Summary 15

C++ because understanding the low-level model helps you reason about performance even when using higher-level libraries. Every NVIDIA GPU you can buy today supports CUDA. For this book, a GPU with compute capability 7.0 or later (V100, or any RTX 20-series card and newer) is enough to follow chapters 2 through 6, which cover the core programming model, neuralnetwork building blocks, and profiling. The current CUDA 13 toolchain no longer compiles for older architectures such as Pascal. Chapters 7 and 8 (tensor cores and Flash Attention) require a GPU with tensor cores, meaning the RTX 20 series (Turing) or newer; some of the chapter 11 CUTLASS examples target the same hardware. The distributed material in chapter 10 is written as an advanced case study, so a single GPU is fine for understanding the concepts, even though the measured runs use an 8× GPU server. Data center GPUs (V100, A100, H100, B200) work great if you have access to them, but no chapter strictly requires one. You'll need the CUDA Toolkit, which provides the nvcc compiler; NVIDIA drivers, which you can verify with nvidia-smi; and a Python environment. I use uv for Python package management. For operating systems, Linux provides the smoothest experience. On Windows, NVIDIA ships a native CUDA Toolkit and also actively supports CUDA inside WSL (Windows Subsystem for Linux), which is the closest match to the Linux workflow used in this book. Detailed installation instructions are provided in appendix A. The beauty is that you can start with basic tools and add advanced profiling and multi-GPU capabilities as you progress through the book. You should be comfortable with basic C concepts: pointers, arrays, functions, and loops. If you've written Python with NumPy or basic PyTorch code, you're set. An understanding of linear algebra at the level of matrix multiplication, dot products, and basic calculus will help you grasp the DL applications. But we focus more on computational patterns than on mathematical theory.

###### Summary

¡ GPUs excel at throughput by running the same instruction across massive data

sets in parallel, making them ideal for deep learning workloads. ¡ Host code (CPU) orchestrates work while device code (GPU) executes it. Every CUDA project involves explicit memory transfers between CPU RAM and GPU memory. ¡ CUDA launches thousands of threads organized into grids and blocks. Each

thread computes a unique index to determine which data element it processes. ¡ Matrix multiplications, convolutions, activations, and attention computations all fit the "same operation, different data" pattern. Recognizing these opportunities is the first skill of a CUDA performance engineer.

-----

###### 16 Chapter 1 When PyTorch just isn't enough

¡ Fast kernels respect the GPU memory hierarchy: registers and shared memory

are scarce but fast, and global memory is plentiful but slower. The key optimization strategy is to keep data in the fastest memory tier for as long as possible. ¡ The rest of this book climbs an optimization ladder, from writing your first CUDA

program in chapter 2 to applying advanced memory and computation techniques in later chapters.

-----

# 2

## Your first CUDA program

---

| This chapter covers |
|---|
| ¡ Deconstructing your first CUDA kernel, vectorAdd, line by line |
| ¡ Understanding the core CUDA programming |
| model: host versus device and data transfers |
| ¡ Mastering the GPU's threading hierarchy: grids, |
| blocks, and threads |
| ¡ Introducing essential C concepts like pointers |
| as they appear in the code |
| ¡ Compiling and running your program with the |
| nvcc compiler |

We are going to take a vectorAdd kernel, put it under a microscope, and dissect it piece by piece. We won't just look at the kernel itself; we'll look at the entire program required to launch it: the "host" code that runs on your CPU. By the end of this chapter, you will have written, compiled, and run your very first complete CUDA program. You will understand not just what the code does, but why it's structured the way it is.

-----

###### 18 Chapter 2 Your first CUDA program

###### 2.1 The "why": Throughput vs. latency

We're going to peek into some code in a second, but first, the reason to use a GPU over a faster CPU comes down to two concepts throughput and latency: ¡ *Latency is* about how fast you can complete a single task. Think of a Formula 1 car. It's built to get one person from point A to point B at blistering speed. CPUs are latency-optimized machines. They have a few incredibly powerful cores designed to execute complex, sequential instructions as quickly as possible. This is perfect for tasks like loading a web page or responding to a mouse click. ¡ *Throughput is about how much total work you can get done in a given amount of* time. Think of a massive cargo ship. It's much slower to start and stop than the F1 car, but it can move thousands of shipping containers from one continent to another. GPUs are throughput-optimized machines. Deep learning is a throughput problem. We don't care about doing one multiplication super fast; we care about doing billions of them. A single GPU operation has more startup overhead than a CPU instruction, but once running, the GPU processes vastly more data per second. That tradeoff defines the entire programming model we're about to learn. A GPU core is a collection of simple execution units. The most important of these is the Arithmetic Logic Unit(ALU). An ALU is a digital circuit that performs basic math and logic operations; think add, multiply, and, and or. A single CPU core has very sophisticated ALUs and a lot of complex control logic around them. A GPU core is much simpler and is often just a handful of ALUs designed for floating-point and integer math. The magic is that a single CUDA thread you write in your code will be scheduled to run on one of these hardware cores. When you launch thousands of threads, you are effectively lighting up thousands of these simple ALUs to perform their calculations at the same time. We'll revisit the deeper architecture of streaming multiprocessors (SMs) as we need it in later chapters, but for now, remember the contrast: a CPU thread is a heavyweight, OS-managed context, whereas a CUDA thread is a lightweight, hardware-managed lane scheduled in groups on the GPU. That difference is why we can launch thousands of CUDA threads without the overhead you'd expect on the CPU.

## 2.2 Your first run

Before we dissect what's happening, let's get something working. Create a new file called vecadd.cu and paste in the complete program shown in listing 2.1. Don't worry about understanding every line yet; that's what the rest of this chapter is for.

**Listing 2.1 Completing vecadd.cu program**

```
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
```

-----

###### Your first run 19

```
__global__ void vectorAdd(float *a, float *b, float *c) {
int i = threadIdx.x;
c[i] = a[i] + b[i];
}
int main() {
int n = 8;
size_t bytes = n * sizeof(float);
float *h_a = (float*)malloc(bytes);
float *h_b = (float*)malloc(bytes);
float *h_c = (float*)malloc(bytes);
for (int i = 0; i < n; ++i) {
h_a[i] = (float)i;
h_b[i] = (float)(i * 2);
}
float *d_a, *d_b, *d_c;
cudaMalloc((void**)&d_a, bytes);
cudaMalloc((void**)&d_b, bytes);
cudaMalloc((void**)&d_c, bytes);
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
vectorAdd<<<1, 8>>>(d_a, d_b, d_c);
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
int success = 1;
for (int i = 0; i < n; ++i) {
if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) { // never compare
floats with ==
printf("Error at index %d: Got %f, expected %f\n",
i, h_c[i], (h_a[i] + h_b[i]));
success = 0;
break;
}
}
if (success) {
printf("All elements are correct.\n");
}
free(h_a);
free(h_b);
free(h_c);
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
return 0;
}
```

Now compile it with the CUDA compiler.

-----

###### 20 Chapter 2 Your first CUDA program

| Listing 2.2 Compiling a CUDA program with nvcc |
|---|
| nvcc vecadd.cu -o vecadd And run it. |
| Listing 2.3 Running the compiled CUDA program |
| ./vecadd You should see the satisfying output: All elements are correct. Congrats, you've just orchestrated a parallel computation across GPU cores. The program added two 8-element arrays together, with each addition happening simulta- neously on a separate GPU thread. Now let's understand exactly what we just ran. NOTE To keep the listings readable, the versions printed in this chapter omit the CUDA error checking (the if (err != cudaSuccess) guards after every API call) that production code should have. The runnable vecadd.cu in the book's code repository includes that error handling, so its exact console output is more verbose than what is shown here. We cover error checking in more detail later; for now, we focus on the structure. A quick note on the toolchain: CUDA source files use the .cu extension. You can edit them in any text editor or IDE you prefer. If you need help setting up your CUDA envi- ronment, see the appendix for installation instructions. Now let's walk through the program we just ran, piece by piece. Most of the magic is in the four lines that include and follow the global keyword. First we have the includes and the kernel itself. The following listing shows the GPU function that does the actual work. |
| Listing 2.4 The vectorAdd kernel |

```
#include <cstdio>
```

| #include <cstdlib> | global marks this as a GPU kernel |
|---|---|
| #include <cmath> | callable from the CPU. The pointers |

`#include <cuda_runtime.h>` **a, b, and c will point to GPU memory.**

```
__global__ void vectorAdd(float *a, float *b, float *c) {
int i = threadIdx.x;
```

`c[i] = a[i] + b[i];` **Each thread gets its unique ID from threadIdx.x** `}` **to determine which element to process.**

###### Performs the addition. All eight threads execute this line simultaneously.

Now for the main() function. First, we set up variables and allocate host (CPU) memory.

-----

###### Anatomy of a CUDA program: The vecadd.cu file 21

**Listing 2.5 Host memory allocation**

```
int main() {
```

`int n = 8;` **Calculates total bytes needed:** `size_t bytes = n * sizeof(float);` **8 floats × 4 bytes each = 32 bytes.**

| float *h_a = (float*)malloc(bytes); | malloc(bytes) allocates memory |
|---|---|
| float *h_b = (float*)malloc(bytes); | and returns a pointer. The h_ prefix |
| float *h_c = (float*)malloc(bytes); | is a convention meaning "host." |

Next, initialize the host arrays with test data.

**Listing 2.6 Initializing host data**

```
for (int i = 0; i < n; ++i) {
h_a[i] = (float)i; // h_a = [0, 1, 2, 3, 4, 5, 6, 7]
h_b[i] = (float)(i * 2); // h_b = [0, 2, 4, 6, 8, 10, 12, 14]
}
```

Now we need GPU memory. The following listing declares device pointers and allocates memory on the GPU.

**Listing 2.7 Device memory allocation**

###### The d_ prefix is a convention meaning "device." These pointers will hold GPU memory addresses.

```
float *d_a, *d_b, *d_c;
```

| cudaMalloc((void**)&d_a, bytes); | cudaMalloc allocates GPU memory. The |
|---|---|
| cudaMalloc((void**)&d_b, bytes); | (void**)&d_a syntax passes the address |
| cudaMalloc((void**)&d_c, bytes); | of our pointer so cudaMalloc can fill it in. |

With memory allocated on both sides, we copy the input data from the host to the device (see the following listing).

**Listing 2.8 Copying data to device**

```
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
```

**cudaMemcpy(destination, source, size, direction) copies**

**data. The direction flag cudaMemcpyHostToDevice specifies that we're copying from CPU to GPU.**

Now for the actual kernel launch.

**Listing 2.9 Launching the kernel**

```
vectorAdd<<<1, 8>>>(d_a, d_b, d_c);
```

###### Launches with 8 threads. The <<<1, 8>>> syntax is explained in detail in the next section.

-----

###### 22 Chapter 2 Your first CUDA program

After the kernel completes, copy results back to the host.

| Listing 2.10 Copy results back to host |
|---|
| cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost); Now the direction is cudaMemcpyDeviceToHost, copying from GPU back to CPU. Finally, verify the results and clean up. |
| Listing 2.11 Verification and cleanup |

```
int success = 1;
for (int i = 0; i < n; ++i) {
if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) { // never compare floats
with ==
printf("Error at index %d: Got %f, expected %f\n",
i, h_c[i], (h_a[i] + h_b[i]));
success = 0;
break;
}
}
if (success) {
printf("All elements are correct.\n");
}
```

| free(h_a); | free() releases host memory |
|---|---|
| free(h_b); | allocated with malloc(). |

```
free(h_c);
```

| cudaFree(d_a); | cudaFree() releases device memory |
|---|---|
| cudaFree(d_b); | allocated with cudaMalloc(). |

```
cudaFree(d_c);
return 0;
```

This sequence of steps forms the fundamental pattern of almost all CUDA programs:

**1** Allocate memory on the CPU (host). **2** Initialize data on the CPU. **3** Allocate memory on the GPU (device). **4** Copy data from host to device. **5** Launch the kernel to compute on the device. **6** Copy results from device back to host. **7** Verify results and free all memory. Figure 2.1 illustrates this pattern as a sequence of eight numbered panels. The panel order differs slightly from the list above: the figure performs both allocations first (panel 1 malloc on the host and panel 2 `cudaMalloc for device memory) and then` initializes the host data in panel 3, whereas the list initializes the host data before

-----

###### Where the parallelism comes in 23

touching the device. Both orders work because the two allocations are independent. From there, the panels follow the list directly: the host-to-device copy (panel 4), the kernel (panels 5 and 6), the device-to-host copy (panel 7), and the use of the results on the host (panel 8). One subtlety: only the kernel in panel 5 actually executes on the GPU. cudaMalloc in panel 2 is a host-issued API call that happens to allocate device memory; the CPU still runs it. The most important concept to grasp in CUDA is that your CPU (host) and GPU (device) have separate memory spaces, as shown in figure 2.1.Your program orchestrates moving data to the GPU, running a calculation there, and moving the results back. Now that we've walked through the memory allocation and copying in listings 2.5 through 2.8, let's focus on the kernel launch and the kernel itself.

## 2.4 Where the parallelism comes in

Look back at the kernel launch from listing 2.9:

```
vectorAdd<<<1, 8>>>(d_a, d_b, d_c);
```

The CPU tells the GPU, "Okay, it's your turn. Run the vectorAdd function." Those triple angle brackets <<<...>>> are CUDA's special syntax for launching a kernel. The numbers inside tell the GPU how many parallel workers to launch. In this case, we're launching 8 threads. What's a thread? A thread is a single instance of our kernel function running on a GPU core. By launching 8 threads, we tell the GPU to run our vectorAdd function 8 times simultaneously. (The 1 in <<<1, 8>>> represents one "block" of threads, a concept we'll explain shortly when we scale up.)

Now look back at the kernel code from listing 2.4:

```
__global__ void vectorAdd(float *a, float *b, float *c) {
int i = threadIdx.x;
c[i] = a[i] + b[i];
}
```

The global keyword says, "This function runs on the device and is called from the host." The line int i = threadIdx.x; is the heart of CUDA programming. threadIdx.x is a built-in CUDA variable that gives each thread a unique ID. CUDA provides several such built-in variables: threadIdx, blockIdx, blockDim, and gridDim. Each of the 8 threads we launched gets its own unique threadIdx.x value, from 0 to 7:

¡ Thread 0 gets threadIdx.x = 0. ¡ Thread 1 gets threadIdx.x = 1. ¡ … ¡ Thread 7 gets threadIdx.x = 7. Each thread uses its unique index i to perform one addition on its assigned element. All eight additions happen at the same time.

-----

###### 24 Chapter 2 Your first CUDA program

###### Initialize input and output pointers before anything else!

malloc on host

CPU DRAM (ram sticks)

---

Populate this allocated **3 4** Copy data from host pointer

memory with data (starting point in memory to float\* arr = {......}; "sizeof(float) \* length of array" (end))

CPU DRAM (ram sticks)

---

\_\_global\_\_ void kernel1 ( input pointer, output pointer, etc... Kernel Completes... ) {gpu kernel code}

---

GPU Global Mem

CPU DRAM (ram sticks)

**1 2**

cudaMalloc() on device

GPU Global Mem

to pointer (from cudaMalloc) on GPU

GPU Global Mem

**5 6**

**7 8**

print output on host!

cudaMemcpy(device to host)

**Figure 2.1 Anatomy of memory management in a CUDA program (host on left, device on right)**

Remember that a, b, and c inside the kernel point to memory on the GPU. We already copied the host buffers into those device pointers with cudaMemcpy before the launch. Instead of a for loop on the CPU that runs 8 iterations one after another, we have 8 threads on the GPU doing one "iteration" each, all in parallel:

-----

###### Scaling up: Grids, blocks, and boundary checks 25

```
CPU (sequential): GPU (parallel):
i=0: c[0] = a[0] + b[0] (tick 1) Thread 0: c[0] = a[0] + b[0] ─┐
i=1: c[1] = a[1] + b[1] (tick 2) Thread 1: c[1] = a[1] + b[1] │
i=2: c[2] = a[2] + b[2] (tick 3) Thread 2: c[2] = a[2] + b[2] ├─ (tick 1)
i=3: c[3] = a[3] + b[3] (tick 4) Thread 3: c[3] = a[3] + b[3] │
```

| ... i=7: c[7] = a[7] + b[7] (tick 8) Thread 7: c[7] = a[7] + b[7] ─┘ | ... | │ |
|---|---|---|
| Total: 8 time units | Total: 1 time unit |  |

On the left, the CPU tackles each element one after another; on the right, the GPU fires off many lightweight threads so every element is processed at the same time. Keep that side-by-side contrast in mind: it is the mental pivot you need whenever you convert a loop into a CUDA launch. We don't write loops to iterate over data; we launch threads, and each thread handles one piece of the data.

## 2.5 Scaling up: Grids, blocks, and boundary checks

The 8-thread example is great for learning, but what happens when we want to process a million elements? A single block can only hold a limited number of threads (typically 1,024). To go bigger, we need to launch multiple blocks.

CUDA organizes threads into a hierarchy, illustrated in figure 2.2: ¡ *Threads are grouped into blocks.* ¡ *Blocks are organized into a grid.* Think of it like an apartment complex. The grid is the entire complex. Each block is a single floor. Each thread is a single room on that floor. To find a specific room, you need to know its floor number (blockIdx.x) and its room number on that floor (threadIdx.x). Figure 2.2 shows the hierarchy as a 2D grid so you can see multiple blocks at once. We're still working with a 1D row of data in this chapter, but the same indexing principles apply. The specific block and thread numbers in this figure (and in figures 2.3 and 2.4) are there to illustrate the layout. The order in which blocks actually run is scheduled by the hardware and is not guaranteed to be the same across GPUs, so never write a kernel that depends on it. This leads to the universal formula for calculating a unique, global index for any thread in a 1D grid.

**Listing 2.12 1D global thread index calculation**

```
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

###### Calculate global thread ID: (block ID × threads per block) + thread ID within block.

-----

###### 26 Chapter 2 Your first CUDA program

gridDim.x

---

gridDim.y

| BLOCK 0 = GRID[0][0] EEN | BLOCK 1 = GRID[0][1] | BLOCK 2 = GRID[0][2] EERE |
|---|---|---|
| BLOCK 3 = GRID[1][0] | BLOCK 4 = GRID[1][1] ERE | BLOCK 5 = GRID[1][2] ERY |

---

Which block are we jumping to within the entire grid? block\_idx = (blockIdx.y \* gridDim.x) + blockIdx.x

Which thread are we jumping to within this block? thread\_idx = (threadIdx.y \* blockDim.x) + threadIdx.x

###### GRID

blockDim.y

###### BLOCK

| THREAD 0 = BLOCK[0][0] | THREAD 1 = BLOCK[0][1] |
|---|---|
| THREAD 2 = BLOCK[1][0] | THREAD 3 = BLOCK[1][1] |

blockDim.x

**Figure 2.2 CUDA organizes parallel execution into a two-level hierarchy. Threads are the smallest unit of execution and are grouped into blocks. Blocks are then organized into a grid. Each thread has a unique ID within its block (threadIdx), and each block has a unique ID within the grid (blockIdx). Combined with the block size (blockDim), these indices let every thread compute a globally unique position to process.**

```
¡ blockIdx.x: The ID of the current block (the floor number)
¡ blockDim.x: The total number of threads in a block (how many rooms are on
```

each floor)

```
¡ threadIdx.x: The ID of the current thread within its block (the room number)
```

Let's say we want to process n = 1,000,000 elements. We can choose a reasonable block size, like threadsPerBlock = 256. To figure out how many blocks we need, we use the integer math trick in the following listing.

-----

###### Scaling up: Grids, blocks, and boundary checks 27

**Listing 2.13 Ceiling division for block calculation**

###### Choose a reasonable block size. It must be a multiple of the 32-thread warp size and no larger

**than 1024; values from 128 to 1024 are typical.**

```
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
```

###### Integer ceiling division to ensure enough blocks to cover all elements

This is a robust way to calculate ceil(n / threadsPerBlock). It ensures that we have enough blocks to cover all the elements.

Now, our kernel launch looks like this:

```
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(...);
```

And our kernel needs to use the full global index formula. But there's one more catch. If n isn't perfectly divisible by threadsPerBlock, we'll launch more threads than we have elements. For example, if n = 1,000 and threadsPerBlock = 256, we'll launch 4 blocks, for a total of 4 \* 256 = 1024 threads. The last 24 threads would try to access memory outside the valid range. To prevent this, we add a simple boundary check, shown in figure 2.3.

The final, scalable kernel looks like the following listing.

**Listing 2.14 Scalable vector addition kernel with boundary checking**

```
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`if (i < n) {` **Calculates this** `c[i] = a[i] + b[i];` **thread's global index**

```
}
```

`}` **Boundary check to prevent out-of-bounds memory**

###### Performs the work only if this access when thread count exceeds data size thread has valid data to process

This is the canonical pattern for simple CUDA kernels. Each thread calculates its unique global ID, checks whether it's within the bounds of the data, and, if so, does its work. Most real workloads quickly move beyond 1D vectors. When you launch a 2D grid, you upgrade the configuration objects to dim3 and compute indexes in both the x and

```
y dimensions. The dim3 type is simply a struct with three integer fields (x, y, z), default-
```

ing to 1 for any unspecified dimension. So dim3(16, 16) creates a 16 × 16 block, while

```
dim3(256) creates a 1D configuration of 256 threads. The following listing sketches
```

what a 2D configuration looks like.

-----

###### 28 Chapter 2 Your first CUDA program

N

###### Matrix we are processing

M

###### We only care about threads and block up to blockIdx.x = 1 and blockIdx.y = 1 (second blocks on both axes) because

**that's the shape of our matrix.**

---

if (row < M && col < N): proceed....

else do nothing

###### All the threads in the full grid are active, but we haven't given them any tasks so only a chunk of them are executing instructions.

**Figure 2.3 When the total number of elements is not evenly divisible by the block size, CUDA launches extra threads beyond the valid data range. The shaded region shows threads whose computed index exceeds the array bounds. Each thread must check whether its index is valid before accessing memory; threads outside the bounds simply return early without doing any work. This boundary check prevents illegal memory accesses that would otherwise crash the program or corrupt data.**

**Listing 2.15 Sketching a 2D launch configuration**

```
dim3 threadsPerBlock(16, 16);
```

```
dim3 blocksPerGrid(
(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
(height + threadsPerBlock.y - 1) / threadsPerBlock.y
);
```

```
vectorAdd2D<<<blocksPerGrid, threadsPerBlock>>>(...);
```

Inside the kernel, you would compute x = blockIdx.x \* blockDim.x + threadIdx.x and

```
y = blockIdx.y * blockDim.y + threadIdx.y. We'll build those kernels step by step in
```

chapter 3 when we tackle image operations; for now, this preview simply shows how the launch syntax scales.

-----

###### Expanding to 3D arrays 29

## 2.6 Expanding to 3D arrays

Now that we've mastered the 1D indexing pattern (and peeked at how it generalizes to 2D), let's take the final step to three dimensions. A natural example is a single RGB image: if you have a 256 × 256 pixel image with three color channels, you're working with a 3D tensor of shape [3, 256, 256] (channels, height, width). The pattern remains consistent: we extend our thread grid to 3D and use three coordinates (x, y, z) to identify each thread's unique position in 3D space.

### 2.6.1 3D tensor addition: CPU implementation

First, let's establish our ground truth with a CPU version. The code in the following listing uses three nested loops to iterate through every element in the 3D tensor.

**Listing 2.16 CPU implementation of 3D tensor addition**

```
void tensorAdd3D_cpu(
const float* A,
const float* B,
float* C,
int depth,
int height,
```

`int width` **Three nested loops iterate over** `) {` **all elements in the 3D tensor.**

```
for (int d = 0; d < depth; ++d) {
for (int h = 0; h < height; ++h) { Converts 3D coordinates
for (int w = 0; w < width; ++w) { to 1D memory index using
int index = d * (height * width) + row-major layout
h * width + w;
```

`C[index] = A[index] + B[index];` **Performs element-wise addition**

```
}
}
}
```

```
}
```

Note the 3D-to-1D index conversion: d \* (height \* width) + h \* width `+ w. This flattens` our 3D coordinates into a linear memory address, where

```
¡ d * (height * width)-Skips past all the complete 2D "slices" before the current
```

depth

```
¡ h * width-Skips past all the complete rows in our current 2D slice
¡ w-The column position within the current row
```

### 2.6.2 The CUDA kernel for 3D tensor addition

The CUDA kernel in listing 2.17 extends our 2D pattern by adding a third dimension. Each thread calculates its unique 3D coordinate (depth, height, width) and performs one addition.

-----

###### 30 Chapter 2 Your first CUDA program

**Listing 2.17 CUDA kernel for 3D tensor addition**

```
__global__ void tensorAdd3D_kernel(
const float* A,
const float* B,
float* C,
int depth,
int height,
int width) {
int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
int y_dim = blockIdx.y * blockDim.y + threadIdx.y;
int z_dim = blockIdx.z * blockDim.z + threadIdx.z;
if (z_dim < depth && y_dim < height && x_dim
int index = z_dim * (height * width) + y_dim
C[index] = A[index] + B[index];
}
```

`}` **Each thread performs one addition.** Notice the new built-in variables:

```
¡ blockIdx.z and threadIdx.z-The z-coordinate identifiers
¡ blockDim.z-The number of threads in the z-dimension of each block
```

The boundary check now requires three conditions, and our index calculation follows the same 3D-to-1D flattening formula as the CPU version.

### 2.6.3 3D kernel launch configuration

Launching a 3D kernel requires specifying all three dimensions for both the thread blocks and the grid, as you can see in the following listing.

**Listing 2.18 Launching the 3D tensor addition kernel**

```
int depth = 32, height = 128, width = 128;
dim3 threadsPerBlock(8, 8, 8);
dim3 blocksPerGrid(
(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
(height + threadsPerBlock.y - 1) / threadsPerBlock.y,
(depth + threadsPerBlock.z - 1) / threadsPerBlock.z
);
tensorAdd3D_kernel<<<blocksPerGrid, threadsPerBlock>>>(
d_A, d_B, d_C, depth, height, width
);
```

###### Calculates global thread indices in all three dimensions Boundary check for all three dimensions

```
< width) {
* width + x_dim;
```

###### Converts 3D coordinates to 1D memory index

###### Defines 3D block dimensions (8 × 8 × 8 = 512 threads per block) Calculates the number of blocks needed in each dimension using

**ceiling division Launches the kernel with a 3D grid and block configuration**

The dim3 constructor now takes three arguments instead of two. A common choice is

```
(8, 8, 8) threads per block, giving us 512 threads total. The hardware limit is 1,024
```

threads per block, so 512 leaves room for the scheduler. A single streaming multiprocessor (SM) can handle many such blocks concurrently, and modern GPUs have

-----

###### Expanding to 3D arrays 31

dozens to well over a hundred SMs: 82 on an RTX 3090, 132 on an H100 SXM, and 170 on an RTX 5090. Figure 2.4 illustrates the logic.

###### This cube is the grid!

###### Single thread within a thread block

blockDim.y

|  | blockDim.z |
|---|---|
| blockDim.x |  |

###### One block issued to one SM with their own shared memory on-chip (VERY FAST)

gridDim.x gridDim.y

gridDim .z

**Figure 2.4 For 3D data, such as volumetric images or batched 2D tensors, CUDA supports 3D grids of 3D blocks. Each thread identifies its position using six built-in variables: blockIdx in x, y, and z for the block's position in the grid, and threadIdx in x, y, and z for the thread's position within its block. The total number of threads per block (x times y times z) must not exceed 1,024 on most GPUs.**

Think of the 3D grid as a cube of thread blocks, where each block is itself a smaller cube of threads.

### 2.6.4 Real-world example: Batch image processing

Here's a concrete scenario where 3D indexing shines. Imagine you're preprocessing a batch of RGB images for a neural network. Listing 2.19 shows the implementation.

**Listing 2.19 Batch image normalization kernel**

```python
// Host-side launch configuration (these are separate from the kernel's
// parameters of the same name below, which receive their own copies).
int batch_size = 64, channels = 3, height = 128, width = 128;
__global__ void normalizeImageBatch(
const float* input,
float* output,
float mean,
float std,
int batch_size,
int channels,
```

-----

###### 32 Chapter 2 Your first CUDA program

```
int height,
```

`int width` **Each thread handles one** `) {` **spatial position (height,**

```
int w = blockIdx.x * blockDim.x + threadIdx.x; width, channel).
int h = blockIdx.y * blockDim.y + threadIdx.y;
int c = blockIdx.z * blockDim.z + threadIdx.z; Boundary check for all
```

**three spatial dimensions**

```
if (w < width && h < height && c < channels) {
for (int b = 0; b < batch_size; ++b) { Loops over the
int index = b * (channels * height * width) + batch dimension
c * (height * width) + (each thread
h * width + w; normalizes
output[index] = (input[index] - mean) / std; the same position
```

`}` **across all images)** `}` **Normalizes the pixel value using** `}` **the mean and standard deviation**

###### Calculates the 4D index for batch, channel, height, width

This kernel assigns each thread to a unique (height, width, channel) position, and that thread normalizes that same pixel position across all images in the batch. The full tensor is 4D: (batch, channels, height, width). Our thread block covers the spatial and channel dimensions, and each thread loops over the batch dimension to linearize that 4D structure into the 1D memory layout that PyTorch, NumPy, and CUDA all use by default (row-major order). Keep in mind that this per-thread batch loop is a deliberate simplification: it serializes the batch, so a thread performs batch\_size iterations one after another. For a large batch, you would instead map the batch onto a grid dimension (for example, folding batch and channels together into blockIdx.z) so those images are processed in parallel rather than in a loop.

## 2.7 The universal pattern: From nD to 1D

Whether you're working with 1D vectors, 2D matrices, 3D tensors, or even higherdimensional data, the pattern remains the same:

**1** Calculate your thread's unique coordinates using blockIdx and threadIdx.

**2** Apply boundary checks to ensure you don't access invalid memory. **3** Linearize the nD coordinates into a row-major 1D memory index by using stride

calculations (this matches the layout used by C, NumPy, and PyTorch). **4** Perform your computation on the data at that index. That third step is often called "linearizing" or "flattening" your index. Because PyTorch, NumPy, and plain C default to row-major layout, the stride formulas you've seen throughout this chapter follow that convention.

-----

###### Summary 33

###### Summary

¡ A CUDA program consists of two main parts: host code that runs on the CPU and

orchestrates the process and device code (kernels) that runs in massively parallel threads on the GPU. ¡ The terms host and device are the official CUDA terminology for the CPU and

GPU, respectively. They have separate memory spaces. ¡ Data must be explicitly transferred between host (RAM) and device (VRAM)

memory using cudaMemcpy. This transfer is a potential performance bottleneck. ¡ GPU memory is managed with cudaMalloc and cudaFree, analogous to C's malloc

and free for host memory. ¡ A kernel is a C++ function marked with global that is launched from the host and

executed by many threads on the device. ¡ CUDA organizes threads into a two-level hierarchy: threads are grouped into

*blocks, and blocks are organized into a grid.* ¡ Each thread calculates a unique global index using built-in variables (blockIdx,

```
threadIdx, blockDim, and gridDim) to determine which piece of data it is respon-
```

sible for processing. ¡ Essential C concepts like pointers (\*) are fundamental to CUDA, as they are used

to manage memory addresses on both the host and device. ¡ A simple CUDA program can be compiled from the command line using the

CUDA compiler (nvcc).

-----

# 3

## Naive kernels

---

| This chapter covers |
|---|
| ¡ Naive but correct CUDA kernels |
| ¡ dim3 to launch 2D and 3D thread arrays |
| ¡ Implementing naive matrix transpose, GEMM, |
| and softmax |
| ¡ Sliding-window operations: convolutions and |
| pooling |

In the last chapter, we crossed a major threshold. We went from being a passenger on the .to("cuda") express to getting into the driver's seat. This chapter is where we put that knowledge to work. We're moving from the classic "Hello, World" vector addition kernel to the real substance of deep learning. Our goal is to get some serious practice by implementing several of the most important operations in neural networks from scratch. This chapter serves as a road map where we will build naive versions of six key kernels:

¡ *Matrix transpose-A fundamental data-reshaping operation* ¡ *GEMM-The computational heart of every dense layer*

-----

###### From 1D to 3D element-wise addition 35

¡ *Softmax-The essential final activation function for classification* ¡ *1D convolution-The core of processing sequential data like time series* ¡ *2D convolution-The engine of modern computer vision*

¡ *2D max pooling-A critical downsampling and feature-invariance operation*

A quick warning: the key word here is naive. Our goal in this chapter is correctness, not speed. We want to translate the mathematical logic of these operations into parallel CUDA code and prove that they work. Speed comes later, starting in chapter 6. The goal there is a kernel you no longer need to touch, one that is correct, drops in cleanly, and holds up across workloads. A kernel tuned for one workload can slow down when the GPU architecture, the number of SMs it keeps busy, or the input shapes change, so when you put work into a kernel, think about where else it will run. When you're writing parallel code, especially with the complex indexing required for GPUs, bugs are not a matter of "if" but "when." A single off-by-one error in your thread index calculation can lead to silent data corruption, garbage results, or spectacular crashes. Debugging this can be a nightmare. We establish a ground truth. For every GPU kernel we write, we first write a simple, clear, single-threaded CPU version of the same operation. This CPU version is our oracle. It's slow, but it's easy to reason about and trust. After we run our fancy new GPU kernel, our very last step will always be to compare its output, element by element, to the output of our trusted CPU version. If they match, we can proceed with confidence. If not, we know exactly where the problem is. This simple pattern (implement on CPU, implement on GPU, verify) will be our mantra throughout this chapter. To demonstrate this pattern and establish a common format for the rest of our examples, we will begin with a detailed look at 2D matrix addition before proceeding to our six main kernels.

## 3.1 From 1D to 3D element-wise addition

Before tackling complex data-movement patterns like transposes and convolutions, we need to solidify our understanding of the absolute core of CUDA programming: mapping a thread's identity to an element in memory. We'll do this by extending the simple addition operation from 1D to 2D and finally to 3D. Treat figure 3.1 as an abstraction of what we map to a GPU. **NOTE** For the remainder of this chapter, we focus on the CPU and GPU kernel implementations and the kernel launch configuration. The full, repetitive host code (memory allocation, data transfer, etc.) is omitted for brevity but is available in the book's reference GitHub repository. In the previous chapter, we mastered adding two vectors, c[i] = a[i] + b[i]. The strategy was to launch one thread for each element in the output vector. Each thread used its unique 1D identity to find its corresponding element and perform one addition.

-----

###### 36 Chapter 3 Naive kernels

###### 1D

+ C

**2D** +

+

###### 3D Figure 3.1 Element-wise addition stays element-wise as dimensions increase.

**Listing 3.1 1D vector addition kernel**

```
__global__ void vectorAdd_kernel(const float* a, const float* b, float* c,
int num_elements) {
int index = blockIdx.x * blockDim.x + threadIdx.x; Calculates the
if (index < num_elements) { global thread index
c[index] = a[index] + b[index];
}
```

`}` **Each thread performs one Boundary check to prevent**

**element-wise addition. out-of-bounds access**

Awesome sauce! Now we can comfortably step up to 2D. Adding two matrices is conceptually identical to adding two vectors: each element in the output matrix C is the sum of the corresponding elements in matrices A and B. The only new challenge is mapping our threads to a 2D structure instead of a 1D line. Let's stay consistent with best practices and implement the simple CPU version first. This gives us a better intuition of what the kernel requires (we'll parallelize the most deeply nested for loop) and provides a trusted reference to verify our kernel's output. The code in the following listing uses two nested loops to iterate through every element, perform the addition, and store the result.

**Listing 3.2 CPU implementation of 2D matrix addition**

```
void matrixAdd_cpu(const float* A, const float* B, float* C, int num_rows,
int num_cols) {
for (int row = 0; row < num_rows; ++row) { Nested loops iterate over
for (int col = 0; col < num_cols; ++col) { all matrix elements.
int index = row * num_cols + col;
```

`C[index] = A[index] + B[index];` **Converts 2D coordinates to a 1D** `}` **memory index (row-major layout)** `}` **Performs element-wise addition**

```
}
```

-----

###### From 1D to 3D element-wise addition

### 3.1.1 From loops to parallel threads

The two nested for loops in the CPU code are the perfect target for parallelization. Instead of having a single CPU core iterate through every row and column one by one, we can launch a 2D grid of GPU threads. Each thread in this grid is responsible for a single (row, column) pair, effectively executing one iteration of the loop pair in parallel with all other threads.

###### The CUDA kernel: Line-by-line walk-through

Now, let's translate this logic into a CUDA kernel. As shown in listing 3.3, we'll replace the nested loops with a 2D grid of threads. Each thread calculates its unique 2D (row,

```
column) coordinate, flattens it into a 1D memory index, and performs the addition for
```

that single element.

**Listing 3.3 Naive CUDA kernel for 2D matrix addition**

```
__global__ void matrixAdd_kernel(const float* A, const float* B, float* C,
int num_rows, int num_cols) {
int column = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < num_rows && column < num_cols) {
int index = row * num_cols + column;
C[index] = A[index] + B[index];
}
```

`}` **Flattens the 2D (row, column) coordinate**

**into a 1D memory index to access the data, which is stored linearly**

###### Calculates the thread's global column index

###### Calculates the thread's global row index A boundary check ensures that threads outside the matrix dimensions do not

**work. This is critical when matrix dimensions are not perfectly divisible by the block dimensions.**

###### Kernel launch configuration

To execute our kernel, we need to define the grid and block dimensions. As shown in listing 3.4, we configure a 2D grid of thread blocks. The total number of blocks in each dimension is calculated by rounding up the division of the matrix dimension by the block dimension, using the standard C++ idiom for ceiling division: (a + b

```
- 1) / b.
```

To see why this rounding is necessary, consider our 1000 × 500 matrix with 16 × 16 thread blocks. For columns: (500 + 16 - 1) / 16 = 515 / 16 = 32 blocks (integer division). For rows: (1000 + 16 - 1) / 16 = 1015 / 16 = 63 blocks. This gives us a 32 × 63 grid, launching 32 × 16 = 512 threads in the x-dimension and 63 × 16 = 1,008 threads in the y-dimension. We overshoot the matrix dimensions slightly (512 > 500, 1,008 > 1,000), which is why the boundary check in the kernel is essential.

-----

###### 38 Chapter 3 Naive kernels

**Listing 3.4 Launching the 2D matrix addition kernel**

```
int num_rows = 1000;
int num_cols = 500;
```

###### Defines a 2D block with Calculates the

`dim3 threadsPerBlock(16, 16);` **16 × 16 = 256 threads blocks needed**

```
dim3 blocksPerGrid( using ceiling
(num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, division to cover
(num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y all matrix
```

`);` **elements**

```
matrixAdd_kernel<<<blocksPerGrid, threadsPerBlock>>>( Launches the kernel with a 2D
d_A, d_B, d_C, num_rows, num_cols); grid and block configuration
```

### 3.1.2 Kernel 1, matrix transpose: When data needs to change shape

Matrix transposition is a fundamental data reshaping operation. It flips a matrix over its diagonal, so an element at (row, col) moves to (col, row). This might seem simple, but it's essential in deep learning. For instance, when a convolutional neural network (CNN) 2D feature map needs to be flattened and fed into a dense layer, or when attention mechanisms in Transformers need to align query and key matrices for multiplication, transposition is the tool for the job. It allows us to rearrange data to meet the specific dimensional requirements of different network layers. Figure 3.2 shows how a transpose swaps rows and columns.

| A | in memory: | A^T | in memory: |
|---|---|---|---|
| row-major | 1, 2, 3, 4, 5, 6 | col-major | 1, 2, 3, 4, 5, 6 |

**1 2**

= **1 3 5**

**3 4**

**2 4 6**

**5 6**

| A | in memory: | A^T | in memory: |
|---|---|---|---|
| col-major | 1, 3, 5, 2, 4, 6 | row-major | 1, 3, 5, 2, 4, 6 |

**Figure 3.2 Matrix 1 2 transposition swaps**

**1 3 5 rows and columns**

= **across the diagonal.**

**3 4 The arrows show how**

**2 4 6 elements map between**

**5 6** looks the **row-major and column-**

same in memory **major memory layouts.**

Let's explore this through a concrete scenario: building a simple image classification pipeline where we need to transform image data between different representations.

-----

###### From 1D to 3D element-wise addition 39

| Scenario: Preparing image data for a neural network |
|---|
| You're processing a batch of 100 images, each 28 × 28 pixels. Your convolutional |
| layers work with this as a 100 × 28 × 28 tensor, but your final dense layer expects a flattened representation. However, before flattening, you need to apply a learned transformation matrix that operates on the spatial dimensions. This requires trans- posing the spatial dimensions while keeping the batch dimension intact. |
| The challenge: efficiently transpose each 28 × 28 image in your batch while maintain- ing the batch structure. Our 100 × 28 × 28 tensor remains 100 × 28 × 28 after trans- position, but the spatial relationships have been inverted, a critical preprocessing step for certain architectural patterns. |
| Because a 28 × 28 image is square, the output has the same shape as the input; only the data layout changes (row i becomes column i). To see the transpose actu- ally change a matrix's dimensions, look at the nonsquare 3 × 4 example traced in listing 3.6, where a 3 × 4 input becomes a 4 × 3 output. |

###### Understanding the problem with a CPU reference

Let's first build our trusted CPU version. The code in listing 3.5 shows how the indexing changes: where we once read from in[row \* num\_cols + column], we now write to

```
out[column * num_rows + row].
```

| Listing 3.5 CPU implementation of matrix transposition |  |
|---|---|
| void transpose_cpu(const float* in, float* out, int num_rows, | int num_cols) { |
| for (int row = 0; row < num_rows; ++row) { |  |

```
for (int column = 0; column < num_cols; ++column) { Iterates over
out[column * num_rows + row] = in[row * all elements
num_cols + column]; in the matrix
}
```

`}` **Reads from the (row, col) position and writes** `}` **to the (col, row) position to transpose**

###### From sequential loops to a parallel kernel

The transition from CPU loops to GPU threads follows the same pattern we've established. The two nested loops in our CPU code become a 2D grid of threads, where each thread takes responsibility for moving a single element from its original position to its transposed location. The parallel version in the following listing directly implements this pattern.

| Listing 3.6 CUDA kernel for out-of-place matrix transposition |  |  |
|---|---|---|
| __global__ void transpose_kernel(const float* in, float* int num_cols) { | out, | int num_rows, |
| int column = blockIdx.x * blockDim.x + threadIdx.x; |  |  |
|  |  | Calculates this thread's |

```
int row = blockIdx.y * blockDim.y + threadIdx.y; 2D coordinates in the
```

**input matrix**

-----

###### 40 Chapter 3 Naive kernels

```
if (row < num_rows && column < num_cols) {
int in_index = row * num_cols + column;
int out_index = column * num_rows + row;
out[out_index] = in[in_index];
}
```

`}` **Copies an element from the input to**

**the transposed position in the output**

###### Boundary check to ensure valid indices Calculates the 1D index for the source element in the input matrix

###### Calculates the 1D index for

Let's trace through a concrete example. Imagine we're transposing a 3 × 4 matrix. A thread responsible for the input element at (row=1, col=2) will read the value 6 and write it to the output position (col=2, row=1). All threads perform this calculation for their assigned element in parallel:

```
Input matrix (3×4):
[ 0 1 2 3 ]
[ 4 5 6 7 ]
[ 8 9 10 11 ]
Thread (row=1, col=2) processes element at position (1,2):
* Reads from input: in[1 * 4 + 2] = in[6] = 6
* Writes to output: out[2 * 3 + 1] = out[7] = 6
Output matrix (4×3):
[ 0 4 8 ]
[ 1 5 9 ]
[ 2 6 10 ]
[ 3 7 11 ]
```

**the destination element in the output (swaps row/col)**

| Launching the transpose kernel Finally, to launch this kernel for our image batch scenario, block dimensions and loop through each image in the batch, This configuration launches a 2 × 2 grid of 16 × 16 image. The ceiling division ensures we cover all 784 though 28 isn't perfectly divisible by 16. | we configure the grid and as shown in listing 3.7. thread blocks for each 28 × 28 elements of each image, even |
|---|---|
| Listing 3.7 Launching the transpose kernel for a batch of | images |
| int num_rows = 28; int num_cols = 28; 16 × 16 thread dim3 threadsPerBlock(16, 16); (256 threads total) | blocks Calculates the grid dimensions using ceiling division |
| dim3 blocksPerGrid( |  |
| (num_cols + threadsPerBlock.x - 1) / (num_rows + threadsPerBlock.y - 1) / ); | threadsPerBlock.x, threadsPerBlock.y Loops over the batch to transpose each image |
| for (int batch = 0; batch < 100; ++batch) { |  |

-----

###### Core neural network kernels

```
float* d_in = device_input + batch * 28 * 28;
float* d_out = device_output + batch
transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
d_in, d_out, num_rows, num_cols
);
}
```

###### Pointer

`* 28 * 28;` **arithmetic to**

**access each image in the batch Launches the kernel for the current image in the batch**

## 3.2 Core neural network kernels

Now we will move from our recap to some fundamental use cases in deep learning. We will consider GEMM and softmax. Not only are these basic operations in neural networks, but they're also great examples of parallelization with GPUs.

### 3.2.1 Kernel 2, GEMM: The heart of neural networks

If there is one operation that defines modern deep learning, it's GEMM (C = A \* B). It is the computational heart of dense (fully connected) layers in MLPs, the attention mechanism in Transformers, and even parts of convolutional layers. For our purposes, we launch a 2D grid of threads matching the dimensions of the output matrix

```
C. Each thread is responsible for computing a single element of C by performing a
```

dot product.

###### GEMM CPU implementation

As is our practice, we start with the CPU implementation in listing 3.8 to establish our ground truth. It uses a triple-nested loop. The outer two loops iterate over each element of the output matrix C, while the innermost loop performs the dot product between a row of A and a column of B.

The naming convention here reflects matrix multiplication dimensions: if `A is` M × K and B is K × N, then C is M × N. The K\_shared\_dim parameter is the dimension that

```
A and B share: it's the number of columns in A and the number of rows in B. This shared
```

dimension determines how many multiply-add operations each dot product requires.

**Listing 3.8 CPU implementation of GEMM**

```
void gemm_cpu(
const float* A, const float* B,
```

`float* C, int M_rows, int N_cols,` **The outer loops iterate over the** `int K_shared_dim` **output matrix C dimensions.**

```
) {
```

###### Accumulator

```
for (int row = 0; row < M_rows; ++row) { for dot product
for (int column = 0; column < N_cols; ++column) { computation
float sum = 0.0f;
for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
sum += A[row * K_shared_dim + k_idx]
* B[k_idx * N_cols + column]; The inner loop
```

`}` **computes the dot**

**product of A's row with B's column.**

-----

###### 42 Chapter 3 Naive kernels

| C[row * N_cols + column] = sum; } } Stores the computed dot } product in output matrix C tion of each output element is completely independent of the others. The CUDA kernel for GEMM pute the dot product and writes the final sum to the output matrix C. | * N_cols + column] = sum; Stores the computed dot product in output matrix C tion of each output element is completely independent of the others. pute the dot product and writes the final sum to the output matrix C. | Stores the computed dot product in output matrix C tion of each output element is completely independent of the others. pute the dot product and writes the final sum to the output matrix C. | tion of each output element is completely independent of the others. pute the dot product and writes the final sum to the output matrix C. | tion of each output element is completely independent of the others. pute the dot product and writes the final sum to the output matrix C. |
|---|---|---|---|---|
| Listing 3.9 Naive CUDA kernel for GEMM |  |  |  |  |
| __global__ void gemm_kernel(const float* A, int M_rows, int N_cols, int K_shared_dim) | const { | output float* | Each element B, | thread computes one at (row, column). float* C, |
| int row = blockIdx.y * blockDim.y | + | threadIdx.y; |  |  |
| int column = blockIdx.x * blockDim.x | + | threadIdx.x; | matrix | dimensions |
| if (row < M_rows && column < N_cols) { |  |  |  |  |
| float sum = 0.0f; for (int k_idx = 0; k_idx < sum += A[row * K_shared_dim + | K_shared_dim; k_idx] | ++k_idx) * B[k_idx | { * | Loops over shared |
| N_cols + column]; |  |  |  |  |
|  |  |  |  | dimension K |
| } |  | Multiplies | the | to compute |
| C[row * N_cols + column] = sum; |  | corresponding |  |  |
|  |  |  |  | the dot |
| } } Writes the computed product to output GEMM kernel launch configuration configure the grid and block dimensions for this | dot matrix C task. | elements row and | from B's | A's product column |
| Listing 3.10 Launching the naive GEMM | kernel |  |  |  |
| int M_rows = 512, N_cols = 512, K_shared_dim dim3 threadsPerBlock(16, 16); | = | 256; 16 for | × 16 the | thread blocks output matrix |
| dim3 blocksPerGrid( |  |  |  |  |

```
(N_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, Sizes the grid to cover
(M_rows + threadsPerBlock.y - 1) / threadsPerBlock.y the entire output
```

`);` **matrix (M × N)**

-----

###### Core neural network kernels 43

```
gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M_rows,
N_cols, K_shared_dim);
```

###### Launches the kernel to compute C = A × B

### 3.2.2 Kernel 3, softmax: The final activation

Softmax is a normalization method used across neural networks, from the final layer of classifiers to attention mechanisms in large language models. Its purpose is to take a vector of arbitrary real-valued scores (called logits) and transform them into a probability distribution, where each element is between 0 and 1 and all elements sum to 1. This naive version is highly inefficient because it involves redundant calculations, but it's an excellent way to understand the data-dependent nature of reduction operations on a GPU. The operation proceeds in four steps, as illustrated by this numerical example for a vector z = [2.0, 1.0, 0.1, -1.0]: **1** Find the max for numerical stability: max\_val = 2.0. We subtract this from all logits to prevent exp() from overflowing with large inputs. **2** Compute exponentials: exp(z - max\_val) results in [1.0, 0.3679, 0.1496, 0.0498]. **3** Sum exponentials: sum\_exp = 1.0 + 0.3679 + … = 1.5673. **4** Normalize: Divide each exponentiated value by the sum to get the final probabilities: [0.638, 0.235, 0.095, 0.032].

###### Softmax CPU implementation

The CPU code in listing 3.11 follows these steps exactly. For each row in a batch, it performs three passes: one to find the max, one to calculate the sum, and a final one to normalize and write the output.

**Listing 3.11 CPU implementation of the softmax function**

```
void softmax_cpu(const float* in, float* out, int num_rows, int num_cols) {
for (int row = 0; row < num_rows; ++row) {
```

###### Processes each row

```
float max_val = in[row * num_cols]; independently
for (int col = 1; col < num_cols; ++col) {
if (in[row * num_cols + col] > max_val) { First pass: finds the
max_val = in[row * num_cols + col]; maximum value for
```

`}` **numerical stability**

```
}
```

###### Second pass: computes

`float sum_exp = 0.0f;` **the sum of exponentials**

```
for (int col = 0; col < num_cols; ++col) {
sum_exp += expf(in[row * num_cols + col] - max_val);
}
for (int col = 0; col < num_cols; ++col) {
out[row * num_cols + col] = expf(in[row * num_cols + col]
- max_val) / sum_exp;
```

`}` **Third pass: normalizes**

`}` **to produce probabilities**

```
}
```

-----

###### 44 Chapter 3 Naive kernels

###### The naive CUDA kernel for softmax

Our naive CUDA kernel, shown in listing 3.12, parallelizes this process by assigning one thread to each output element. The major inefficiency is that, to calculate its single output value, each thread must first read its entire row twice: once to find the maximum value and again to find the sum of exponentials. This means all threads working on the same row perform the same redundant reads and calculations. We will fix this with more advanced techniques later.

| Listing 3.12 Naive CUDA kernel for softmax |  |
|---|---|
| __global__ void softmax_naive_kernel(const float* int num_rows, int num_cols) { int row = blockIdx.y * blockDim.y + int column = blockIdx.x * blockDim.x + if (row < num_rows && column < num_cols) float max_val = -1e20f; | in, threadIdx.y; threadIdx.x; |
| for (int col_idx = 0; col_idx < | num_cols; |
| if (in[row * num_cols + col_idx] num_cols + col_idx]; } float sum_exp = 0.0f; |  |
| for (int col_idx = 0; col_idx < | num_cols; |
| sum_exp += expf(in[row * num_cols } out[row * num_cols + column] = expf(in[row | * num_cols |
| max_val) / sum_exp; |  |
| } } |  |
|  |  |

###### Softmax kernel launch configuration

The launch configuration in listing 3.13 is standard: we create a 2D grid of threads that covers the entire input/output matrix.

**Listing 3.13 Launching the naive softmax kernel**

```
int num_rows = 128, num_cols = 1000;
dim3 threadsPerBlock(16, 16);
dim3 blocksPerGrid(
(num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
(num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
);
softmax_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in,
num_rows, num_cols);
```

```
float* out,
```

`{` **find the maximum value.**

```
++col_idx) {
> max_val) max_val = in[row *
```

###### Second reduction: every thread in that same row reads the entire row again to calculate the sum of

**exponentials. This is also highly redundant.**

```
++col_idx) {
+ col_idx] - max_val);
```

###### Finally, each thread uses the max_val and sum_exp it computed to calculate and write its single output value.

###### 16 × 16 thread blocks (one thread per output element)

###### Launches the naive softmax kernel (each thread computes one output)

###### First reduction: every thread in a given row reads the entire row to This is highly redundant.

```
+ column] -
```

###### The grid covers the entire input/output matrix.

```
d_out,
```

-----

###### Convolution and pooling toolkit 45

## 3.3 Convolution and pooling toolkit

Now we'll look at a few more computational tricks that are fundamental to various neural network operations. 1D and 2D convolution put the "convolutional" in convolutional neural networks, and 2D max pooling is an essential downsampling operation that makes training networks more efficient and their output less prone to overfitting on training data.

### 3.3.1 Kernel 4, 1D convolution: Processing signals and sequences

1D convolution is a cornerstone of signal processing and is used in deep learning to process sequential data such as time series or text. The operation works by sliding a small filter (the kernel) across an input signal. At each position, it computes a dot product between the filter and the segment of the input it currently overlaps. This process extracts features, such as patterns or trends, from the local neighborhood of each point in the signal. **NOTE** Strictly speaking, the operation we implement here (and in the 2D case that follows) is cross-correlation, not the mathematical convolution from signal processing, because we do not flip the filter before applying it. Deep learning frameworks such as PyTorch and TensorFlow do the same thing and still call it "convolution," so we follow that convention throughout the book. Since the filter weights are learned, the missing flip makes no difference to what the network can represent.

###### 1D convolution CPU implementation

The CPU implementation in listing 3.14 makes this sliding window operation clear. The outer loop iterates over each position in the output signal, and the inner loop performs the dot product with the kernel.

**Listing 3.14 CPU implementation of 1D convolution**

```
void conv1d_cpu(const float* in, float* out, const float* kernel,
int input_size, int kernel_size) {
int output_size = input_size - kernel_size + 1; Calculates the output
for (int i = 0; i < output_size; ++i) { size (input shrinks by
```

`float sum = 0.0f;` **kernel\_size -1).**

```
for (int j = 0; j < kernel_size; ++j) {
sum += in[i + j] * kernel[j]; Iterates over each
```

`}` **output position** `out[i] = sum;` **Stores the computed** `}` **The inner loop performs a**

**convolution result dot product with the kernel**

```
}
```

**at the current position.**

###### CUDA kernel for 1D convolution

In our parallel version, shown in listing 3.15, we assign one thread to each element of the output signal. That thread is then responsible for performing the entire dot product required to compute its single output value.

-----

###### 46 Chapter 3 Naive kernels

| Listing 3.15 Naive CUDA kernel for 1D | convolution |
|---|---|
| __global__ void conv1d_kernel(const float* const float* kernel, int input_size, int kernel_size) int output_idx = blockIdx.x * blockDim.x int output_size = input_size - kernel_size | in, float* out, { + threadIdx.x; Each thread + 1; computes one Boundary check output element. |
| if (output_idx < output_size) { |  |
| float sum = 0.0f; for (int k_idx = 0; k_idx < kernel_size; sum += in[output_idx + k_idx] * } out[output_idx] = sum; Writes } convolution } 1D Convolution kernel launch configuration We launch a 1D grid of threads with enough threads output array. | for output size ++k_idx) { Performs a dot kernel[k_idx]; product with the kernel at the computed this position result to compute every element in the |
| Listing 3.16 Launching the 1D convolution | kernel |
| int input_size = 100000, kernel_size = 32; int output_size = input_size - kernel_size + dim3 threadsPerBlock(256); dim3 blocksPerGrid((output_size + threadsPerBlock.x / threadsPerBlock.x); Calculates conv1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, | 1; Uses 256 threads per block for the 1D layout - 1) the blocks needed to cover the output d_out, d_kernel, |
| input_size, kernel_size); |  |

###### Launches the kernel with the 1D grid

### 3.3.2 Kernel 5, 2D convolution: The engine of computer vision

2D convolution is the fundamental operation of CNNs. It works just like its 1D cousin but operates on 2D data, such as images. A small 2D filter (e.g., 3 × 3 or 5 × 5) slides over the input image. At each position, the output pixel is the dot product of the filter and the patch of the image it covers. This allows the network to learn to detect simple features such as edges, corners, and textures, which are then combined by deeper layers into more complex object representations. Figure 3.3 illustrates how different filters can detect different features in an image.

###### 2D Convolution CPU implementation

The CPU code in listing 3.17 uses four nested loops. The outer two iterate over each pixel of the output feature map, while the inner two iterate over the 2D kernel to perform the dot product.

-----

###### Convolution and pooling toolkit 47

| Listing 3.17 CPU implementation of 2D convolution |  |
|---|---|
| void conv2d_cpu(const float* in, float* out, const float* kernel, int height, int width, int kernel_dim) { int output_h = height - kernel_dim + 1; Calculates the output int output_w = width - kernel_dim + 1; by kernel_dim -1 in each for (int r = 0; r < output_h; ++r) { for (int c = 0; c < output_w; ++c) { The outer loops iterate float sum = 0.0f; each output pixel | dimensions (shrinks direction) over position. |
| for (int kr = 0; kr < kernel_dim; ++kr) { |  |
| for (int kc = 0; kc < kernel_dim; ++kc) { | The inner loops iterate over |
| int input_row = r + kr; |  |
| int input_col = c + kc; sum += in[input_row * width + input_col] * kernel[kr * kernel_dim + kc]; } } Multiplies and out[r * output_w + c] = sum; accumulates the input } } Stores the computed convolution pixel with the } result for this output position kernel weight The CUDA kernel for 2D convolution As before, we parallelize the two outer loops. In the CUDA kernel shown in each thread is assigned an (output_row, output_col) coordinate and is computing the value of that single output pixel by performing the two inner the kernel. Figure 3.4 shows the sliding-window operation in detail. Listing 3.18 translates this sliding-window operation into CUDA code. computes one output pixel by iterating over the kernel dimensions and the dot product. | the kernel dimensions. Calculates the input position for the current kernel element listing 3.18, responsible for loops over Each thread accumulating |
| Listing 3.18 Naive CUDA kernel for 2D convolution |  |
| __global__ void conv2d_kernel(const float* in, float* out, const kernel, int height, int width, int kernel_dim) { int output_col = blockIdx.x * blockDim.x + threadIdx.x; Calculates int output_row = blockIdx.y * blockDim.y + threadIdx.y; input on the int output_h = height - kernel_dim + 1; of the int output_w = width - kernel_dim + 1; (output_row, and the if (output_row < output_h && output_col < output_w) { the float sum = 0.0f; for (int kernel_row = 0; kernel_row < kernel_dim; for (int kernel_col = 0; kernel_col < kernel_dim; | float* the current coordinate based top-left corner sliding window output_col) position within kernel (kernel_row, kernel_col) ++kernel_row) { ++kernel_col) { |
| int input_row = output_row + kernel_row; |  |
| int input_col = output_col + kernel_col; sum += in[input_row * width + input_col] * kernel[ |  |
| kernel_row * kernel_dim + kernel_col]; |  |

```
}
```

`}` **Performs one multiply-add operation, accumulating the**

**dot product between the input patch and the kernel**

-----

```
}
```

###### Chapter 3 Naive kernels

```
out[output_row * output_w + output_col] = sum;
}
```

**stride=1, padding=0, input\_channels=1, output\_channels=1, dilation=1, bias=False**

tid = 0 tid = 1 1\*1 + 2\*0 + 4\*0 + 5\*-1 2\*1 + 3\*0 + 5\*0 + 6\*-1 1 + 0 + 0 - 5 = -4 H 2 + 0 + 0 - 6 = -4

1

0

0

-1

| 1 | 2 | 3 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| 4 | 5 | 6 | 4 | 5 | 6 |
| 7 | 8 | 9 | 7 | 8 | 9 |

-4 -4 1 2 3 -4 -4

| 4 | 5 | 6 | 4 + 0 + 0 - | 4*1 + 5*0 + 7*0 + 8*-1 8 = | -4 | 5 + 0 + 0 - 9 | 5*1 + 6*0 + 8*0 + 9*-1 = | = -4 |
|---|---|---|---|---|---|---|---|---|
| 7 | 8 | 9 | 1 4 | 2 5 | 3 6 | 1 4 | 2 5 | 3 6 |

| J | 7 | 8 tid = 2 | 9 | 7 | 8 tid = 3 | 9 |
|---|---|---|---|---|---|---|

**Figure 3.4 The 2D convolution filter slides across the input image, computing a dot product at each position. Each output pixel corresponds to one filter position over the input.**

###### 2D convolution kernel launch configuration

We launch a 2D grid of threads sized to cover the entire 2D output feature map.

| Listing 3.19 Launching the 2D convolution kernel |  |  |
|---|---|---|
| int height = 256, width = 256, kernel_dim = 3; |  |  |
| int output_h = height - kernel_dim + 1; |  |  |
|  | Calculates | the output |

```
int output_w = width - kernel_dim + 1; feature map dimensions
```

`dim3 threadsPerBlock(16, 16);` **16 × 16 thread blocks for 2D output** `dim3 blocksPerGrid(` **Sizes the grid to cover the**

```
(output_w + threadsPerBlock.x - 1) / threadsPerBlock.x, entire output feature map
(output_h + threadsPerBlock.y - 1) / threadsPerBlock.y
);
```

```
conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, d_kernel,
height, width, kernel_dim);
```

###### Launches the 2D convolution kernel

-----

###### Convolution and pooling toolkit 49

### 3.3.3 Kernel 6, 2D max pooling: Downsampling and feature detection

Max pooling is a downsampling operation commonly used in CNNs, typically after a convolutional layer. Its purpose is twofold: it reduces the spatial dimensions (height and width) of the feature maps, which makes the network more computationally efficient, and it provides a degree of translation invariance, meaning the network becomes less sensitive to the exact location of a feature in the image. It works by sliding a window (e.g., 2 × 2) over the input and, for each window, outputting only the maximum value. Figure 3.5 illustrates this downsampling process with a stride of 1, so consecutive windows overlap and a 3 × 3 input produces a 2 × 2 output; one thread computes each output value. Note that the implementation we write next uses the more common configuration, where the stride equals the window size, so the windows tile the input without overlapping and each dimension shrinks by the pool factor.

same idea for Avgpool2D...

**stride=1, padding=0, dilation=1**

tid = 0 tid = 1 max(1, 2, 4, 5) = 5 max(2, 3, 5, 6) = 6

no filter, just

| 1 | 2 | 3 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| 4 | 5 | 6 | 4 | 5 | 6 |

find the max

7 8 9 7 8 9

5 6 1 2 3 8 9

max(4, 5, 7, 8) = 8 J max(5, 6, 8, 9) = 9 H

| 4 | 5 | 6 |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| 7 | 8 | 9 | 1 | 2 | 3 | 1 | 2 | 3 |
|  |  |  | 4 | 5 | 6 | 4 | 5 | 6 |
|  |  |  | 7 | 8 | 9 | 7 | 8 | 9 |
|  |  |  |  | tid = 2 |  |  | tid = 3 |  |

**Figure 3.5 Max pooling slides a window over the input and outputs the maximum value from each window position, reducing spatial dimensions while preserving the strongest activations.**

-----

###### 50 Chapter 3 Naive kernels

###### 2D Max pooling CPU implementation

The CPU code in listing 3.20 is very similar to 2D convolution. The outer loops iterate over the output pixels, and the inner loops scan the pooling window in the input image, but instead of a dot product, they simply find the maximum value.

**Listing 3.20 CPU implementation of 2D max pooling The output size is the input size divided by the pool dimension.**

```
void maxpool2d_cpu(const float* in, float* out, int height, int width,
```

`int pool_dim) {` **The outer loops iterate**

```
int output_h = height / pool_dim; over the output pixels.
int output_w = width / pool_dim;
for (int r = 0; r < output_h; ++r) { Initializes the max value
for (int c = 0; c < output_w; ++c) { to a very small number
float max_val = -1e20f;
for (int pr = 0; pr < pool_dim; ++pr) { The inner loops scan
for (int pc = 0; pc < pool_dim; ++pc) { the pooling window.
int input_row = r * pool_dim + pr; Calculates
int input_col = c * pool_dim + pc; the input
float val = in[input_row * width + input_col]; position for
if (val > max_val) max_val = val; the current
```

`}` **element in** `}` **the pool**

```
out[r * output_w + c] = max_val;
```

`}` **Tracks the maximum**

| } | Stores the maximum | value in the pooling |
|---|---|---|
| } | value as an output pixel | window |

###### The CUDA kernel for 2D max pooling

The parallelization strategy, shown in listing 3.21, is identical to that for 2D convolution. We assign one thread to each output pixel, and that thread scans the corresponding input window to find its maximum value.

**Listing 3.21 Naive CUDA kernel for 2D max pooling**

```
__global__ void maxpool2d_kernel(const float* in, float* out, int height,
int width, int pool_dim) {
int output_col = blockIdx.x * blockDim.x + threadIdx.x;
int output_row = blockIdx.y * blockDim.y + threadIdx.y;
int output_h = height / pool_dim;
int output_w = width / pool_dim;
```

```
if (output_row < output_h && output_col < output_w) {
float max_val = -1e20f;
for (int pool_row = 0; pool_row < pool_dim; ++pool_row) {
for (int pool_col = 0; pool_col < pool_dim; ++pool_col) {
int input_row = output_row * pool_dim + pool_row;
int input_col = output_col * pool_dim + pool_col;
```

-----

###### Convolution and pooling toolkit 51

```
float val = in[input_row * width + input_col];
if (val > max_val) max_val = val;
}
```

```
}
out[output_row * output_w + output_col] = max_val;
}
}
```

###### 2D max pooling kernel launch configuration

Finally, we launch a 2D grid of threads sized to cover the smaller, downsampled output feature map.

**Listing 3.22 Launching the 2D max pooling kernel**

```
int height = 256, width = 256, pool_dim = 2; Calculates the downsampled
int output_h = height / pool_dim; output dimensions
int output_w = width / pool_dim;
```

###### 16 × 16 thread blocks for

```
dim3 threadsPerBlock(16, 16); the downsampled output
dim3 blocksPerGrid(
(output_w + threadsPerBlock.x - 1) / threadsPerBlock.x, The grid covers
(output_h + threadsPerBlock.y - 1) / threadsPerBlock.y the smaller output
```

`);` **feature map.**

```
maxpool2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, height,
width, pool_dim);
```

###### Launches the max pooling kernel

And with that, let's pause for a moment to take a breath. We've just implemented the fundamental building blocks for a huge variety of neural networks, entirely from scratch in CUDA. More importantly, we have dissected the single most important part of writing correct kernels: mapping a thread's abstract ID to a concrete memory location. Calculating a global index and using it to access data is the foundation of all GPU programming.

These kernels are indeed reference implementations: they produce the same numerical output as their optimized counterparts, but they run tens to hundreds of times slower than production libraries like cuDNN and cuBLAS. Still, they embody the exact logic at the core of real neural networks. The gemm\_kernel we wrote performs the same computation as a Linear layer. Our softmax\_kernel normalizes logits the same way a classifier does. And the conv2d\_kernel and maxpool2d\_kernel match the operations inside CNNs, which we will continue investigating in this book. In the next chapter, we will put these pieces together. We will take on the classic MNIST handwritten digit classification challenge. Our mission will be to build and train a complete Multilayer Perceptron (MLP) from the ground up. To truly understand the system, we will build it at four distinct levels of abstraction: starting with basic PyTorch,

-----

###### 52 Chapter 3 Naive kernels

then moving to NumPy, pure C++, and finally culminating in our own training loop powered by the CUDA kernels we've just created.

The kernels we built are simple, but they are correct. Our goal in the next chapter is not to break speed records but to prove that our low-level components can be assembled into a functioning, learning system.

###### Summary

¡ Every kernel starts with a simple, slow CPU version that serves as the gold standard. Only after the GPU output matches the CPU output element by element do we move on to performance tuning. ¡ Transpose, GEMM, softmax, convolution, and pooling can all be written in 10

to 25 lines of CUDA. These kernels are tens to hundreds of times slower than cuDNN and cuBLAS, but they are complete, understandable, and debuggable. ¡ A transpose implemented as out[col \* rows + row] = in[row \* cols + col] works

but immediately exposes a performance cliff: the reads from in are coalesced (consecutive threads touch consecutive addresses), while the writes to out are strided and therefore noncoalesced. The standard fix (staging a tile through shared memory with padding so both the reads and writes are coalesced) is left as an exercise here; we spend our optimization budget in chapter 6 on the kernels most central to transformer inference. ¡ Reductions are data dependent. Softmax requires two passes over each row (max

and sum), which naively leads to every thread in the row recomputing the same values. This redundancy is exactly what we eliminate in chapter 6, where a blockper-row strategy and the online-softmax algorithm turn this naive version into a heavily optimized one. ¡ 1D, 2D, and 3D convolutions share the same loop structure: an outer loop over

output positions and an inner loop (or loops) over the filter. Pooling is identical, except the inner loop performs max instead of multiply-accumulate. ¡ Once you can launch a grid of threads and compute the correct index, you can

express any per-element, per-row, or per-window operation found in modern deep learning frameworks.

-----

# 4

## Training and verifying neural nets in raw CUDA

---

| This chapter covers |
|---|
| ¡ The challenge of building and verifying a neural network from scratch for reproducibility |
| ¡ Establishing a "gold standard" implementation |
| in PyTorch |
| ¡ Using NumPy and C to peel back layers of abstraction |
| ¡ Building a naive CUDA MLP from our C blueprint |
| ¡ Optimizing the CUDA implementation with the |
| cuBLAS library |
| ¡ Techniques for debugging and profiling low-level |
| GPU code |

## 4.1 Level 1: The PyTorch approach

In our journey from high-level frameworks to bare-metal CUDA, we need a reliable point of reference: a "gold standard" against which we can verify every subsequent implementation. For this, we turn to PyTorch. Its combination of a high-level API, automatic differentiation, and highly optimized, battle-tested backend libraries makes it the perfect candidate for our trusted implementation.

-----

###### 54 Chapter 4 Training and verifying neural nets in raw CUDA

The goal in this section is to build a PyTorch model in a way that is reproducible and *comparable to our future low-level versions. This PyTorch implementation will be our* benchmark for correctness (does it converge to a similar loss curve?) and performance. One caveat up front: each version in this chapter initializes its weights with a different random number generator (PyTorch, then NumPy, then C's rand()), so the exact per-epoch loss values will not match across implementations. The C and CUDA versions even seed from the clock, so they change from run to run. What we are checking is that every version trains, drives the loss down at a comparable rate, and reaches a similar final accuracy, not that they print identical numbers. The full code for this section is in the v1.py script. Before we start our PyTorch implementation, let's introduce the problem we'll be solving. Our task is to classify handwritten digits using the famous MNIST dataset, often considered the "Hello, World" of computer

| vision, as shown in figure 4.1. The dataset contains thousands of | label: 5 | label: 0 | label: 4 | label: 1 |
|---|---|---|---|---|
| 28 × 28 pixel grayscale images of digits from 0 to 9. Our goal is simple: build and |  |  | L |  |

optimize models that can look at one of these images and correctly identify the **Figure 4.1 MNIST dataset example images:**

digit it represents. **28 × 28 grayscale handwritten digits**

### 4.1.1 PyTorch implementation: Setup, training, and benchmarking

First, we import the necessary libraries, define the hyperparameters, and load the dataset from preprocessed binary files. These binary files (X\_train.bin, y\_train. bin, etc.) are generated by the v0.py script, which is available in the book's source code repository. This script handles downloading, normalizing, and converting the MNIST dataset, providing a common data foundation for all our implementations. We will load this data directly into GPU memory, bypassing the typical DataLoader pipeline.

**Listing 4.1 PyTorch setup and data loading**

```
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
```

###### Seeds the random number generator

`torch.manual_seed(1)` **for reproducible weight initialization**

```
TRAIN_SIZE = 10000
epochs = 10
learning_rate = 1e-2
```

`batch_size = 8` **Enables TensorFloat-32 (TF32)**

**for improved performance on**

```
torch.set_float32_matmul_precision("high") compatible GPUs
```

-----

###### Level1: The PyTorch approach 55

```
X_train_np = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(
60000, 784)
y_train_np = np.fromfile("data/y_train.bin", dtype=np.int32)
X_test_np = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(
10000, 784)
y_test_np = np.fromfile("data/y_test.bin", dtype=np.int32)
```

| mean, std = 0.1307, 0.3081 X_train_np = (X_train_np - mean) / std X_test_np = (X_test_np - mean) / std train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape( -1, 1, 28, 28)).to("cuda") train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda") test_data = torch.from_numpy(X_test_np.reshape(-1, 1, 28, test_labels = torch.from_numpy(y_test_np).long().to("cuda") The MLP architecture is defined in a standard PyTorch nn.Module. A critical step in our verification process is to override PyTorch's default weight initialization with a man- ual Kaiming He initialization (also called He initialization). This weight initialization scheme, named after researcher Kaiming He, helps prevent vanishing and exploding gradients during training by scaling initial weights based on layer size. Using the same initialization scheme in all our implementations ensures that the PyTorch, NumPy, C, and CUDA versions all start from the same weight distribution. The actual weight val- ues differ because each language draws from a different random number generator, but the loss curves should closely track one another, which is what we need for compar- ison. The model definition and initialization are shown in the following listing. | train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape( train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda") = torch.from_numpy(X_test_np.reshape(-1, 1, 28, test_labels = torch.from_numpy(y_test_np).long().to("cuda") The MLP architecture is defined in a standard PyTorch nn.Module. A critical step in our verification process is to override PyTorch's default weight initialization with a man- ual Kaiming He initialization (also called He initialization). This weight initialization scheme, named after researcher Kaiming He, helps prevent vanishing and exploding gradients during training by scaling initial weights based on layer size. Using the same initialization scheme in all our implementations ensures that the PyTorch, NumPy, C, and CUDA versions all start from the same weight distribution. The actual weight val- ues differ because each language draws from a different random number generator, but the loss curves should closely track one another, which is what we need for compar- ison. The model definition and initialization are shown in the following listing. | Applies standard MNIST normalization constants for consistency train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape( train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda") = torch.from_numpy(X_test_np.reshape(-1, 1, 28, test_labels = torch.from_numpy(y_test_np).long().to("cuda") The MLP architecture is defined in a standard PyTorch nn.Module. A critical step in our verification process is to override PyTorch's default weight initialization with a man- ual Kaiming He initialization (also called He initialization). This weight initialization scheme, named after researcher Kaiming He, helps prevent vanishing and exploding gradients during training by scaling initial weights based on layer size. Using the same initialization scheme in all our implementations ensures that the PyTorch, NumPy, C, and CUDA versions all start from the same weight distribution. The actual weight val- ues differ because each language draws from a different random number generator, but the loss curves should closely track one another, which is what we need for compar- ison. The model definition and initialization are shown in the following listing. | Applies standard MNIST normalization constants for consistency train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape( train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda") = torch.from_numpy(X_test_np.reshape(-1, 1, 28, test_labels = torch.from_numpy(y_test_np).long().to("cuda") Converts NumPy arrays to PyTorch tensors and moves them to the GPU immediately The MLP architecture is defined in a standard PyTorch nn.Module. A critical step in our verification process is to override PyTorch's default weight initialization with a man- ual Kaiming He initialization (also called He initialization). This weight initialization scheme, named after researcher Kaiming He, helps prevent vanishing and exploding gradients during training by scaling initial weights based on layer size. Using the same initialization scheme in all our implementations ensures that the PyTorch, NumPy, C, and CUDA versions all start from the same weight distribution. The actual weight val- ues differ because each language draws from a different random number generator, but the loss curves should closely track one another, which is what we need for compar- ison. The model definition and initialization are shown in the following listing. | Applies standard MNIST normalization constants for consistency train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape( train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda") = torch.from_numpy(X_test_np.reshape(-1, 1, 28, 28)).to("cuda") test_labels = torch.from_numpy(y_test_np).long().to("cuda") Converts NumPy arrays to PyTorch tensors and moves them to the GPU immediately The MLP architecture is defined in a standard PyTorch nn.Module. A critical step in our verification process is to override PyTorch's default weight initialization with a man- ual Kaiming He initialization (also called He initialization). This weight initialization scheme, named after researcher Kaiming He, helps prevent vanishing and exploding gradients during training by scaling initial weights based on layer size. Using the same initialization scheme in all our implementations ensures that the PyTorch, NumPy, C, and CUDA versions all start from the same weight distribution. The actual weight val- ues differ because each language draws from a different random number generator, but the loss curves should closely track one another, which is what we need for compar- ison. The model definition and initialization are shown in the following listing. | train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda") 28)).to("cuda") Converts NumPy arrays to PyTorch tensors and moves them to the GPU immediately The MLP architecture is defined in a standard PyTorch nn.Module. A critical step in our verification process is to override PyTorch's default weight initialization with a man- ual Kaiming He initialization (also called He initialization). This weight initialization scheme, named after researcher Kaiming He, helps prevent vanishing and exploding gradients during training by scaling initial weights based on layer size. Using the same initialization scheme in all our implementations ensures that the PyTorch, NumPy, C, and CUDA versions all start from the same weight distribution. The actual weight val- ues differ because each language draws from a different random number generator, but the loss curves should closely track one another, which is what we need for compar- |
|---|---|---|---|---|---|
| Listing 4.2 MLP with custom He | initialization |  |  |  |  |
| class MLP(nn.Module): def __init__(self, in_features, super(MLP, self).__init__() self.fc1 = self.relu = nn.ReLU() self.fc2 = nn.Linear(hidden_features, def forward(self, x): | nn.Linear(in_features, | hidden_features, |  | num_classes): hidden_features) num_classes) |  |
| x = x.reshape(batch_size, 28 * | 28) |  |  |  |  |
| x = self.fc1(x) x = self.relu(x) x = self.fc2(x) return x model = MLP(in_features=784, | hidden_features=256, |  |  | Flattens the 28 × 28 image into a 784-element vector Disables gradient to manually initialize num_classes=10).to("cuda") | calculations weights |
| with torch.no_grad(): |  |  |  |  |  |
| fan_in_fc1 = | model.fc1.weight.size(1) |  |  |  |  |
| scale_fc1 = (6.0 / fan_in_fc1) ** | 0.5 |  |  | Applies He initialization to both for consistent, reproducible | layers training |

-----

###### 56 Chapter 4 Training and verifying neural nets in raw CUDA

```
model.fc1.weight.uniform_(-scale_fc1, scale_fc1)
```

| model.fc1.bias.zero_() | Applies He initialization to both layers for consistent, |
|---|---|
| fan_in_fc2 = model.fc2.weight.size(1) | reproducible training |

```
scale_fc2 = (6.0 / fan_in_fc2) ** 0.5
model.fc2.weight.uniform_(-scale_fc2, scale_fc2)
model.fc2.bias.zero_()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

With the model initialized, we need a training loop that not only trains the model but also carefully times each distinct step of the process. The train\_timed function in listing 4.3 is designed for this purpose. It iterates through the data and records the time spent on data loading, the forward pass, loss computation, the backward pass, and the optimizer's weight updates. This level of detail is crucial for identifying performance bottlenecks later on.

**Listing 4.3 Detailed timed training loop**

```python
def train_timed(model, criterion, optimizer, epoch, timing_stats,
epoch_losses):
model.train()
epoch_loss = 0.0
iters_per_epoch = math.ceil(TRAIN_SIZE / batch_size)
for i in range(iters_per_epoch):
data_start = time.time()
data = train_data[i * batch_size : (i + 1) * batch_size]
target = train_labels[i * batch_size : (i + 1) * batch_size]
data_end = time.time()
timing_stats['data_loading'] += data_end -
```

`data_start` **Times data slicing (data**

**is already on the GPU)**

```
optimizer.zero_grad()
forward_start = time.time()
outputs = model(data)
forward_end = time.time()
timing_stats['forward'] += forward_end -
```

`forward_start` **Times the forward pass**

**through the model**

```
loss_start = time.time()
loss = criterion(outputs, target)
epoch_loss += loss.item()
loss_end = time.time()
timing_stats['loss_computation'] += loss_end -
```

`loss_start` **Times the loss calculation**

```
backward_start = time.time()
loss.backward()
backward_end = time.time()
timing_stats['backward'] += backward_end - Times the backward pass
```

`backward_start` **to compute gradients**

-----

###### Level2: Backpropagation theory 57

```
update_start = time.time()
optimizer.step()
optimizer.zero_grad()
update_end = time.time()
timing_stats['weight_updates'] += update_end -
```

`update_start` **Times the optimizer step**

```
epoch_losses.append(epoch_loss / iters_per_epoch) to update model weights
```

After running the timed training loop from v1.py, we get a clear performance breakdown that will serve as our benchmark. This breakdown allows us to see not just the total training time but exactly where that time is spent. This detailed profile is our "gold standard" for performance, which we will strive to understand and eventually beat with our custom CUDA code.

**Listing 4.4 PyTorch timing results**

```
Epoch 0 loss: 0.4547
Epoch 1 loss: 0.2360
Epoch 2 loss: 0.1762
Epoch 3 loss: 0.1371
Epoch 4 loss: 0.1093
Epoch 5 loss: 0.0886
Epoch 6 loss: 0.0725
Epoch 7 loss: 0.0598
Epoch 8 loss: 0.0496
Epoch 9 loss: 0.0412
Total training time: 12.9 seconds
Detailed Breakdown:
```

| Data loading: | 0.264s ( 2.1%) |
|---|---|
| Forward pass: | 1.966s ( 15.3%) |
| Loss computation: | 1.094s ( 8.5%) |
| Backward pass: | 6.952s ( 54.1%) |
| Weight updates: | 2.181s ( 17.0%) |

**TIP** This carefully constructed PyTorch implementation serves as our verified baseline. We have controlled for data loading and weight initialization, and we have a detailed performance profile. Any future implementation must first closely track the epoch-by-epoch loss of this model to be considered correct (remember, the weights are drawn from different random number generators, so the values will not be identical), and only then can we begin to compare its performance.

## 4.2 Level 2: Backpropagation theory

Before we can write our own neural network training code in NumPy or C, we must first build a solid, intuitive understanding of the engine that drives it all: backpropaga*tion. This may sound like a daunting topic, but at its heart, it's just a clever and efficient*

-----

###### 58 Chapter 4 Training and verifying neural nets in raw CUDA

way to apply the chain rule of calculus to a neural network. To visualize the entire process, from data flowing forward to gradients flowing backward, consider the diagram in figure 4.2.

---

dLoss = loss = -ln(0.1) = ~2.3 element-wise...

softmax\_probs - y\_true\_one\_hot

X2 @ W2 dW2 = dX2 =

(B, 256) @ (256, 10)

= (B, 10)

X2.T @ dLoss dLoss @ W2.T (256, B) @ (B, 10) (B, 10) @ (10, 256) = (256, 10) = (B, 256)

ReLU

(B, 256) -> x2 = relu(x)

d\_ReLU\_out

dX2 \* dReLU(x) X1 @ W1 = (B, 256)

(B, 784) @ (784, 256)

| = (B, 256) | dW1 = | dX1 = |
|---|---|---|
|  | X1.T @ d_ReLU_out | d_ReLU_out @ W1.T |
|  | (784, B) @ (B, 256) | (B, 256) @ (256, 784) |

= (784, 256) = (B, 784) Element-wise gradient updates. Flattened images - batched w\_i -= dw\_i \* learning\_rate B (B, 784)

Set grads back to zero. 784

28x28 px

Image **Training loop**

batched

For reference, A @ B means matrix multiply A times B.

###### Inputs

###### START HERE

**Figure 4.2 Forward pass, backward pass, weight update, and shape flow**

-----

###### Level2: Backpropagation theory 59

Notice in figure 4.2 that the shapes of the weights and their corresponding gradients are identical. This is a fundamental property of backpropagation and a powerful debugging tool. If your gradient shapes don't match your parameter shapes, something is wrong. As we will see, ensuring that these dimensions align is half the battle.

### 4.2.1 Building intuition: From slopes to gradients

To kick off this bag of theory, we'll begin with the most basic concept: the slope of a linear function. For y = mx + b, the slope is dy/dx = m, as shown in figure 4.3.

Linear function Gradient (derivative) dy/dx = 2

4 2.0 2 1.5

slope = rise/run = 2 0 1.0 -2 0.5

-4 0.0

-3 -2 -1 0 1 2 3 -3 -2 -1 0 1 2 3

**Figure 4.3 Linear function and its constant gradient**

This slope tells us, "If I increase x by 1, y increases by m." Now consider a function of two variables, such as z = 2x + 3y + 1. We now have partial derivatives: dz/dx = 2 (how

```
z changes when only x changes) and dz/dy = 3 (how z changes when only y changes).
```

At its core, backpropagation is an elegant application of the chain rule from calculus. Imagine a simple chain of calculations: an input x is passed to a function g to produce y, and y is then passed to a function f to produce the final result z. The chain rule allows us to find how a small change in x affects z by multiplying the sensitivities of each step along the chain. This concept is the key to training neural networks. We have a loss function L that depends on the network's output, which in turn depends on many layers of functions involving weights and inputs. The chain rule lets us calculate how any given weight affects the final loss by propagating the gradient backward from the loss, one step at a time. To trace this flow, it's helpful to visualize our operations as a computational graph. In figure 4.4, each node in the graph represents a value, and the edges show the operations that connect those values. The forward pass flows from left to right, calculating the final loss L. The backward pass flows from right to left, starting with the gradient of the

-----

###### 60 Chapter 4 Training and verifying neural nets in raw CUDA

loss with respect to itself (dL/dL = 1) and using the chain rule at each step to find the gradient of the loss with respect to every input variable.

A=2.0

A

dL/dA=42.0 C=6.0 D=7.0 L=49.0

C add D \*\*2 L Cycle back after the mul loss is computed

dL/dC=14.0 dL/dD=14.0 dL/dL=1.0 B=3.0

B

dL/dB=28.0

**Figure 4.4 Computational graph showing forward and backward passes: the forward pass calculates values from inputs to loss, and the backward pass propagates gradients from loss back to inputs using the chain rule.**

To make this concrete, consider a simple example that matches figure 4.4. We have two inputs, A and B, that are multiplied to produce C. Then C is added to a constant (1.0) to produce D. Finally, D is squared to produce the loss L. The following listing shows how the values flow forward and how the gradients flow backward using the chain rule.

**Listing 4.5 Simple backpropagation example matching figure 4.4**

```python
A = 2.0
B = 3.0
C = A * B
D = C + 1.0
L = D ** 2
dL_dL = 1.0
dL_dD = dL_dL * (2 * D)
dL_dC = dL_dD * 1.0
dL_dA = dL_dC * B
dL_dB = dL_dC * A
print(f"Forward:
print(f"Backward:
```

```
# C = 6.0
# D = 7.0
# L = 49.0
```

```
A={A}, B={B} -> C={C}
dL/dD={dL_dD}, dL/dC={dL_dC},
```

```
# starting
# dL_dD
# dL_dC
# dL_dA
# dL_dB
```

###### The gradient of L with respect to itself is always 1.

```
gradient
= 1 * 14 = 14.0
= 14 * 1 = 14.0
= 14 * 3 = 42.0
= 14 * 2 = 28.0
```

###### Derivative of L = D^2

**gives dL/dD = 2D = 14**

###### Because D = C + 1, the gradient passes through unchanged (dD/dC = 1). Because C = A * B, the gradient with respect to A is dL/dC

**\* B = 14 \* 3 = 42.**

```
-> D={D} -> L={L}")
dL/dA={dL_dA}, dL/dB={dL_dB}")
```

###### Similarly, the gradient with respect to B is dL/dC * A = 14 * 2 = 28.

This example shows the essence of backpropagation: we compute values in the forward pass and then use the chain rule to propagate gradients backward. Each intermediate node in figure 4.4 shows both its forward value (data) and its backward gradient. The

-----

###### Level2: Backpropagation theory 61

same principles apply whether we have five values or millions. Although this example works with single numbers (scalars), real neural networks use operations on vectors and matrices for efficiency.

### 4.2.2 Scaling to vectors and matrices

The forward pass of a linear layer is a matrix multiplication, Y = XW + b, where X is the input, W is the weight matrix, and b is the bias vector. To add a vector b to a matrix XW, we use broadcasting: the bias vector is added to each row of the output matrix. If X has shape (batch\_size, input\_dim) and W has shape (input\_dim, output\_dim), then

```
XW produces a matrix of shape (batch_size, output_dim), and the bias b, with shape
```

(output\_dim,), is added element-wise to each of those rows. Before we proceed, a note on notation: throughout this chapter, we use @ to denote matrix multiplication, following Python/NumPy convention. The superscript ^T means transpose: X^T flips a matrix so that rows become columns. The symbol L represents the loss, a single scalar value computed from the network's output Y and the true labels. When we write dL/dW, we mean the gradient of the loss with respect to the weight matrix: a matrix of the same shape as W, where each element tells us how much the loss would change if we adjusted that particular weight. The backward pass is more interesting. We must calculate three gradients: one for the weights (dL/dW), one for the bias (dL/db), and one for the input (dL/dX) to be passed to the previous layer. As shown in figures 4.5 and 4.6, these are also matrix multiplications:

```
¡ dL/dW = X^T @ dL/dY
¡ dL/dX = dL/dY @ W^T
¡ dL/db = sum(dL/dY, axis=0)
```

dL/dY W^T (B, H) (H, C)

| X^T | dL/dW | dL/dY | dL/dX |
|---|---|---|---|
| (C, B) | (C, H) | (B, H) | (B, C) |

| Figure 4.5 | Linear layer backward pass for | Figure 4.6 | Linear layer backward pass for |
|---|---|---|---|
| weight | gradients | input | gradients |

-----

###### 62 Chapter 4 Training and verifying neural nets in raw CUDA

Notice what the X^T @ dL/dY product does along the batch dimension: each weight gradient element sums contributions from every sample in the batch. A single weight accounts for how it affected the loss across all inputs at once, which is exactly why larger batches give smoother gradient estimates. The input gradient dL/dX = dL/dY @ W^T is the same kind of product, just with the shapes arranged so the result matches X. A crucial detail is that the backward pass needs values computed during the forward pass. For example, to calculate dL/dW, we need the original input X that was fed into the layer. This is why we must cache these intermediate values for later use.

### 4.2.3 Layer-by-layer gradient derivations

We can now summarize the gradient formulas for each component of our MLP. To do so, let's break down the gradients for each layer type individually.

###### Linear layer gradients

```
For a linear layer Y = XW + b, the gradients are
¡ dL/dW = X^T @ dL/dY
¡ dL/db = sum(dL/dY, axis=0)
¡ dL/dX = dL/dY @ W^T
```

###### ReLU activation gradients

The ReLU activation function, Y = max(0, X), acts as a simple gate for gradients. During the forward pass, ReLU takes the input Z and produces an activation output, A. During the backward pass, we receive the gradient from the next layer, dL/dA (the gradient of the loss with respect to the ReLU output), and need to compute dL/dZ (the gradient with respect to the ReLU input) to pass to the previous layer. As illustrated in figure 4.7, if a neuron's input Z during the forward pass is positive, the gate is open, and the gradient passes through unchanged (indicated by the arrows). If the input is negative, the gate is closed, and the gradient is blocked (becomes 0). This is implemented as an element-wise multiplication: dL/dZ = dL/dA \* (Z > 0), where (Z > 0) creates a binary mask that is 1 for positive values and 0 for negative values.

| [1.2, | Z (pre-act) -0.5, 0.8, | -0.1] |  |  |  |  |
|---|---|---|---|---|---|---|
| Z > | 0 (binary [1, 0, 1, 0] | mask) |  |  |  |  |

Z > 0 -> mask = 1 (passes grad)

Z <= 0 -> mask = 0 (blocks grad)

**dL/dA (upstream grads) [0.1, 0.9, 0.4, 0.6]**

**dL/dZ (gated grads) [0.1, 0, 0.4, 0]**

1.2 > 0 -0.5 <= 0 0.8 > 0 -0.1 <= 0

**Figure 4.7 ReLU activation backward-pass gradient gating**

-----

###### Level2: Backpropagation theory 63

###### Softmax + cross-entropy combined gradient

In classification tasks, we typically combine two operations at the output: the softmax activation, which converts raw scores into probabilities that sum to 1, and the crossentropy loss function. Cross-entropy measures how close our predicted probability distribution is to the true distribution. For a correct classification, the true distribution has a value of 1 for the correct class and 0 for all other classes. This representation is called a "one-hot-encoded label" or simply OneHotLabels. The combined gradient for the softmax activation and cross-entropy loss has a remarkably simple and numerically stable formula: dL/dZ = (Probabilities - OneHotLabels)

```
/ batch_size.
```

### 4.2.4 Common pitfalls and debugging tips

When implementing backpropagation manually, several common problems can arise. The most frequent error is getting matrix dimensions wrong. Always ensure that the shapes of your gradients match the shapes of the parameters or inputs they correspond to. During training, monitor the magnitude of your gradients. If they become excessively large (exploding) or excessively small (vanishing), they can destabilize or stall the learning process. A simple utility function, shown in the following listing, can help track the norm and maximum value of gradients to detect these problems early.

**Listing 4.6 Utility function to monitor gradient health**

```python
def check_gradients(gradients, name):
for param_name, grad in gradients.items():
grad_norm = np.linalg.norm(grad)
grad_max = np.max(np.abs(grad))
print(f"{name} - {param_name}: norm={grad_norm:.6f},
max={grad_max:.6f}")
if grad_norm > 10:
print(f" WARNING: Large gradient norm in {param_name}")
if grad_max < 1e-6:
print(f" WARNING: Very small gradients in {param_name}")
```

### 4.2.5 Vectorized implementation for batches

For efficiency, we process multiple samples (a batch) at once. The forward pass naturally handles this through matrix multiplication. The backward pass requires careful handling of the batch dimension. The key difference, as shown in listing 4.7, is that we use NumPy's advanced indexing to subtract 1 from the correct class position for every sample in the batch simultaneously. The weight gradients then naturally accumulate the contributions from all samples.

-----

###### 64 Chapter 4 Training and verifying neural nets in raw CUDA

**Listing 4.7 Backward pass for batches**

```python
def backward_batch(X, Z1, A1, Z2, y_true, W1, W2):
batch_size = X.shape[0]
```

`probs = softmax(Z2)` **Computes the combined** `grad_Z2 = probs.copy()` **softmax and cross-entropy**

```python
grad_Z2[np.arange(batch_size), y_true] -= 1 gradient for the whole batch
grad_Z2 /= batch_size
```

`grad_W2 = A1.T @ grad_Z2` **Computes the gradients for**

```
grad_b2 = np.sum(grad_Z2, axis=0) the second-layer weights
grad_A1 = grad_Z2 @ W2.T
grad_Z1 = grad_A1 * relu_derivative(Z1)
grad_W1 = X.T @ grad_Z1
grad_b1 = np.sum(grad_Z1, axis=0)
return grad_W1, grad_b1, grad_W2, grad_b2
```

## 4.3 Level 3: The NumPy approach

Now that we have our PyTorch gold standard, we take our first step down the abstraction ladder by reimplementing the entire training pipeline in NumPy. This forces us to manually implement every mathematical operation: the forward pass, the activation functions, the cross-entropy loss, and, most importantly, the entire backpropagation algorithm. Previously, we explored automatic differentiation theory through the computational graph in section 4.2.1 and relied on PyTorch's .backward() method; now we put that theory into practice. This NumPy version serves two critical purposes. First, it's a direct, code-based application of the backpropagation theory we've discussed. Second, it acts as an essential debugging bridge between the "magic" of autograd and the explicit, lowlevel logic we'll need for C and CUDA. If our NumPy implementation drives the loss down at the same rate and reaches a similar final accuracy as our PyTorch implementation (recall that the exact per-epoch numbers differ because the two use different random initializations), we can be much more confident that our understanding of the underlying mathematics is correct. The full code for this section is in the v2.py script.

### 4.3.1 Manual implementation in NumPy

The core of the NumPy version involves creating functions for each step of the process. In listing 4.8, we define the building blocks: the linear layer's forward pass, the ReLU activation, the softmax function for output probabilities, and the crossentropy loss. Each function directly translates its mathematical definition into NumPy code.

-----

###### Level3: The NumPy approach 65

**Listing 4.8 Core mathematical functions in NumPy**

```python
def relu(x):
```

`return np.maximum(0, x)` **ReLU activation: max(0, x)** `def relu_derivative(x):` **ReLU gradient: 1 if x > 0, else 0**

```python
return (x > 0).astype(float)
def initialize_weights(input_size, output_size): He initialization for weights
scale = np.sqrt(6.0 / input_size)
return (np.random.rand(input_size, output_size) * 2.0 - 1.0) * scale
def initialize_bias(output_size): Initializes biases to zero
return np.zeros((1, output_size))
```

`def linear_forward(x, weights, bias):` **Linear layer forward pass: y = Wx + b**

```
return x @ weights + bias
```

###### Softmax converts

`def softmax(x):` **logits to probabilities.**

```python
exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
return exp_x / np.sum(exp_x, axis=1, keepdims=True)
def cross_entropy_loss(y_pred, y_true): Cross-entropy loss
batch_size = y_pred.shape[0] for classification
probabilities = softmax(y_pred)
correct_log_probs = np.log(probabilities[np.arange(batch_size), y_true])
loss = -np.sum(correct_log_probs) / batch_size
return loss
```

With these building blocks, we can construct the NeuralNetwork class, shown in listing 4.9. This class holds the model's weights and biases and defines the forward and

```
backward passes. Figure 4.8 visualizes the data flow for these passes. To manually imple-
```

ment backpropagation, we must explicitly manage the intermediate results from the forward pass. The forward method doesn't just return the final output; it also returns a

```
cache containing the inputs and activation outputs from each layer. This cache is then
```

passed to the backward method, which uses these stored values to calculate the gradients for each weight and bias correctly according to the chain rule. The final step in training is updating the weights. Once we have the gradients, we adjust each weight by taking a step in the direction opposite of its gradient. The size of this step is controlled by the learning\_rate hyperparameter. A smaller learning rate means smaller, more cautious steps toward the minimum loss, while a larger learning rate takes bigger steps but risks overshooting. This weight-update step, combined with the forward and backward passes, completes one iteration of the training loop. As you can see in the following listing, this architecture is simple at the diagram level and at the NumPy code level (forward, backward, update weights).

-----

###### 66 Chapter 4 Training and verifying neural nets in raw CUDA

Forward pass

| x | linear_forward 1 | fc1 out | relu | relu out | linear_forward 2 | fc2 out |
|---|---|---|---|---|---|---|

cache x cache fc1 out cache relu out

Backward pass

linear\_backward 1 grad relu\_derivative grad linear\_backward 2 grad loss

relu input2 logits

| grad | grad | grad | grad |
|---|---|---|---|
| weights1 | bias1 | weights2 | bias2 |

**Figure 4.8 Data flow of forward and backward passes in NumPy**

**Listing 4.9 Complete neural network class in NumPy Returns both the final output and the cached values needed for backpropagation**

`class NeuralNetwork:` **Initializes the weights and**

```python
def __init__(self, input_size, hidden_size, biases for a two-layer network
output_size):
self.weights1 = initialize_weights(input_size, hidden_size)
self.bias1 = initialize_bias(hidden_size)
self.weights2 = initialize_weights(hidden_size, output_size)
self.bias2 = initialize_bias(output_size)
```

`def forward(self, x):` **Forward pass: computes the output**

`batch_size = x.shape[0]` **and caches intermediate values**

```
fc1_input = x.reshape(batch_size, -1)
fc1_output = linear_forward(fc1_input, self.weights1, self.bias1)
relu_output = relu(fc1_output)
fc2_output = linear_forward(relu_output, self.weights2, self.bias2)
return fc2_output, (fc1_input, fc1_output,
```

`relu_output)` **Backward pass: computes the**

**gradients using the chain rule**

```python
def backward(self, grad_output, cache):
x, fc1_output, relu_output = cache
grad_fc2, grad_weights2, grad_bias2 = linear_backward(grad_output,
relu_output, self.weights2)
grad_relu = grad_fc2 * relu_derivative(fc1_output)
```

###### Unpacks cached intermediate values from the forward pass

-----

###### Level3: The NumPy approach

```python
grad_fc1, grad_weights1, grad_bias1 = linear_backward(grad_relu, x,
self.weights1)
return grad_weights1, grad_bias1, grad_weights2,
def update_weights(self, grad_weights1, grad_bias1, grad_weights2,
grad_bias2, learning_rate):
self.weights1 -= learning_rate * grad_weights1
self.bias1 -= learning_rate * grad_bias1
self.weights2 -= learning_rate * grad_weights2
self.bias2 -= learning_rate * grad_bias2
```

### 4.3.2 Verification technique: Isolate and conquer

To check our manual implementation, we use a debugging approach called: Isolate and Conquer. When a complex system breaks, the most effective strategy is to chop it into smaller, testable pieces. Instead of debugging the whole training loop, we verify each component individually against our trusted PyTorch model. We can load the exact same initial weights into both models, pass the same single batch of data through them, and assert that the output of each layer is numerically identical. The following listing shows how we could write a simple test to verify our first linear layer and ReLU activation in NumPy against the PyTorch equivalent.

**Listing 4.10 Verifying NumPy against PyTorch layer outputs**

```
import torch
import numpy as np
state_dict = torch.load("initial_weights.pth")
np_model.weights1 = state_dict['fc1.weight'].T.numpy()
np_model.bias1 = state_dict['fc1.bias'].numpy()
pytorch_hidden_output = model.relu(model.fc1(batch_data))
np_fc1_output = linear_forward(batch_data_np,
np_model.weights1, np_model.bias1)
np_relu_output = relu(np_fc1_output)
assert np.allclose(pytorch_hidden_output.detach().cpu().
numpy(), np_relu_output, atol=1e-6)
```

```
grad_bias2
```

###### Updates weights using gradient descent with learning_rate as the step size

###### Loads the PyTorch model weights Copies the weights to the NumPy model (transposes for NumPy convention) Computes the PyTorch forward

**pass through the**

**first layer**

###### Computes the NumPy forward pass through the first layer Asserts that the outputs match within numerical tolerance

By performing this check for each layer in the forward pass and then for each gradient in the backward pass, we can systematically pinpoint any discrepancies and gain high confidence in our manual implementation before moving to C.

### 4.3.3 Performance

The NumPy implementation runs entirely on the CPU. Although NumPy's underlying operations are written in optimized C, the overall performance is significantly slower

-----

###### 68 Chapter 4 Training and verifying neural nets in raw CUDA

than our GPU-accelerated PyTorch version. The timed training loop in v2.py gives us a CPU baseline.

**Listing 4.11 NumPy timing results**

```
Epoch 0 loss: 0.4702
Epoch 1 loss: 0.2384
Epoch 2 loss: 0.1783
Epoch 3 loss: 0.1391
Epoch 4 loss: 0.1111
Epoch 5 loss: 0.0899
Epoch 6 loss: 0.0734
Epoch 7 loss: 0.0605
Epoch 8 loss: 0.0501
Epoch 9 loss: 0.0417
Total training time: 11.8 seconds
Detailed Breakdown:
```

| Data loading: | 0.010s ( 0.1%) |
|---|---|
| Forward pass: | 2.483s ( 21.0%) |
| Loss computation: | 0.567s ( 4.8%) |
| Backward pass: | 3.200s ( 27.1%) |
| Weight updates: | 5.539s ( 46.9%) |

## 4.4 Level 4: The C approach

Moving from NumPy to C takes us another step closer to the hardware. Here, we lose the convenience of NumPy's high-level array operations and must implement everything from scratch. The C version, found in v3.c, is a direct, line-by-line translation of the logic we developed in NumPy. Its purpose is not speed, but to serve as an explicit, low-level blueprint for the CUDA implementation to come. This section will be more verbose than the previous ones. Even though the underlying logic is the same as NumPy, the C implementation requires us to manually handle every detail, from memory allocation to the loops for matrix multiplication.

### 4.4.1 Setup: Headers, constants, and data structures

We begin by including the necessary C standard libraries and defining the constants for our network, as shown in listing 4.12. These are identical to the hyperparameters used in our PyTorch and NumPy versions to ensure a fair comparison.

**Listing 4.12 C headers and constants**

```
#include <stdio.h>
#include <stdlib.h>
```

```
#include <math.h>
#include <time.h>
#include <string.h> // for memset
```

-----

###### Level4: The C approach 69

```
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000
```

**NOTE** The C and CUDA listings that follow are excerpts that highlight the logic being discussed, so a single listing pasted on its own will not compile. The complete, compilable files (v3.c, v4.cu, and v5.cu), with every #include and

```
#define in place, are in the book's code repository. The CUDA version (v4.cu)
```

reuses the same standard-library headers and constants shown here in listing 4.12 and adds the CUDA-specific ones in listing 4.19. Next, we define a struct in listing 4.13 to hold all the pointers for our network's weights, biases, and their corresponding gradients. This is our C equivalent of the Python NeuralNetwork class, and it acts as a container for the model's parameters.

| Listing 4.13 Neural network structure in C |
|---|
| typedef struct { float *weights1, *weights2, *bias1, *bias2; float *grad_weights1, *grad_weights2, *grad_bias1, *grad_bias2; } NeuralNetwork; Unlike Python, C requires manual memory management. We must explicitly allocate memory for every component of our network using malloc. The initialize_neural_ network function in listing 4.14 handles this by allocating space for all weights and gradients. It then calls helper functions to initialize the weights with our standard He initialization and the biases to zero. This explicit control over memory will be central to our work in CUDA. |
| Listing 4.14 Network initialization and memory allocation in C |

```
void initialize_weights(float *weights, int input_size, int output_size) {
float scale = sqrtf(6.0f / input_size);
for (int i = 0; i < input_size * output_size; i++) {
weights[i] = ((float)rand() / RAND_MAX) *
```

`2.0f * scale - scale;` **He initialization, identical to our** `}` **PyTorch and NumPy versions**

```
}
void initialize_bias(float *bias, int size) {
for (int i = 0; i < size; i++) {
bias[i] = 0.0f;
```

-----

###### 70 Chapter 4 Training and verifying neural nets in raw CUDA

```
}
```

```
}
void initialize_neural_network(NeuralNetwork *nn) {
nn->weights1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
nn->weights2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
nn->bias1 = malloc(HIDDEN_SIZE * sizeof(float));
nn->bias2 = malloc(OUTPUT_SIZE * sizeof(float));
nn->grad_weights1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
nn->grad_weights2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
nn->grad_bias1 = malloc(HIDDEN_SIZE * sizeof(float));
nn->grad_bias2 = malloc(OUTPUT_SIZE * sizeof(float));
initialize_weights(nn->weights1, INPUT_SIZE, HIDDEN_SIZE);
initialize_weights(nn->weights2, HIDDEN_SIZE, OUTPUT_SIZE);
initialize_bias(nn->bias1, HIDDEN_SIZE);
initialize_bias(nn->bias2, OUTPUT_SIZE);
}
```

### 4.4.3 The forward pass in C

The forward pass requires us to implement each mathematical operation manually. The most computationally intensive part is matrix multiplication, which we implement with three nested loops, as shown in the matmul\_a\_b function in listing 4.15. The function performs the operation C = A @ B, where matrix A has shape (m, n), matrix B has shape (n, k), and the result C has shape (m, k). The parameters m, n, and k therefore define the dimensions of the matrices involved. We also need explicit loops for adding the bias and applying the ReLU activation. These handwritten loops are the exact computations we will later parallelize in CUDA.

**Listing 4.15 C forward pass operations**

```
void matmul_a_b(float *A, float *B, float *C, int m, int n, int k) {
for (int i = 0; i < m; i++) {
for (int j = 0; j < k; j++) {
C[i * k + j] = 0.0f;
for (int l = 0; l < n; l++) {
C[i * k + j] += A[i * n + l] * B[l * k + j];
}
}
```

```
}
}
void bias_forward(float *x, float *bias, int batch_size, int size) {
for (int b = 0; b < batch_size; b++) {
for (int i = 0; i < size; i++) {
x[b * size + i] += bias[i];
}
}
```

```
}
void relu_forward(float *x, int size) {
```

-----

###### Level4: The C approach 71

```
for (int i = 0; i < size; i++) {
x[i] = fmaxf(0.0f, x[i]);
}
}
```

### 4.4.4 The backward pass in C

The backward pass similarly requires manually implementing the gradient calculations. This involves creating transposed matrix multiplication variants (matmul\_at\_b for A.T @ B and matmul\_a\_bt for A @ B.T) and functions for the ReLU and bias backward passes, as shown in listing 4.16. Each of these functions corresponds directly to a step in the chain rule we derived for backpropagation.

**Listing 4.16 C backward pass operations**

```
void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
for (int i = 0; i < n; i++) {
for (int j = 0; j < k; j++) {
C[i * k + j] = 0.0f;
for (int l = 0; l < m; l++) {
C[i * k + j] += A[l * n + i] * B[l * k + j];
}
}
```

```
}
}
void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
for (int i = 0; i < m; i++) {
for (int j = 0; j < k; j++) {
C[i * k + j] = 0.0f;
for (int l = 0; l < n; l++) {
C[i * k + j] += A[i * n + l] * B[j * n + l];
}
}
}
```

```
}
void relu_backward(float *grad, float *x, int size) {
for (int i = 0; i < size; i++) {
grad[i] *= (x[i] > 0);
}
}
```

```
void bias_backward(float *grad_bias, float *grad, int batch_size, int size) {
for (int i = 0; i < size; i++) {
grad_bias[i] = 0.0f;
for (int b = 0; b < batch_size; b++) {
grad_bias[i] += grad[b * size + i];
}
}
}
```

-----

###### 72 Chapter 4 Training and verifying neural nets in raw CUDA

### 4.4.5 Training loop and performance

The training loop in C orchestrates these functions. It iterates through epochs and batches, using pointer arithmetic to efficiently slice the data without making copies, as illustrated in figure 4.9.

###### X_train

X\_train (entire dataset in memory)

---

Batch i

---

**batch\_input**

batch\_input = X\_train + i \* BATCH\_SIZE \* INPUT\_SIZE

**Figure 4.9 Pointer arithmetic for efficient mini-batch selection**

The performance, as shown in listing 4.17, is significantly slower than that of both PyTorch and NumPy, providing a clear baseline for a naive, single-threaded CPU implementation. This stark difference in performance is our primary motivation for moving to the GPU.

**Listing 4.17 C training loop with integrated timing**

```
void train_timed(NeuralNetwork *nn, float *X_train, int *y_train) {
float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
int num_batches = TRAIN_SIZE / BATCH_SIZE;
TimingStats stats = {0};
```

```
struct timespec total_start, total_end, step_start, step_end;
clock_gettime(CLOCK_MONOTONIC, &total_start);
for (int epoch = 0; epoch < EPOCHS; epoch++) {
float total_loss = 0.0f;
for (int batch = 0; batch < num_batches; batch++) {
int start_idx = batch * BATCH_SIZE;
clock_gettime(CLOCK_MONOTONIC, &step_start);
float *batch_input = &X_train[start_idx * INPUT_SIZE];
int *batch_labels = &y_train[start_idx];
clock_gettime(CLOCK_MONOTONIC, &step_end);
stats.data_loading += get_time_diff(step_start, step_end);
```

-----

###### Level4: The C approach 73

```
forward_timed(nn, batch_input, hidden, output,
```

`BATCH_SIZE, &stats);` **Executes the forward**

**pass, from input to**

```
clock_gettime(CLOCK_MONOTONIC, &step_start); final logits
float loss = cross_entropy_loss(output,
```

`batch_labels, BATCH_SIZE);` **Computes the cross-entropy**

```
total_loss += loss; loss on the CPU
clock_gettime(CLOCK_MONOTONIC, &step_end);
stats.cross_entropy += get_time_diff(step_start, step_end);
backward_timed(nn, batch_input, hidden, output, Executes the
batch_labels, BATCH_SIZE, &stats); backward pass to
```

**compute all gradients**

```
update_weights_timed(nn, &stats); Updates the network weights
```

`}` **using the computed gradients**

```
printf("Epoch %d loss: %.4f\n", epoch, total_loss / num_batches);
}
clock_gettime(CLOCK_MONOTONIC, &total_end);
stats.total_time = get_time_diff(total_start, total_end);
printf("\n=== C CPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
printf("Total training time: %.1f seconds\n\n", stats.total_time);
printf("Detailed Breakdown:\n");
printf(" Data loading: %6.3fs (%5.1f%%)\n", stats.data_loading,
100.0 * stats.data_loading / stats.total_time);
double forward_pass = stats.fwd_matmul1 + stats.fwd_bias1 +
stats.fwd_relu + stats.fwd_matmul2 + stats.fwd_bias2 +
stats.fwd_softmax;
printf(" Forward pass: %6.3fs (%5.1f%%)\n", forward_pass,
100.0 * forward_pass / stats.total_time);
printf(" Loss computation: %6.3fs (%5.1f%%)\n", stats.cross_entropy,
100.0 * stats.cross_entropy / stats.total_time);
double backward_pass = stats.bwd_output_grad + stats.bwd_matmul2 +
stats.bwd_bias2 + stats.bwd_relu + stats.bwd_matmul1 +
stats.bwd_bias1;
```

| printf(" Backward pass: | %6.3fs (%5.1f%%)\\n", backward_pass, 100.0 * |
|---|---|
| backward_pass / | stats.total_time); |
| printf(" Weight updates: | %6.3fs (%5.1f%%)\\n", stats.weight_updates, |
| 100.0 * stats.weight_updates | / stats.total_time); |

```
free(hidden);
free(output);
}
```

Running this training loop produces the results in the following listing.

**Listing 4.18 C timing results**

```
Epoch 0 loss: 0.4646
Epoch 1 loss: 0.2360
```

-----

###### 74 Chapter 4 Training and verifying neural nets in raw CUDA

```
Epoch 2 loss: 0.1751
Epoch 3 loss: 0.1363
Epoch 4 loss: 0.1087
Epoch 5 loss: 0.0879
Epoch 6 loss: 0.0719
Epoch 7 loss: 0.0594
Epoch 8 loss: 0.0494
Epoch 9 loss: 0.0412
Total training time: 185.2 seconds
Detailed Breakdown:
```

| Data loading: | 0.000s ( 0.0%) |
|---|---|
| Forward pass: | 86.815s ( 46.9%) |
| Loss computation: | 0.002s ( 0.0%) |
| Backward pass: | 90.718s ( 49.0%) |
| Weight updates: | 6.391s ( 3.5%) |

## 4.5 Level 5: The naive CUDA approach

We now arrive at the core of this chapter: translating our C implementation into a parallel version using CUDA. The goal of this first "naive" implementation is correctness, not performance. We will take the explicit loops from our C functions and convert them, one by one, into simple CUDA kernels. This direct translation provides a functional GPU-based MLP and a solid foundation on which we will build optimizations in later chapters. The full implementation is in v4.cu. It mirrors the structure of v3.c but uses host functions that launch GPU kernels instead of executing CPU loops.

### 4.5.1 CUDA setup and memory management

The first major change is the introduction of the CUDA runtime API. We include the cuda\_runtime.h header and define a CUDA\_CHECK macro for proper error handling, as shown in listing 4.19. All CUDA API calls will be wrapped in this macro to ensure that any runtime errors are caught and reported immediately.

**Listing 4.19 CUDA headers and error handling**

```
#include <cuda_runtime.h>
#define CUDA_CHECK(call) \
do { \
cudaError_t error = call; \
if (error != cudaSuccess) { \
fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
cudaGetErrorString(error)); \
cudaDeviceReset(); \
exit(EXIT_FAILURE); \
} \
} while(0)
```

-----

###### Level5: The naive CUDA approach 75

In CUDA, we must distinguish between host (CPU) memory and device (GPU) memory, as illustrated in figure 4.10. Our network parameters and data must reside on the GPU to be accessed by the kernels.

"cudaMemcpy"

| *h_data (short for host data on RAM) |  | *d_data (short for device data on VRAM) | Figure 4.10 Host memory (RAM) and device memory (VRAM) are physically separate. A host pointer from malloc cannot be dereferenced in a kernel, and a device pointer from cudaMalloc cannot be dereferenced on the host. |
|---|---|---|---|
| We modify the malloc, as shown in weights are still This two-step process applications where | listing 4.20. generated on (CPU we need precise | initialize_neural_network This allocates the CPU and then initialization followed control over the | function to use cudaMalloc instead of memory directly on the GPU. The initial copied to the GPU using cudaMemcpy. by GPU transfer) is typical in CUDA initial values. |
| Listing 4.20 | Allocating GPU | memory and copying | initial weights |

```
void initialize_neural_network(NeuralNetwork *nn) {
CUDA_CHECK(cudaMalloc(&nn->weights1, INPUT_SIZE * HIDDEN_SIZE *
sizeof(float)));
CUDA_CHECK(cudaMalloc(&nn->weights2, HIDDEN_SIZE * OUTPUT_SIZE *
sizeof(float)));
CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
CUDA_CHECK(cudaMalloc(&nn->grad_weights1, INPUT_SIZE * HIDDEN_SIZE
* sizeof(float)));
CUDA_CHECK(cudaMalloc(&nn->grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE
* sizeof(float)));
CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));
initialize_random_weights(nn);
}
void initialize_random_weights(NeuralNetwork *nn) {
float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE
* sizeof(float));
initialize_weights(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, INPUT_SIZE
* HIDDEN_SIZE * sizeof(float),
```

| cudaMemcpyHostToDevice)); | Copies the initialized weight data from |
|---|---|
| free(h_weights1); | the host (CPU) to the device (GPU) |

```
float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE
* sizeof(float));
initialize_weights(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
```

-----

###### 76 Chapter 4 Training and verifying neural nets in raw CUDA

```
CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, HIDDEN_SIZE
* OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
free(h_weights2);
float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
initialize_bias(h_bias1, HIDDEN_SIZE);
CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE
* sizeof(float), cudaMemcpyHostToDevice));
free(h_bias1);
float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
initialize_bias(h_bias2, OUTPUT_SIZE);
CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE
* sizeof(float), cudaMemcpyHostToDevice));
free(h_bias2);
}
```

This explicit separation of host and device memory is a core concept in CUDA. Every piece of data your GPU needs must be explicitly allocated on and copied to the device. Forgetting a cudaMemcpy is one of the most common sources of error in CUDA programming.

### 4.5.2 From C loops to CUDA grids: The matmul kernel

The most fundamental change is converting a function like matmul\_a\_b into a global kernel. Where the C code had nested for loops to iterate over the output matrix elements, the CUDA version launches a 2D grid of threads. Each thread is responsible for computing a single element of the output matrix, as shown in figure 4.11.

T 0,0 T 0,1 T 1,0 T 1,1

M

K

**Figure 4.11 Each thread in the 2D grid computes one element of the output matrix. A thread's position (row, col) maps directly to the element it calculates.**

The kernel code in listing 4.21 implements this mapping by using the thread and block indices to calculate a unique global row and col for each thread. This pattern of mapping thread indices to data indices is fundamental to all CUDA programming.

-----

###### Level5: The naive CUDA approach 77

| Listing 4.21 Naive matrix multiplication kernel |
|---|
| __global__ void matmul_a_b_kernel(float *A, float int n, int k) { |
| int row = blockIdx.y * blockDim.y + threadIdx.y; |
| int col = blockIdx.x * blockDim.x + threadIdx.x; |
| if (row < m && col < k) { float sum = 0.0f; for (int i = 0; i < n; ++i) { sum += A[row * n + i] * B[i * k + col]; } C[row * k + col] = sum; } } |
|  |
|  |

### 4.5.3 Forward and backward pass kernels

We apply the same transformation to all the other functions from our C implementation. Each function that performs a computation on our data becomes a CUDA kernel. The following listing shows the collection of kernels for the forward pass.

**Listing 4.22 Forward pass CUDA kernels**

```
__global__ void relu_forward_kernel(float *x, int
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) {
x[idx] = fmaxf(0.0f, x[idx]);
}
}
__global__ void bias_forward_kernel(float *x, float *bias, int
int size) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int b = idx / size;
int i = idx % size;
if (b < batch_size && i < size) {
x[idx] += bias[i];
}
}
__global__ void softmax_kernel(float *x, int batch_size, int
int b = blockIdx.x;
if (b < batch_size) {
float max_val = x[b * size];
for (int i = 1; i < size; ++i) {
max_val = fmaxf(max_val, x[b * size + i]);
}
```

```
*B, float *C, int m,
```

###### Calculates the global row index in the output matrix Calculates the global column index in the output matrix A boundary check to ensure

**threads for padding do not access invalid memory**

```
size) {
```

###### ReLU is a simple elementwise operation: each thread handles one element.

```
batch_size,
```

###### Decomposes the linear index into batch index b and feature index i

```
size) {
```

###### One block per batch sample: each block processes one row of the output.

```
float sum = 0.0f;
for (int i = 0; i < size; ++i) {
```

###### Finds the max value in the row for numerical stability before exponentiation

-----

###### 78 Chapter 4 Training and verifying neural nets in raw CUDA

```
x[b * size + i] = expf(x[b * size + i] - max_val);
sum += x[b * size + i];
}
```

```
for (int i = 0; i < size; ++i) {
x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
}
```

`}` **Clamps to the minimum value to prevent**

`}` **log(0) in the subsequent loss computation** The bias\_forward\_kernel uses a common CUDA pattern for handling batched data. It launches a 1D grid of threads large enough to cover all elements in the x tensor (batch\_size \* size). Each thread computes a unique linear index, idx. It then decomposes this linear index back into a 2D coordinate: the batch index b = idx / size and the feature index i = idx % size. This allows each thread to identify its target element in the batch and add the corresponding bias term, bias[i]. Similarly, the following listing shows the kernels for the backward pass, which handle the gradient computations.

**Listing 4.23 Backward pass CUDA kernels**

```
__global__ void matmul_a_bt_kernel(float *A, float *B, float *C, int m,
int n, int k) {
int row = blockIdx.y * blockDim.y + threadIdx.y; Computes C = A @ B.T:
int col = blockIdx.x * blockDim.x + threadIdx.x; used to propagate gradients
```

**backward through layers**

```
if (row < m && col < k) {
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
sum += A[row * n + i] * B[col * n + i];
}
C[row * k + col] = sum;
}
}
__global__ void matmul_at_b_kernel(float *A, float *B, float *C, int m,
int n, int k) {
int row = blockIdx.y * blockDim.y + threadIdx.y; Computes C = A.T @ B:
int col = blockIdx.x * blockDim.x + threadIdx.x; used to compute weight
```

**gradients (input.T @ grad)**

```
if (row < n && col < k) {
float sum = 0.0f;
for (int i = 0; i < m; ++i) {
sum += A[i * n + row] * B[i * k + col];
}
C[row * k + col] = sum;
}
```

```
}
__global__ void
float *output,
```

```
compute_output_gradients_kernel(float *grad_output,
int *labels, int batch_size) {
```

-----

###### Level5: The naive CUDA approach 79

```
int b = blockIdx.x * blockDim.x + threadIdx.x;
if (b < batch_size) {
for (int i = 0; i < OUTPUT_SIZE; ++i) {
grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
}
grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
for (int i = 0; i < OUTPUT_SIZE; ++i) {
grad_output[b * OUTPUT_SIZE + i] /= batch_size;
}
```

`}` **Combined softmax + cross-entropy** `}` **gradient: subtracts 1 from the correct class**

```
__global__ void relu_backward_kernel(float *grad, float *x, int size) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) {
grad[idx] *= (x[idx] > 0.0f ? 1.0f : 0.0f);
```

`}` **ReLU derivative: gradient passes through** `}` **where the input was positive; zero otherwise**

```
__global__ void bias_backward_kernel(float *grad_bias, float *grad,
int batch_size, int size) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size) {
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) { Naive reduction: each
sum += grad[b * size + i]; thread sums across
```

`}` **the batch dimension**

```
grad_bias[i] = sum; for one bias term.
}
}
```

The compute\_output\_gradients\_kernel is launched with a 1D grid where each thread is responsible for one sample in the batch. The thread then calculates the gradient for its corresponding output vector by subtracting 1 from the correct class logit. The

```
bias_backward_kernel demonstrates a parallel reduction. Each thread is assigned to a
```

single bias term (i). It then iterates through all samples in the batch (for (int b = 0;

```
b < batch_size; b++)) to sum up the gradients that correspond to its assigned bias. This
```

is a naive reduction; we will see much more efficient ways to do this later.

### 4.5.4 The training loop and performance

The training loop now orchestrates kernel launches instead of function calls. For each batch, it first copies the input data and labels from the host to the device using

```
cudaMemcpy, then calls the forward, backward, and weight update functions, which in
```

turn launch their respective GPU kernels. A key difference is the need for synchronization. Since kernel launches are asynchronous, we must call cudaDeviceSynchronize() after each kernel, as in the following listing, to ensure its computation is complete before we time it.

-----

###### 80 Chapter 4 Training and verifying neural nets in raw CUDA

**Listing 4.24 Naive CUDA forward pass orchestration**

```
void forward_pass(NeuralNetwork *nn, float *input, float *hidden,
float *output, int batch_size) {
```

`dim3 block_size(32, 32);` **Launches the matrix multiplication**

**kernel for the first linear layer**

```
dim3 grid_size1((HIDDEN_SIZE + block_size.x - 1) / block_size.x,
(batch_size + block_size.y - 1) / block_size.y);
matmul_a_b_kernel<<<grid_size1, block_size>>>(input, nn->weights1,
hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
CUDA_CHECK(cudaDeviceSynchronize());
bias_forward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256,
256>>>(hidden, nn->bias1, batch_size,
```

| HIDDEN_SIZE); | Launches the kernel to add |
|---|---|
| CUDA_CHECK(cudaDeviceSynchronize()); | the bias vector to the result |

```
relu_forward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256,
256>>>(hidden, batch_size * HIDDEN_SIZE);
CUDA_CHECK(cudaDeviceSynchronize()); Launches the ReLU activation kernel
dim3 grid_size2((OUTPUT_SIZE + block_size.x - 1) / block_size.x,
(batch_size + block_size.y - 1) / block_size.y);
matmul_a_b_kernel<<<grid_size2, block_size>>>(hidden, nn->weights2,
output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
CUDA_CHECK(cudaDeviceSynchronize()); Launches
```

**the matrix**

| bias_forward_kernel<<<(batch_size 256>>>(output, nn->bias2, | * OUTPUT_SIZE + 255) batch_size, | / 256, multiplication kernel for the |
|---|---|---|
| OUTPUT_SIZE); | Launches the kernel to add | second linear |
| CUDA_CHECK(cudaDeviceSynchronize()); | the second bias vector | layer |
| softmax_kernel<<<batch_size, | 1>>>(output, batch_size, |  |
| OUTPUT_SIZE); |  | Launches the softmax kernel to |
| CUDA_CHECK(cudaDeviceSynchronize()); |  | compute the final probabilities |

```
}
```

Look closely at that softmax launch: <<<batch\_size, 1>>> puts one thread in each block, so each block occupies a full 32-thread warp with 31 lanes idle, and the whole launch keeps only a tiny fraction of the GPU busy. We accept that here because the output layer is just 10 values wide per sample, and our goal is a correct translation, but this is exactly the kind of launch configuration that profiling flags immediately. Chapter 6 shows how to structure reductions like softmax so a full block cooperates on each row. The backward pass follows a similar pattern but in reverse order. We start by computing the output gradients from the softmax and cross-entropy loss, then propagate these gradients backward through each layer. Each gradient computation requires the intermediate values cached during the forward pass, which is why we must carefully manage memory allocation and synchronization.

-----

###### Level5: The naive CUDA approach 81

**Listing 4.25 Naive CUDA backward pass orchestration**

```
void backward_pass(NeuralNetwork *nn, float *input, float *hidden,
float *output, int *labels, int batch_size) {
```

`dim3 block_size(32, 32);` **Computes the initial gradient from**

**the softmax and cross-entropy loss**

```
float *grad_output, *dX2, *d_ReLU_out;
CUDA_CHECK(cudaMalloc(&grad_output, batch_size * OUTPUT_SIZE *
sizeof(float)));
CUDA_CHECK(cudaMalloc(&dX2, batch_size * HIDDEN_SIZE * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_ReLU_out, batch_size * HIDDEN_SIZE *
sizeof(float)));
compute_output_gradients_kernel<<<(batch_size + 255) / 256, 256>>>(
grad_output, output, labels, batch_size);
CUDA_CHECK(cudaDeviceSynchronize());
dim3 grid_weights2((OUTPUT_SIZE + block_size.x - 1) / block_size.x, (
HIDDEN_SIZE + block_size.y - 1) / block_size.y);
matmul_at_b_kernel<<<grid_weights2, block_size>>>(hidden, grad_output,
nn->grad_weights2, batch_size, HIDDEN_SIZE,
```

| OUTPUT_SIZE); | Calculates the weight gradients for the |
|---|---|
| CUDA_CHECK(cudaDeviceSynchronize()); | second layer (hidden.T @ grad_output) |

```
bias_backward_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->grad_bias2,
grad_output, batch_size, OUTPUT_SIZE);
CUDA_CHECK(cudaDeviceSynchronize()); Calculates the bias gradients
```

**for the second layer**

```
dim3 grid_hidden((HIDDEN_SIZE + block_size.x - 1) / block_size.x,
(batch_size + block_size.y - 1) / block_size.y);
matmul_a_bt_kernel<<<grid_hidden, block_size>>>(grad_output, nn->
weights2, dX2, batch_size, OUTPUT_SIZE,
```

| HIDDEN_SIZE); | Propagates the gradients back to the ReLU |
|---|---|
| CUDA_CHECK(cudaDeviceSynchronize()); | layer's input (grad_output @ weights2.T) |

```
relu_backward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(
dX2, hidden, batch_size * HIDDEN_SIZE);
CUDA_CHECK(cudaMemcpy(d_ReLU_out, dX2, batch_size * HIDDEN_SIZE *
sizeof(float), cudaMemcpyDeviceToDevice)); Applies the ReLU
CUDA_CHECK(cudaDeviceSynchronize()); derivative to the gradients
dim3 grid_weights1((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (
INPUT_SIZE + block_size.y - 1) / block_size.y);
matmul_at_b_kernel<<<grid_weights1, block_size>>>(input, d_ReLU_out,
nn->grad_weights1, batch_size, INPUT_SIZE,
```

| HIDDEN_SIZE); | Calculates the weight gradients for the |
|---|---|
| CUDA_CHECK(cudaDeviceSynchronize()); | first layer (input.T @ d_ReLU_out) |

```
bias_backward_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_bias1,
d_ReLU_out, batch_size, HIDDEN_SIZE);
CUDA_CHECK(cudaDeviceSynchronize()); Calculates the bias
```

**gradients for the first layer**

```
CUDA_CHECK(cudaFree(grad_output));
```

-----

###### 82 Chapter 4 Training and verifying neural nets in raw CUDA

```
CUDA_CHECK(cudaFree(dX2));
CUDA_CHECK(cudaFree(d_ReLU_out));
}
```

Notice also that backward\_pass calls cudaMalloc for three scratch buffers on entry and

```
cudaFree on exit, which means every training iteration pays for device allocation and
```

deallocation. Device memory allocation is expensive and can synchronize the device, so production code allocates scratch space once up front and reuses it across iterations, just as we already do for the persistent gradient buffers in the NeuralNetwork struct. We keep the per-call allocations here so the listing stays self-contained.

### 4.5.5 A note on profiling: Wall-clock vs. GPU time

A critical concept in CUDA development is understanding how to measure performance accurately. Because GPU operations are asynchronous (the CPU launches a kernel and immediately moves on), using CPU-side timers like clock\_gettime() around a block of kernel launches can be misleading. The CPU finishes launching the kernels very quickly, but the GPU may still be working. A final cudaDeviceSynchronize() call then forces the CPU to wait for all GPU work to complete, making that single line appear to take a very long time.

This "wrong timing attribution" is a common pitfall. The correct way to measure the time a kernel takes on the GPU is with CUDA Events. By recording an event before and after a kernel launch and then calculating the elapsed time between them, we can get a precise measurement of the GPU's execution time, separate from any CPU or memory transfer overhead. The following listing shows a simple timing utility using CUDA Events.

**Listing 4.26 CUDA Events timing utility**

```
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
my_kernel<<<grid, block>>>(args); // kernel to time
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel time: %.3f ms\n", milliseconds);
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

Although we use simple wall-clock time in this naive version for simplicity, you should use CUDA Events for any serious performance measurement. The wall-clock approach works here because we call cudaDeviceSynchronize() after each kernel, but this adds

-----

###### Level6: The cuBLAS approach 83

unnecessary overhead in production code. The following listing shows the training output from our naive CUDA implementation.

**Listing 4.27 Naive CUDA timing results**

```
Epoch 0 loss: 0.4653
Epoch 1 loss: 0.2369
Epoch 2 loss: 0.1759
Epoch 3 loss: 0.1370
Epoch 4 loss: 0.1095
Epoch 5 loss: 0.0887
Epoch 6 loss: 0.0725
Epoch 7 loss: 0.0596
Epoch 8 loss: 0.0493
Epoch 9 loss: 0.0409
Total training time: 2.9 seconds
Detailed Breakdown:
```

| Data loading: | 0.220s ( 7.6%) |
|---|---|
| Forward pass: | 1.200s ( 41.5%) |
| Loss computation: | 0.004s ( 0.1%) |
| Backward pass: | 0.784s ( 27.1%) |
| Weight updates: | 0.222s ( 7.7%) |

**TIP** Our naive CUDA implementation successfully parallelizes the MLP training logic. Although it's much faster than the C version, its performance is still limited by unoptimized memory access patterns and host-device communication bottlenecks. It is a working baseline that we can now dramatically improve.

## 4.6 Level 6: The cuBLAS approach

Our naive CUDA implementation is a great academic exercise, but in practice, no one writes their own matrix multiplication kernels for deep learning. NVIDIA provides highly optimized, hand-tuned libraries for these exact operations. The three most important are the following:

¡ *cuBLAS (CUDA Basic Linear Algebra Subroutines)-A library for dense linear alge-*

bra operations like matrix multiplication. ¡ *cuDNN (CUDA Deep Neural Network library)-A higher-level library with optimized*

implementations for common DNN operations like convolutions, pooling, and activation functions. ¡ *CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers)-A collection*

of CUDA C++ template abstractions for implementing high-performance matrix operations. Unlike cuBLAS, CUTLASS provides source code you can customize; on newer architectures like Hopper and Blackwell, a hand-tuned CUTLASS kernel can match or beat a generic cuBLAS call, especially for shapes or fusions cuBLAS does not specialize for.

-----

###### 84 Chapter 4 Training and verifying neural nets in raw CUDA

cuBLAS and cuDNN are opaque; we don't see their source code. NVIDIA's engineers have spent years tuning these libraries for each GPU generation. They are delivered as compiled shared object files (files with a .so suffix on Linux). For our MLP, which relies heavily on matrix multiplication, the cuBLAS library is a perfect fit. The cuDNN library is more geared toward complex architectures like convolutional neural networks (CNNs), recurrent neural networks (RNNs), and Transformers. For brevity and to focus on the core principles, we will explore only cuBLAS in this chapter. It has a much simpler API and serves as a powerful example of how to use NVIDIA's libraries to accelerate our code with minimal effort. We will cover CUTLASS in a later chapter because of its importance for modern GPU architectures.

### 4.6.1 Integrating cuBLAS and optimizing the training loop

The v5.cu script demonstrates how to integrate cuBLAS. First, we include the header, add a CUBLAS\_CHECK macro for error handling, and create a cublasHandle\_t, which is passed to all cuBLAS functions.

| Listing 4.28 Initializing cuBLAS and error handling |
|---|
| #include <cublas_v2.h> #define CUBLAS_CHECK(call) \\ do { \\ cublasStatus_t status = call; \\ if (status != CUBLAS_STATUS_SUCCESS) { \\ fprintf(stderr, "cuBLAS error at %s:%d: %d\\n", __FILE__, __LINE__, status); \\ exit(EXIT_FAILURE); \\ } \\ } while(0) cublasHandle_t cublas_handle; CUBLAS_CHECK(cublasCreate(&cublas_handle)); The core of the optimization is replacing our manual matrix multiplication kernels with calls to cublasSgemm. Listing 4.29 shows the forward pass, now using cublasSgemm for both linear layers. Note that the other operations, such as adding the bias and applying the ReLU activation, are still handled by our custom kernels. This hybrid approach uses NVIDIA's optimized library for the computationally intensive matrix operations while maintaining control over the simpler element-wise operations. |
| Listing 4.29 Forward pass with cuBLAS |

```
void forward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
const float alpha = 1.0f, beta = 0.0f;
CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
HIDDEN_SIZE, batch_size, INPUT_SIZE,
&alpha, nn->d_weights1, HIDDEN_SIZE,
```

-----

###### Level6: The cuBLAS approach

```
int total_hidden = batch_size
int grid_hidden = (total_hidden
bias_add_kernel<<<grid_hidden,
batch_size, HIDDEN_SIZE);
relu_kernel<<<grid_hidden, 256>>>(
nn->d_fc1_output, total_hidden);
CUBLAS_CHECK(cublasSgemm(nn->cublas_handle,
int total_out = batch_size
int grid_out = (total_out
bias_add_kernel<<<grid_out, 256>>>(nn->d_fc2_output,
batch_size, OUTPUT_SIZE);
CUDA_CHECK(cudaDeviceSynchronize());
}
```

The backward pass is similarly refactored in the following listing to use cublasSgemm for gradient calculations that involve matrix multiplications, using the CUBLAS\_OP\_T (transpose) option to handle the required matrix transpositions without separate kernels.

**Listing 4.30 Backward pass with cuBLAS**

```
void backward_pass_only(NeuralNetworkCUDA *nn,
const float alpha = 1.0f, beta
CUDA_CHECK(cudaMemset(nn->d_grad_weights1, 0, INPUT_SIZE *
HIDDEN_SIZE * sizeof(float)));
CUDA_CHECK(cudaMemset(nn->d_grad_weights2, 0, HIDDEN_SIZE *
OUTPUT_SIZE * sizeof(float)));
CUDA_CHECK(cudaMemset(nn->d_grad_bias1,
CUDA_CHECK(cudaMemset(nn->d_grad_bias2,
CUBLAS_CHECK(cublasSgemm(nn->cublas_handle,
OUTPUT_SIZE, HIDDEN_SIZE, batch_size,
&alpha, nn->d_grad_output,
nn->d_fc1_output, HIDDEN_SIZE, &beta,
nn->d_grad_weights2,
OUTPUT_SIZE));
int total_out = batch_size
int grid_out = (total_out + 255)
bias_backward_kernel<<<grid_out, 256>>>(nn->d_grad_output,
nn->d_grad_bias2, batch_size, OUTPUT_SIZE);
```

```
nn->d_input_batch,
nn->d_fc1_output,
HIDDEN_SIZE));
* HIDDEN_SIZE;
+ 255) / 256;
256>>>(nn->d_fc1_output,
```

```
OUTPUT_SIZE, batch_size, HIDDEN_SIZE,
```

| &alpha, nn->d_fc2_output, | nn->d_weights2, |
|---|---|
| OUTPUT_SIZE)); |  |
| * + 255) / 256; | OUTPUT_SIZE; |
|  |  |

```
INPUT_SIZE, &beta,
```

###### First linear layer: fc1_output = 1.0 * (input @ weights1) + 0.0 * fc1_output

```
nn->d_bias1,
```

###### Adds the bias vector for the first layer Applies the ReLU activation function

```
CUBLAS_OP_N, CUBLAS_OP_N,
OUTPUT_SIZE,
```

###### Second linear layer: fc2_output = 1.0 * (relu_output @ weights2) + 0.0 * fc2_output

```
nn->d_bias2,
```

###### Adds the bias vector for the second layer

```
int batch_size) {
= 0.0f;
```

```
0, HIDDEN_SIZE * sizeof(float)));
0, OUTPUT_SIZE * sizeof(float)));
CUBLAS_OP_N, CUBLAS_OP_T,
OUTPUT_SIZE,
```

###### Calculates the weight gradients for the second layer

`* OUTPUT_SIZE;` **(grad\_output @ fc1\_output.T)**

```
/ 256;
```

###### Calculates the bias gradients for the second layer

-----

###### 86 Chapter 4 Training and verifying neural nets in raw CUDA

```
CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
```

```
int total_hidden = batch_size * HIDDEN_SIZE;
int grid_hidden = (total_hidden + 255) / 256; Applies the ReLU
relu_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_hidden, derivative to
nn->d_fc1_output, total_hidden); the gradients
CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
```

```
bias_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_hidden,
nn->d_grad_bias1, batch_size, HIDDEN_SIZE);
```

`}` **Calculates the bias**

### 4.6.2 Dissecting the cublasSgemm API call

The cublasSgemm function is the workhorse of the cuBLAS library. Its name stands for *single-precision general matrix multiplication. It computes the following operation:*

```
C = alpha * op(A) @ op(B) + beta * C
```

Here, op(X) can be either the matrix X or its transpose, X.T. The function signature is dense, but each parameter has a clear purpose, as detailed in the following listing.

```
Listing 4.31 cublasSgemm function signature
cublasStatus_t cublasSgemm(cublasHandle_t handle,
```

```
¡ handle: The cuBLAS library context, created once with cublasCreate().
¡ transa/transb: Flags that specify whether the input matrices A and B should be
```

transposed. We use CUBLAS\_OP\_N for no transpose and CUBLAS\_OP\_T for transpose.

```
¡ m, n, k: The dimensions of the matrices. op(A) is an m x k matrix, op(B) is a k x n
```

matrix, and C is an m x n matrix.

```
HIDDEN_SIZE, batch_size, OUTPUT_SIZE,
&alpha, nn->d_weights2, OUTPUT_SIZE,
nn->d_grad_output, OUTPUT_SIZE, &beta,
nn->d_grad_hidden,
```

`HIDDEN_SIZE));` **Propagates the gradients back to the ReLU**

**layer's input (weights2.T @ grad\_output)**

```
HIDDEN_SIZE, INPUT_SIZE, batch_size,
&alpha, nn->d_grad_hidden, HIDDEN_SIZE,
nn->d_input_batch, INPUT_SIZE, &beta,
nn->d_grad_weights1,
```

`HIDDEN_SIZE));` **Calculates the weight gradients for the**

**first layer (grad\_hidden @ input.T)**

**gradients for the first layer**

```
cublasOperation_t transa,
cublasOperation_t transb,
int m, int n, int k,
const float *alpha,
const float *A, int lda,
const float *B, int ldb,
const float *beta,
float *C, int ldc);
```

-----

###### Level6: The cuBLAS approach 87

```
¡ alpha/beta: Scalar floating-point values that scale the product and the existing
C matrix, respectively. Setting beta to 0.0f makes the operation a simple matrix
```

multiplication.

```
¡ A, B, C: Device pointers to the matrices in GPU memory.
¡ lda, ldb, ldc: The "leading dimension" of each matrix. This is a crucial but often
```

confusing parameter. It specifies the distance in memory between the start of two consecutive columns (for column-major layout) or rows (for row-major layout). For a standard, contiguous matrix, this is simply the number of rows. A critical detail is that cuBLAS, like Fortran and the BLAS libraries it's based on, assumes matrices are stored in column-major order. C and C++, by contrast, use row*major order. A common and efficient trick for handling this mismatch is to reverse the* order of the matrix multiplication (B @ A instead of A @ B) and swap the dimensions

```
m and n. This works because (A @ B).T = B.T @ A.T, and transposing a matrix in memory
```

is equivalent to switching its layout from row-major to column-major. Our code uses this trick implicitly, which is why the cublasSgemm calls may look slightly different from the mathematical formulas they represent. Finally, the weight update step can be optimized using the cuBLAS cublasSaxpy function, which efficiently computes Y = alpha\*X + Y. This function is a perfect fit for the gradient descent update rule (weights = weights - lr \* grad\_weights), as shown in the following listing.

**Listing 4.32 Weight updates with cublasSaxpy**

```
void update_weights_only(NeuralNetworkCUDA *nn, float lr) {
```

`float neg_lr = -lr;` **Updates the weights of the first layer**

```
CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE,
&neg_lr, nn->d_grad_weights1, 1,
nn->d_weights1, 1));
CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE * OUTPUT_SIZE,
&neg_lr, nn->d_grad_weights2, 1,
```

`nn->d_weights2, 1));` **Updates the weights**

```
CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE, of the second layer
&neg_lr, nn->d_grad_bias1, 1,
```

`nn->d_bias1, 1));` **Updates the biases**

```
CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, OUTPUT_SIZE, of the first layer
&neg_lr, nn->d_grad_bias2, 1,
```

`nn->d_bias2, 1));` **Updates the biases**

**of the second layer**

```
CUDA_CHECK(cudaDeviceSynchronize());
}
```

### 4.6.3 Debugging gradient instability

Even with an optimized implementation, you might encounter training problems such as "exploding" or "vanishing" gradients. A useful debugging technique is to

-----

###### 88 Chapter 4 Training and verifying neural nets in raw CUDA

periodically copy the gradient tensors from the GPU back to the host and analyze their magnitudes. If the average magnitude of a gradient tensor is growing exponentially, you have exploding gradients. If it's shrinking to near zero, the gradients are vanishing. The following technique can help you pinpoint which layer is causing instability.

| Listing 4.33 Analyzing gradient magnitudes |
|---|
| Naive gradients: Range [-6.869578e-04, 7.698272e-04], Avg \|grad\| = 1.877392e-04 cuBLAS gradients: Range [-5.890325e-04, 6.289262e-04], Avg \|grad\| = 8.478115e-05 The performance improvement from using cuBLAS is dramatic. The final perfor- mance, shown in listing 4.34, is over 100 times faster than our C implementation and roughly twice as fast as our naive CUDA version (2.9s → 1.4s). |
| Listing 4.34 cuBLAS timing results |

```
Epoch 0 loss: 0.4634
Epoch 1 loss: 0.2364
Epoch 2 loss: 0.1764
Epoch 3 loss: 0.1373
Epoch 4 loss: 0.1095
Epoch 5 loss: 0.0888
Epoch 6 loss: 0.0727
Epoch 7 loss: 0.0600
Epoch 8 loss: 0.0497
Epoch 9 loss: 0.0415
Total training time: 1.4 seconds
Detailed Breakdown:
```

| Data loading: | 0.254s ( 17.6%) |
|---|---|
| Forward pass: | 0.466s ( 32.2%) |
| Loss computation: | 0.024s ( 1.7%) |
| Backward pass: | 0.466s ( 32.2%) |
| Weight updates: | 0.233s ( 16.1%) |

**TIP** Although writing custom kernels is a powerful skill, you should use optimized libraries like cuBLAS for standard operations whenever possible. They provide a massive performance boost and let you focus your efforts on the novel parts of your application. This cuBLAS implementation is now our final, high-performance, and fully verified baseline.

### 4.6.5 A necessary lie and the path forward

Earlier, we said that "no one writes their own matrix multiplication kernels for deep learning." This was a necessary simplification. The truth is, people do write their own

-----

###### Summary 89

kernels all the time, especially for novel operations that aren't common enough to be included in libraries like cuBLAS or cuDNN. But for something as fundamental and ubiquitous as matrix multiplication, the statement is mostly true. NVIDIA has put thousands of engineering hours into hand-tuning its GEMM kernels. There is an important caveat: although cuBLAS is excellent and well tested, CUT- LASS is often the fastest option for newer GPU architectures like Hopper and Blackwell. CUTLASS provides source-level templates that can be tuned for specific use cases and take advantage of the latest hardware features before cuBLAS catches up. For production deep learning systems targeting the latest hardware, CUTLASS has become the de facto standard. We will explore CUTLASS in detail in a later chapter. That said, beating these optimized implementations from scratch is an extraordinary feat. This is why we almost always default to using NVIDIA's libraries for standard operations. We will explore the art of hand-optimizing kernels in a future chapter, where we'll see just how complex state-of-the-art CUDA code can become. The real takeaway from this chapter is the sheer complexity involved in implementing, debugging, and optimizing even a simple two-layer MLP at the systems level. Our journey from a few lines of PyTorch to hundreds of lines of CUDA and cuBLAS code illustrates a crucial point: as our models grow more complex, managing this low level of detail becomes untenable. The training principles you've learned here-forward passes, backpropagation, gradient updates-apply directly to more complex architectures. When you train a transformer, a diffusion model, or any other architecture, the same chain rule and the same optimization loop underpin everything. But training is only half the story. In production, inference is where performance engineering becomes critical. Generating tokens one at a time, managing KV caches, and optimizing for latency rather than throughput: these challenges require different CUDA strategies. In the next chapter, we'll tackle inference for the Transformer architecture.

###### Summary

¡ A complete MNIST classifier can be built from high-level PyTorch down to optimized CUDA code, with correctness verified at each abstraction level. ¡ A reproducible PyTorch implementation serves as a "gold standard" for verifying that lower-level implementations (NumPy, C, CUDA) train correctly, with loss curves that closely track the baseline. ¡ Backpropagation in a two-layer MLP requires caching forward pass activations, computing gradients layer by layer using the chain rule, and handling numerical stability in softmax. ¡ CUDA kernels replace nested CPU loops: each thread computes one output element, using blockIdx and threadIdx to calculate its unique position in the output tensor.

-----

###### 90 Chapter 4 Training and verifying neural nets in raw CUDA

¡ GPU execution is asynchronous with respect to the CPU. Wall-clock timing from

the host measures launch overhead and synchronization, not actual kernel execution time. CUDA events provide accurate GPU timing. ¡ NVIDIA's cuBLAS library provides highly optimized matrix multiplication

(cublasSgemm) that runs about 2× faster than naive CUDA kernels and over 100× faster than single-threaded C code.

```
¡ cublasSgemm uses column-major layout by default. When working with row-major
```

data (standard in C/CUDA), the matrix dimensions and transpose flags must be adjusted accordingly.

-----

# 5

## Transformer inference in CUDA

---

| This chapter covers |
|---|
| ¡ Understanding the decoder-only transformer architecture that powers GPT-style models |
| ¡ The fundamental split between prefill (processing |
| the prompt) and decode (generating tokens) |
| ¡ Building naive CUDA kernels for inference: GEMV, |
| softmax, layer normalization, and attention |
| ¡ Key-value caching to avoid redundant |
| computation during autoregressive generation |
| ¡ Mixture-of-experts routing with custom top-k |
| kernels |
| ¡ The CUDA-Python binding pipeline that connects |
| our kernels to PyTorch |

Chapter 4 gave you the complete training story: forward passes, backpropagation, gradient updates. Those principles apply to any architecture. Inference, which generates text one token at a time, has a different profile. Batch sizes shrink to one.

-----

###### 92 Chapter 5 Transformer inference in CUDA

Matrix multiplications become matrix-vector products. Memory bandwidth, not compute throughput, becomes the bottleneck.

## 5.1 Getting started: Transformer architecture and setup

Before writing inference kernels, we need to understand what we're optimizing. This section establishes the transformer architecture, the terminology we'll use throughout, and the PyTorch baseline that our CUDA kernels must match numerically.

### 5.1.1 Establishing the baseline scenario

Inference works fundamentally differently from training. During training, we process entire sequences in parallel: all 64 tokens flow through attention simultaneously. During inference, we generate one token at a time, with each token depending on all previous tokens. This sequential dependency changes how we write CUDA kernels. Our reference implementation uses a sequence length of 64, an embedding dimension of 128, 4 attention heads, and 8 transformer layers. These dimensions are small enough to understand clearly while still being realistic enough to demonstrate the optimization opportunities that become critical in production systems. **NOTE** Two configurations appear in this chapter. The walk-through listings and diagrams use the small dimensions above (embedding 128, 4 heads, 8 layers) so the mechanics stay legible. The end-to-end inference. py comparison near the end of the chapter loads a larger deployment-scale checkpoint (embedding 768, 8 heads, 24 layers) to make the CUDA-versus- PyTorch timing meaningful. The kernels are identical; only the tensor shapes change. Throughout this chapter, diagrams use smaller dimensions (e.g., 4 tokens, 2 layers) for visual clarity, since we can't draw 64-token sequences or 8-layer networks on a page. But the actual code runs at significantly larger scales to surface the inefficiencies and optimization opportunities that matter in practice. Think about training on a 64-token sequence. We already know all the tokens, so we can compute attention between any two positions simultaneously. Position 32 can attend to positions 0 to 31 while position 40 attends to positions 0 to 39. This parallel processing is what makes training a GEMM-dominated workload.

**The reference implementation: A complete transformer codebase**

Throughout this chapter, we'll walk through a complete transformer implementation that demonstrates the practical integration of custom CUDA kernels with PyTorch. The 3\_transformer/ codebase provides both training and inference components, showing the hybrid approach, in which CUDA accelerates specific operations while PyTorch handles the overall framework.

The project structure follows a modular design:

-----

***Getting started: Transformer architecture and setup*** **93**

```
3_transformer/
├── csrc/
```

| │ | ├── | training/ |
|---|---|---|
| │ | │ | ├── binding.cpp |
| │ | │ | └── kernels/ |
| │ | │ | ├── matmul.cu |
| │ | │ | ├── elementwise.cu |
| │ | │ | ├── activation.cu |
| │ | │ | ├── softmax.cu |
| │ | │ | ├── layernorm.cu |
| │ │ | │ | └── embedding.cu └── inference/ |
| │ │ |  | ├── binding.cpp └── kernels/ |
| │ │ |  | ├── matmul_fwd.cu ├── gemv_fwd.cu |
| │ │ |  | ├── elementwise_fwd.cu ├── activation_fwd.cu |
| │ │ |  | ├── softmax_fwd.cu ├── layernorm_fwd.cu |
| │ | ├── wrapper/ | └── topk_fwd.cu |
| │ │ |  | ├── training/ └── inference/ |

```
├── train.py
├── inference.py
├── setup.py
├── README.md
└── wizard_of_oz.txt
```

The codebase shows how custom CUDA kernels slot into a PyTorch project: separate kernel implementations for training and inference, contiguous tensor checks for CUDA compatibility, and verification that our CUDA outputs match PyTorch exactly. We'll reference these files throughout the chapter, showing how theoretical concepts translate into working code.

###### The binding pipeline: Connecting CUDA to Python

The pipeline that makes CUDA code callable from Python has three layers: CUDA kernels, C++ bindings, and Python wrappers.

The journey starts with setup.py, which uses PyTorch's extension build system.

**Listing 5.1 Building CUDA extensions with setup.py**

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(
name='naive-cu-extensions',
ext_modules=[
CUDAExtension(
name='custom_inference_extension',
```

-----

###### 94 Chapter 5 Transformer inference in CUDA

```
sources=[
'csrc/inference/binding.cpp',
'csrc/inference/kernels/matmul_fwd.cu',
'csrc/inference/kernels/gemv_fwd.cu',
# ... other kernel files
],
extra_compile_args={'nvcc': ['-O3']}
)
```

```
],
cmdclass={'build_ext': BuildExtension}
)
```

###### The binding file comes first; it defines the Python-visible interface.

```
CUDAExtension handles all the complexity of compiling CUDA code: finding the right
```

compiler, setting include paths, and linking against PyTorch's C++ libraries. Run pip

```
install -e ., and it produces a Python module you can import directly.
```

The binding file is the bridge between C++ and Python. It uses PyTorch's extension header and pybind11 to expose functions.

**Listing 5.2 The binding file structure**

###### Provides torch::Tensor, error-checking macros, and pybind11 integration

`#include <torch/extension.h>` **Forward declarations of functions**

**implemented in .cu files**

```
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B);
torch::Tensor gemv_forward(torch::Tensor A, torch::Tensor x);
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor w,
torch::Tensor b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("matmul_forward", &matmul_forward, "Matrix multiplication");
m.def("gemv_forward", &gemv_forward, "Matrix-vector multiplication");
m.def("layernorm_forward", &layernorm_forward, "Layer normalization");
}
```

###### Creates the Python module; TORCH_ EXTENSION_NAME comes from setup.py.

The torch/extension.h header is the key include. It brings in

```
¡ torch::Tensor-PyTorch's tensor type in C++, with the same memory layout as
```

Python tensors

```
¡ TORCH_CHECK-Macro for runtime assertions that raise Python exceptions
```

¡ `pybind11-The` library that generates Python bindings from C++ function

signatures Inside each .cu file, the actual kernel implementations need one more piece: device context management.

-----

***Getting started: Transformer architecture and setup*** **95**

```cpp
Listing 5.3 CUDAGuard ensures the correct device context
#include <c10/cuda/CUDAGuard.h>
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
c10::cuda::CUDAGuard device_guard(A.device());
auto C = torch::empty({A.size(0), B.size(1)}, A.options());
```

`// Launch kernel on the correct device` **Sets the CUDA context to match the input** `matmul_kernel<<<grid, block>>>(` **tensor's device for multi-GPU systems**

```
A.data_ptr<float>(),
B.data_ptr<float>(),
C.data_ptr<float>(),
A.size(0), A.size(1), B.size(1)
);
return C;
}
CUDAGuard is essential for multi-GPU systems. When you have multiple GPUs, CUDA
```

operations run on the "current" device. If tensor A lives on GPU 1 but the current device is GPU 0, the kernel would launch on the wrong device and crash. The guard sets the current device to match the input tensor and then restores the previous device when the function returns.

The final layer is a Python wrapper that provides a clean interface.

**Listing 5.4 Python wrapper for clean API**

```
import custom_inference_extension as ext
```

`def matmul(A, B):` **Ensures a row-major memory**

`A = A.contiguous()` **layout before calling C++ code**

```
B = B.contiguous()
return ext.matmul_forward(A, B)
```

The wrapper enforces invariants such as contiguous memory layout (which we'll discuss more later) and provides a Pythonic interface. This three-layer structure-kernel, binding, wrapper-keeps each component focused: kernels handle computation, bindings handle type conversion, and wrappers handle API design.

**Dataset and tokenization: Character-level language modeling**

Our training example uses The Wonderful Wizard of Oz by L. Frank Baum, a publicdomain novel that's perfect for character-level language modeling. Character-level tokenization treats each character as a separate token, creating a vocabulary of 65 unique characters (letters, punctuation, spaces, etc.). The following listing demonstrates how to load and tokenize the dataset.

-----

###### 96 Chapter 5 Transformer inference in CUDA

**Listing 5.5 Loading and tokenizing the Wizard of Oz dataset**

```python
import torch
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
```

With block\_size = 64, the model predicts each character from at most the 64 characters before it.

###### Architecture overview: From text to gradients

We are building a decoder-only transformer, the architecture behind GPT and most modern language models. Unlike encoder-decoder models (used for translation) or encoder-only models (used for classification), decoder-only transformers generate text autoregressively: they predict one token at a time, each prediction conditioned only on previous tokens.

First, let's establish the terminology we'll use throughout this chapter: ¡ *Head-A single attention computation that projects inputs to queries, keys, and*

values, computes attention scores, and produces an output. Each head learns to focus on different relationships between tokens. ¡ *Multihead attention-Multiple* heads run in parallel, and their outputs are

concatenated and projected back to the model dimension. This allows the model to attend to information from different representation subspaces simultaneously. ¡ *Block (or layer)-One complete transformer unit containing multihead attention,*

a feed-forward network, and the residual connections and layer normalization that wrap them. Our model stacks eight of these blocks. ¡ *Model-The complete transformer: token embeddings, position embeddings, a*

stack of blocks, final layer normalization, and the output projection to vocabulary logits (the raw, unnormalized scores for each token in the vocabulary before softmax converts them to probabilities). We also need to be precise about tensor dimensions. Throughout this chapter, our

```
primary activation tensor has shape (batch, sequence_length, embedding_dim) or
(B, T, C) in code:
¡ B (batch)-Number of sequences processed in parallel (16 in our setup)
¡ T (sequence length)-Number of tokens per sequence (64 in our setup)
```

-----

***Getting started: Transformer architecture and setup*** **97**

```
¡ C (channels/embedding dimension)-Size of each token's representation (128 in our
```

setup)

With this vocabulary established, let's trace the complete training flow. We start with a sequence of tokens, process them through multiple transformer blocks, compute a loss against the ground truth, and backpropagate gradients to update the model. Figure 5.1 shows the complete architecture, with data flowing through multiple transformer blocks.

###### Entire model

Say we have n\_layers set to 3 (3 decoder blocks)

---

Embeddings

+ Pos Decoder Decoder Decoder nn.Linear

encoding

---

Tokenized inputs Probs sampling + Softmax (probs

generation to sample from)

Compare targets

to inputs → backpropagate

**Figure 5.1 Complete transformer architecture showing training context**

The complete architecture shows how multiple transformer blocks stack together. Each block processes the sequence independently, applying self-attention to gather information across tokens and then using feed-forward networks to transform each position's representation. The residual connections (shown as bypass paths) allow gradients to flow directly through the network during training, preventing the vanishing-gradient problem in deep networks.

Let's zoom in on a single transformer block to understand its internal structure. Figure 5.2 details the components within a single block.

Each transformer block consists of two main sublayers. The first is multihead selfattention, which allows each token to attend to all previous tokens in the sequence (causal masking ensures that we don't look at future tokens during training). The second is a position-wise feed-forward network that processes each token independently. Both sublayers are wrapped with residual connections and layer normalization. In the

-----

###### 98 Chapter 5 Transformer inference in CUDA

---

Residual

| Multihead | connection | Feed-forward |
|---|---|---|
| attention | (add) then |  |

normalize

Residual connection then normalize

---

Key (K) Linear GELU Linear

n heads Scaled dot- Concatenate

Query (Q) product results Linear

attention

###### We call it multihead attention

Value (V) **because there are a lot of heads**

**learning different semantic info**

Scaled dot-product attention runs n heads in parallel. **from a unique perspective.**

---

| Figure 5.2 | Single transformer block architecture |
|---|---|
|  | implementation we use here, each sublayer sees a normalized input first, and then its |
|  | output is added back to the residual stream. This structure repeats identically across all |
|  | layers, with different learned parameters. |
|  | Now let's examine the core attention mechanism that makes transformers powerful. |
|  | Figure 5.3 illustrates the scaled dot-product attention computation. |

---

K

| Dot | Scale by | Causal mask |
|---|---|---|
| product(K, Q) | 1/sqrt(d_k) | (torch.tril) |

Q

V

---

**Figure 5.3 Scaled dot-product attention mechanism**

###### OUTPUT: blend of input

Softmax Matrix multiply **vector values**

**and attention placed on each token**

Scaled dot-product attention (SDPA) is the fundamental operation inside each attention head. The input is projected into three representations: queries (Q), keys (K), and values (V). We compute attention scores by taking the dot product of queries and keys, then scale by the square root of the dimension before applying softmax. Without scaling, the dot products grow with the embedding dimension: if each element of Q and K has roughly unit variance, the dot product of two d-dimensional vectors has variance

-----

***Getting started: Transformer architecture and setup*** **99**

proportional to d. Large dot products push softmax into regions where gradients vanish (one element dominates, and all others become negligible). Dividing by sqrt(d) keeps the variance stable regardless of dimension, preventing these extreme peaks and maintaining healthy gradient flow. After scaling, we apply a causal mask to enforce autoregressive behavior. The mask is a lower-triangular matrix (torch.tril): the lower triangle, including the diagonal, contains ones, and the upper triangle contains zeros. When we mask positions, we set upper-triangle scores to negative infinity before softmax. After exponentiation, these scores become zero probability: position 5 can attend to positions 0-5, but not 6 or beyond. The lower triangle represents the past, or valid attention targets, while the upper triangle represents the future, or forbidden targets. This is what makes the model autoregressive: each prediction depends only on what came before. Finally, softmax converts scores to probability distributions, and these probabilities weight the values to produce the output. During training, we compute this for all sequence positions in parallel: the full T × T attention matrix at once. Training transforms our problem into a series of matrix operations where every position in the sequence can be computed in parallel. This is why training is dominated by GEMM operations rather than the GEMV operations we'll see during inference.

**Tokenization and embedding: Setting up the sequence**

During training, tokenization happens once per training example, converting our input text into a sequence of token IDs. Figure 5.4 shows this conversion from raw text to integer IDs.

The embedding table then

x = ["the cat sat on the"] transforms these discrete IDs into dense vectors that flow through the network. Figure 5.5 demon- x = [...all tokens in the conversation as str....] strates the embedding lookup pro-

**Figure 5.4 Tokenization process that breaks text into**

cess: each token ID indexes into a

**processable tokens**

learned matrix, retrieving a dense vector representation. Let's build the PyTorch transformer from the ground up, starting with the configuration and progressively adding components. Each piece will connect to form our complete training baseline. The following listing shows the configuration parameters that define our model architecture.

**Listing 5.6 Transformer configuration parameters**

```
batch_size = 16
block_size = 64
n_embd = 128
n_head = 4
```

-----

###### 100 Chapter 5 Transformer inference in CUDA

```
n_layer = 8
vocab_size = 65
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Tokenize(x) x = ["the cat sat on the"] x = [14, 84, 13, 60, 14]

BUT HOW???

Embedding Embedding table embedding = embedding\_table[integer\_token]

dim

---

Input

| 0 1 2 | [-0.2152, 1.0791, -1.1753, -0.4624 ...] [-0.5185, 1.8961, -0.2062, -1.3574 ...] [-0.5584, 0.7408, 0.0704, -1.0846] |  |
|---|---|---|
| 3 | [ 2.1894, -0.4479, 0.0602, -0.8233] | Vocabulary |
| 4 | [ 1.3329, 0.0457, 0.1548, -0.1117], | size |

| embeddings | 5 6 | [-0.8559, 0.0959, 1.4546, 0.8496], [ 0.1901, -0.0831, -0.9108, -1.8238], |
|---|---|---|
|  | 7 . | [-0.9445, 1.0996, -0.3786, -0.3813], [ 0.5265, -0.5278, -1.0180, 0.2370], |

.

. [ 0.4477, -0.4230, 0.6624, -2.2098]

Embedding dim (all vectors are the same length)

---

**Figure 5.5 The embedding table lookup transforms token IDs into dense vectors.**

The configuration sets up our training dimensions. The 8-layer depth is deliberately chosen to make optimization opportunities visible: inefficiencies that seem minor in shallow models compound significantly as depth increases. With a batch size of 16 and a sequence length of64, we're processing 1,024 tokens simultaneously during training, creating abundant parallelism for GPU execution. The embedding dimension of 128 with 4 attention heads means each head operates in a 32-dimensional subspace, balancing expressiveness with computational efficiency. In train.py, vocab\_size is recomputed from the Wizard of Oz corpus immediately after tokenization. For this dataset, the value is 65, which is why the listing uses that concrete number. Now we'll implement the attention mechanism, starting with a single attention head that computes queries, keys, and values. The following listing shows the initialization.

**Listing 5.7 Attention head initialization**

```python
class PytorchHead(nn.Module):
def __init__(self, head_size):
super().__init__()
```

-----

***Getting started: Transformer architecture and setup*** **101**

```
self.key = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
self.register_buffer("mask", torch.tril(torch.ones(
block_size, block_size)))
```

The constructor creates lightweight projections for queries, keys, and values, plus a causal mask buffer so each head can implement autoregressive attention without recomputing the mask on every call.

The following listing shows the forward pass for the attention head.

| Listing 5.8 Attention head forward pass |
|---|
| class PytorchHead(nn.Module): def forward(self, x): q = self.query(x) k = self.key(x) v = self.value(x) weights = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5) weights = weights.masked_fill( self.mask[:weights.size(-2), :weights.size(-1)] == 0, float("-inf"), ) return weights.softmax(dim=-1) @ v The forward pass projects the inputs into Q, K, V tensors, scales and masks the similar- ity scores, normalizes them with softmax, and blends the values to produce the head output. A single attention head computes three projections (query, key, and value) from the input, calculates attention weights between all positions, and applies those weights to the values. The causal mask ensures that each position can attend only to previous posi- tions, which is critical for autoregressive language modeling. This inference-oriented head omits the dropout that a training configuration would apply to the attention weights because dropout is inactive at inference time. With a single head implemented, we can now combine multiple heads. Each head has its own learned Q, K, and V projection matrices, initialized randomly and updated independently during training. Because each head starts with different weights and receives different gradient signals, they naturally specialize: one might learn to weight adjacent tokens heavily (useful for local syntax), while another might develop broader attention patterns (useful for tracking long-range references like pronouns to their antecedents). This specialization emerges from optimization, not from explicit pro- gramming. The following listing shows the multihead attention initialization. |
| Listing 5.9 Multihead attention initialization |

```python
class PytorchMultiHeadAttention(nn.Module):
def __init__(self, num_heads, head_size):
```

-----

###### 102 Chapter 5 Transformer inference in CUDA

```python
super().__init__()
self.heads = nn.ModuleList([PytorchHead(head_size)
for _ in range(num_heads)])
self.proj = nn.Linear(head_size * num_heads, n_embd)
```

We instantiate one PytorchHead per attention head and stash a projection layer that will collapse the concatenated head outputs back to the model dimension.

The following listing shows the multihead attention forward pass.

| Listing 5.10 Multihead attention forward pass |
|---|
| class PytorchMultiHeadAttention(nn.Module): def forward(self, x): return self.proj(torch.cat([head(x) for head in self.heads], dim=-1)) At runtime, we evaluate every head in parallel, concatenate their contributions, and project the result to the original embedding size so that downstream layers see a single fused tensor. After attention gathers information across the sequence, the feed-forward network processes each position independently to transform the representations. The following listing implements the feed-forward layer initialization. |
| Listing 5.11 Feed-forward network initialization |
| class PytorchFeedForward(nn.Module): def __init__(self, n_embd): super().__init__() self.net = nn.Sequential( nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), ) The constructor wires up the two linear layers and an activation that form the clas- sic transformer feed-forward sublayer (the original transformer used ReLU; GELU became standard with BERT and GPT-2, so we use it here). The next listing shows the feed-forward pass. |
| Listing 5.12 Feed-forward network forward pass |

```python
class PytorchFeedForward(nn.Module):
def forward(self, x):
return self.net(x)
```

Applying the sequential module expands, activates, and projects each token independently, enabling nonlinear mixing after attention.

-----

***Getting started: Transformer architecture and setup*** **103**

The feed-forward network processes each position independently. It expands to 4× the embedding dimension (128 → 512), applies the GELU activation, and then projects back down. These two layers hold twice as many weights as the attention projections in the same block (2 × 128 × 512 versus 4 × 128 × 128). Now we combine attention and feed-forward layers into a complete transformer block with residual connections and layer normalization. The following listing shows the transformer block initialization.

| Listing 5.13 Transformer block initialization |
|---|
| class PytorchBlock(nn.Module): def __init__(self, n_embd, n_head): super().__init__() head_size = n_embd // n_head self.attn = PytorchMultiHeadAttention(n_head, head_size) self.ff = PytorchFeedForward(n_embd) self.ln1 = nn.LayerNorm(n_embd) self.ln2 = nn.LayerNorm(n_embd) We prepare an attention sublayer, a feed-forward sublayer, and the two layer-norm modules that frame the residual paths. The following listing shows the transformer block forward pass. |
| Listing 5.14 Transformer block forward pass |

```python
class PytorchBlock(nn.Module):
def forward(self, x):
x = x + self.attn(self.ln1(x))
return x + self.ff(self.ln2(x))
```

Forward execution applies the pre-norm residual pattern twice: normalize, run the sublayer, add the residual, and repeat for the feed-forward layer. Each transformer block has two sublayers: multihead attention (tokens communicate) and feed-forward (tokens compute independently). Residual connections let gradients flow directly through the network, and layer normalization stabilizes training. We stack eight of these blocks to create depth. This implementation uses pre-norm ordering: layer norm runs before each sublayer, and the residual add happens afterward. Every operation here is massively parallel: embedding lookups process all tokens simultaneously, attention computes all query-key-value products across the batch in one launch, and feed-forward layers apply the same transformation to every position independently. PyTorch hides this parallelism behind clean abstractions, but underneath, cuBLAS and cuDNN are dispatching thousands of threads per operation. Chapter 6 optimizes the GEMV, softmax, LayerNorm, top-k, and GEMM kernels behind these layers, and chapter 8 restructures attention so the full attention matrix never goes to global memory. For now, the PyTorch baseline gives us correctness to validate against and performance numbers to beat.

-----

###### 104 Chapter 5 Transformer inference in CUDA

With our blocks defined, we can now assemble the complete transformer model that ties everything together, from token IDs to output logits. The following listing shows the transformer model initialization.

| Listing 5.15 Transformer model initialization |
|---|
| class PytorchTransformer(nn.Module): def __init__(self, vocab_size): super().__init__() self.token_embedding_table = nn.Embedding(vocab_size, n_embd) self.position_embedding_table = nn.Embedding(block_size, n_embd) self.blocks = nn.Sequential(*(PytorchBlock(n_embd, n_head) for _ in range(n_layer))) self.ln_f = nn.LayerNorm(n_embd) self.lm_head = nn.Linear(n_embd, vocab_size) This constructor mirrors the textbook GPT architecture: learned token and position embeddings, a stack of transformer blocks, a final layer norm, and a linear head for vocabulary logits. The modular structure separates concerns cleanly: embeddings han- dle input representation, blocks perform computation, and the language modeling head produces predictions. The initialization and forward pass complete our baseline implementation. The fol- lowing listing shows the transformer forward pass. |
| Listing 5.16 Transformer forward pass |

```python
class PytorchTransformer(nn.Module):
def forward(self, index, targets=None):
T = index.size(1)
tok_emb = self.token_embedding_table(index)
pos_emb = self.position_embedding_table(torch.arange(
T, device=device))
x = tok_emb + pos_emb
x = self.blocks(x)
x = self.ln_f(x)
logits = self.lm_head(x)
if targets is None:
loss = None
else:
logits = logits.view(-1, logits.size(-1))
targets = targets.reshape(-1)
loss = torch.nn.functional.cross_entropy(logits, targets)
return logits, loss
```

The forward method embeds the tokens, adds positional context, runs the block stack, normalizes, projects to logits, and optionally computes cross-entropy loss when training targets are provided.

-----

###### Inference: Theory and optimization 105

This PyTorch baseline from train.py processes every character token through a 128-dimensional embedding space across eight layers of self-attention and feed-forward networks. The modular structure makes it easy to understand each component before we start replacing pieces with custom CUDA kernels.

## 5.2 Inference: Theory and optimization

Inference fundamentally differs from training. We generate tokens one at a time through autoregressive sampling. Without key-value (KV) caching and other optimizations, this sequential generation would require recomputing massive amounts of redundant work.

### 5.2.1 Optimizing autoregressive generation

Instead of processing 64 tokens in parallel as we do during training, we generate tokens one at a time. We'll explore this process using real inference code from our 3\_trans-

```
former/ project, showing how to optimize it from a PyTorch prototype to production
```

CUDA kernels.

Using our realistic dimensions (batch = 1, seq = 64, embd = 768, heads = 8, layers = 24), let's trace through exactly what happens during autoregressive generation and how custom CUDA kernels can accelerate each phase.

Regarding scale considerations, remember that although our training configuration used smaller dimensions for educational clarity, the inference implementation scales up significantly (768-dimensional embeddings, 24 layers) to reflect production-like workloads. This larger scale makes the performance differences between PyTorch and CUDA implementations more pronounced: optimizations that save microseconds per layer compound to milliseconds across 24 layers. At a frontier model's depth (around 100 layers), these savings would be larger still.

Consider starting with a prompt: "The quick brown fox." To generate the next token, we run this sequence through the transformer and sample from the output distribution. Then, to generate the token after that, we run the extended sequence through the transformer again.

This sequential generation creates massive redundancy: we recompute attention for all previous tokens every time we generate a new token. Figure 5.6 illustrates this sequential generation pattern.

This sequential behavior creates both challenges and optimization opportunities. The challenge is that we can't parallelize across the sequence dimension like we do during prefill. The opportunity is that we can cache intermediate computations to avoid redundant work.

###### The KV cache: Avoiding redundant computation

The KV cache is a memory optimization that stores the key and value tensors from previous tokens during autoregressive generation, avoiding the need to recompute them for each new token.

-----

###### 106 Chapter 5 Transformer inference in CUDA

Without KV cache, generating a **the cat jumped over the**

100-token sequence requires computing attention for 1 + 2 + 3 + … + 100 = 5,050 token positions total. With KV cache, you compute only 100 posithe cat jumped over the

tions, once each: a 50× reduction in computation for long sequences. During each generation step, we compute the new token's key and **Next token QUERY** value vectors and append them to our **interacts with every**

**token KEY**

cache. When computing attention, the new token's query attends to all cached keys and values, transforming the operation from GEMM (recom-

Compare the query (vector) of the next token puting everything) to GEMV (one with the contained information of all previous query against cached data). Figure tokens, which is a list of vectors (a matrix). We cache values for K and V... hence "KV" cache

5.7 shows how the cache grows during

| generation. |  | Matrix-vector multiplication |
|---|---|---|
| Think about generating phrase "The cat sat on the". When | the | (GEMV) Figure 5.6 Decoding the generation pattern: |
| generating "the" (the fifth token), we don't need to recompute the keys |  | generating one token at a time, dominated by GEMV operations against cached keys and values |

and values for "The", "cat", "sat", and "on"; they're identical to what we computed in previous steps. We only need to compute the new query, key, and value for "the" and combine them with our cached values.

Rather than copy the entire helper module, focus on the handful of lines you actually need to kick off decoding. Start from the PyTorch baseline so the CUDA wrapper can reuse its weights, then hand the wrapped model to run\_generation.

The following listing shows how to launch CUDA-backed inference.

**Listing 5.17 Launching CUDA-backed inference**

```
base = PytorchTransformer(use_moe=False) Streams one token at a time,
model = CustomTransformer(base, use_moe=False).to(device) updating the KV cache between
```

| model.eval() | steps so attention kernels read |
|---|---|
| run_generation(model, "CUDA Dense") | from cached keys and values |

Set use\_moe=True in both constructors to exercise the mixture-of-experts path; the wrapper simply shares weights before switching to the CUDA kernels.

With KV caching, attention shifts from "full matrix" thinking to the thin column in figure 5.7. Each decode step looks up the new token embedding, adds the positional embedding for the current step, feeds the resulting [B, 1, embed\_dim] slice into the cached attention kernels, and appends the new key and value rows to the cache.

-----

###### Inference: Theory and optimization 107

Precalculate key and value pairs during transformer Predict one token at a time attention before generating all the tokens using matrix-vector mul

**The cat jumped over the** ?

###### WITHOUT KV cache WITH KV cache

M @ K M @ K

K N K 1

(M, K) @ (K, N) = (M, N) (M, K) @ (K, 1) = (M, 1)

inner dims cancel

matrix-matrix multiply matrix-vector multiply

**Figure 5.7 KV cache concept**

###### GEMV-dominated inference operations

The KV cache transforms our computational pattern from GEMM (matrix-matrix) to GEMV (matrix-vector) operations. Instead of computing attention for all positions simultaneously, we compute attention for one new position against all previous positions. The following listing highlights the inference embedding step from inference.py.

**Listing 5.18 Inference embedding step**

```python
class CustomTransformer(nn.Module):
```

|  | def forward(self, idx, kv_cache=None, use_cache=False): |  |
|---|---|---|
| import wrapper.inference.add | as add_op | Reuses PyTorch's |
| B, T = idx.shape |  | position lookups |

```python
tok_emb = self.token_embedding_table(idx)
pos_indices = torch.arange(T, device=device) % block_size
pos_emb = self.position_embedding_table(pos_indices)
if pos_emb.dim() == 2 and tok_emb.dim() == 3:
pos_emb = pos_emb.unsqueeze(0)
```

`x = add_op.add(tok_emb, pos_emb)` **with the forward-only CUDA kernel**

```
return x
```

**embedding tables for token and**

###### Executes the element-wise addition

-----

###### 108 Chapter 5 Transformer inference in CUDA

Because T = 1 during generation (we process a single new token at a time), the returned tensor already matches the narrow column in figure 5.7. From there, the attention heads run matrix-vector products against the cached keys and values instead of full matrix-matrix multiplies, so memory bandwidth becomes the primary bottleneck. Table 5.1 collects the shapes that flow through one decode step.

**Table 5.1 Tensor shapes at each stage of a single decode step**

| Stage | Tensor | Shape at decode step |
|---|---|---|
| Token lookup Position lookup CUDA add output Cached keys/values | tok_emb pos_emb x kv_cache['key'/'value'] | [B, 1, embed_dim] [1, embed_dim] [B, 1, embed_dim] [B, heads, steps_so_far, head_dim] |

###### Exercise: KV cache analysis for inference

Practice problem: Analyze the KV cache implementation in inference.py for our realistic setup (batch=1, max\_seq=64, heads=8, head\_dim=96):

**1** Calculate the total KV cache memory in bytes (assuming float32). **2** Compare memory usage with computation savings when generating 10 new

tokens from a 64-token prompt. **3** Implement a simple KV cache in Python and measure the speedup over

recomputation. **4** At what sequence length does the cache memory exceed 1 GB? The steps to follow are

**1** Implement the cache as a Python dictionary: cache = {'key': torch.zeros(…),

```
'value': torch.zeros(…)}.
```

**2** For each generation step, append new K,V pairs and slice appropriately. **3** Time both cached and uncached implementations. **4** Analyze memory scaling with sequence length. The expected outcome is to understand the practical memory/computation tradeoffs in autoregressive generation and why KV caching is essential for efficient inference.

###### Memory efficiency: Cache vs. recomputation tradeoffs

The KV cache trades memory for computation. Instead of recomputing keys and values for all previous tokens at each step, we store them and only compute new ones. Figure 5.8 compares the two: without the cache every step recomputes keys and values for all previous tokens, and with the cache only the new token is computed, while the stored keys and values are read back.

-----

###### Inference: Theory and optimization

###### Without KV cache

| step 1 | Od |
|---|---|
| step 2 | [I |
| step 3 | I] |
| step 4 | C0 |

###### Every step recomputes K and V for all previous tokens: compute per step grows with sequence length.

K/V computed this step.

**Figure 5.8 Memory usage comparison: KV cache trades memory for computation during decoding.**

For long sequences, this tradeoff becomes crucial. Consider generating a 1,000-token response: without KV cache, we'd recompute attention for all previous tokens at each step. With KV cache, we store the keys and values once and reuse them. The following listing demonstrates the memory versus computation tradeoff.

**Listing 5.19 KV cache memory calculation**

```python
cache_memory_bytes = 2 * batch_size
cache_capacity_bytes = cache_memory_bytes
kv_cache_summary = [
{
"step": step,
"recompute_ops": (step + 1) * 2,
"cached_ops": 2,
}
for step in range(3)
]
for entry in kv_cache_summary:
entry["savings"] = entry["recompute_ops"] - entry["cached_ops"]
```

The longer the sequence, the more computation we save, but the more memory we consume.

###### With KV cache

step 1 step 2 DEO step 3 step 4 OOOO

###### Cached K and V are reused; only the new token is computed: compute per step stays flat, cache memory grows linearly.

K/V read from cache (stored once).

###### Memory for 2 batches × 8 tokens × 1 head_dim × 2 (K+V) × 4 bytes

```
* block_size * head_size * 4
```

###### Tracking the raw allocation and the reusable capacity makes it obvious how much VRAM the cache reserves. Without cache, computes K,V for all

**tokens up to current position With cache, only computes K,V for a new token**

**Savings grow linearly with sequence position.**

###### Mixture-of-experts in inference: Sparse activation

Mixture-of-experts (MoE) is an architectural pattern that replaces a single dense feed-forward layer with multiple parallel feed-forward networks called "experts." Each expert has an identical architecture (the same layer sizes and activation functions) but

-----

###### 110 Chapter 5 Transformer inference in CUDA

different learned weights. A router network scores each expert for a given input and selects the top-k experts to run. The term "expert" is a metaphor from the machine learning literature: through training, each network tends to specialize in different input patterns, becoming an "expert" at processing certain types of tokens. This specialization emerges from optimization, not from explicit design: the router and experts co-adapt so that tokens get routed to whichever expert produces the best predictions. Figure 5.9 illustrates the MoE architecture.

output =

Only the top-2 selected experts weight\_1 \* Expert\_selected1(x) Result: (T, C) contribute to final output. +

weight\_2 \* Expert\_selected2(x)

| expert 1 | expert 2 | expert 3 | expert 4 | torch.concatenate |
|---|---|---|---|---|
| output | output | output | output | side-by-side |

Expert\_i:

1. (T, C) @ (C, 4C) -> (T, 4C)
  2. gelu(T, 4C) -> (T, 4C) MLPs are "experts."
3. (T, 4C) @ (4C, C) -> (T, C)

top\_k(T, 4, k=2) -> (T, 2) Select top-2 experts per token

active (solid) Probability distribution over experts not active (dashed) softmax(T, 4) -> (T, 4)

num\_experts = 4 top\_k = 2

Linear projection to num\_experts (T, C) @ (C, 4) -> (T, 4)

**Figure 5.9 MoE architecture**

MoE allows models to scale to massive parameter counts (trillions) while keeping inference costs manageable. Only a handful of experts (top-8 routing is common) out of dozens to hundreds are active per token, dramatically reducing compute requirements. For each token, a router network computes scores for all experts, selects the top-k highest-scoring experts, normalizes their weights, and combines their outputs. During inference, this sparse activation means you compute only a small fraction of the model's total parameters per token, roughly 3% to 10% in current production models (Deep- Seek-V3: 37B of671B; Qwen3-235B-A22B: 22B of235B).

The routing process involves several key steps: **1** Router computation: A linear layer transforms each token's embedding into

expert scores.

-----

###### Inference: Theory and optimization 111

**2** Softmax normalization: Convert scores to probabilities across all experts. **3** Top-k selection: Choose the k experts with the highest probabilities. **4** Renormalization: Rescale the selected experts' probabilities so they sum to 1. **5** Expert execution: Run each selected expert on the token. **6** Output combination: Weighted sum of expert outputs. This sparsity creates unique optimization opportunities for CUDA kernels, particularly in the top-k selection and routing computation. Figure 5.10 illustrates the top-k selection process that determines which experts process each token.

x **1.**

1 -2 5 0 batch dim 2 4 -4 2

top\_k(x, k=2) **2.** 1 -2 5 0 Keep the top 2 values of

each row; return those values and their indices.

| 2 | 4 top_k(x, k=2) | -4 top_k(x, k=2) | 2 3. |  |
|---|---|---|---|---|
| 1 | 5 | 0 | 2 |  |
| 4 | 2 | 1 | 3 |  |
| values |  | indices |  |  |
| 2 outputs of shape: |  |  |  | Figure 5.10 Top-k |
| (batch_size=2, k=2) |  |  |  | expert selection |

The inference.py code demonstrates how to implement MoE with custom CUDA kernels for the routing operations. The implementation uses eight experts with top-2 routing, creating realistic sparsity patterns. The CUDA version uses optimized top-k selection and softmax kernels that are faster than PyTorch equivalents for this sparse computation pattern. That same implementation demonstrates a real challenge in custom CUDA kernels: numerical stability. Our custom softmax and top-k kernels each introduce small floating-point errors (on the order of1e-6). In isolation, these errors are harmless. But MoE routing chains them together: softmax outputs feed into top-k selection, and tiny score differences can change which experts get selected. When expert 3 scores 0.3001 and

-----

###### 112 Chapter 5 Transformer inference in CUDA

expert 5 scores 0.3000, a 1e-4 error can determine routing. Over many tokens, these routing divergences compound. This is why production systems use battle-tested implementations (cuBLAS, cuDNN, or carefully validated custom kernels) for numerically sensitive operations. Validate custom kernels rigorously: compare GPU outputs against CPU references, and measure average error, relative error, and maximum difference across many inputs. If the statistics fall within acceptable thresholds, the kernel is production ready.

The PyTorch version keeps everything in Python so we can see the mechanics clearly. Each token scores every expert with a dense layer, picks the top-k experts and normalizes those probabilities, then runs only the winners and blends their outputs. The following listing shows that reference implementation.

**Listing 5.20 PyTorch MoE feed-forward layer**

```python
class MoEFeedForward(nn.Module):
"""PyTorch MoE layer for inference"""
def __init__(self, n_embd):
super().__init__()
self.gate = nn.Linear(n_embd, n_experts,
self.experts = nn.ModuleList([
nn.Sequential(
nn.Linear(n_embd, 4 * n_embd),
nn.GELU(),
nn.Linear(4 * n_embd, n_embd)
) for _ in range(n_experts)
])
```

```python
def forward(self, x):
B, T, C = x.shape
x_flat = x.view(B * T, C)
gate_logits = self.gate(x_flat)
gate_probs = torch.nn.functional.softmax(gate_logits, dim=-1)
topk_probs, topk_indices = torch.topk(
gate_probs, top_k, dim=-1)
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
combined = torch.zeros_like(x_flat)
for i in range(x_flat.shape[0]):
token_output = torch.zeros(C,
for j in range(top_k):
expert_idx = topk_indices[i,
expert_prob = topk_probs[i,
expert_output = self.experts[expert_idx](x_flat[i:i+1])
token_output += expert_prob * expert_output.squeeze(0)
combined[i] = token_output
return combined.view(B, T, C)
```

```
bias=True)
```

###### Gates the linear layer for expert scoring

###### Eight experts. Each is MLP with GELU activation (matching training).

###### Flattens the batch and sequence for token-level routing Computes gate scores for all eight experts

###### Selects the top two experts per token based on scores

###### Outer Python loop that still iterates per token (no CUDA acceleration yet)

```
device=x_flat.device)
j]
j]
```

###### Runs the chosen expert MLP and accumulates its weighted output

-----

###### Inference: Theory and optimization 113

This version keeps the routing math in Python, which is fine for clarity but quickly becomes a bottleneck when we scale past a handful of experts.

###### CUDA-optimized MoE routing

To route with CUDA helpers, we keep the same conceptual flow but hand the heavy lifting to CUDA kernels: score experts in PyTorch for clarity, offload the softmax and top-k work to memory-optimized CUDA code, and then combine the selected expert outputs back in Python with only a handful of tensors. The following listing shows the implementation in inference.py.

**Listing 5.21 CUDA-optimized MoE forward pass**

```python
def forward(self, x):
import wrapper.inference.softmax as softmax_op
import wrapper.inference.topk as topk_op
import wrapper.inference.mul as mul_op
```

###### Reshapes the logits for the CUDA

`B, T, C = x.shape` **softmax kernel and runs it on the GPU**

```
x_reshaped = x.view(B * T, C)
gate_logits = self.gate(x_reshaped) Uses the CUDA top-k kernel to pick
```

**experts without leaving the device**

```
gate_probs = softmax_op.softmax(gate_logits.unsqueeze(1)).squeeze(1)
topk_probs, topk_indices = topk_op.topk(gate_probs, top_k)
topk_probs = mul_op.mul(
topk_probs,
topk_probs.sum(dim=-1, keepdim=True).reciprocal().expand(-1, top_k),
)
```

###### Normalizes the selected scores with

```python
combined = torch.zeros_like(x_reshaped) a CUDA element-wise multiply
for token, (scores, experts) in enumerate(zip(topk_probs, topk_indices)):
for prob, expert_idx in zip(scores, experts):
expert_out = self.experts[expert_idx]( The Python loop still
x_reshaped[token:token + 1]).squeeze(0) dispatches experts, but
combined[token] += prob * expert_out now only for the k winners.
```

`out = combined.view(B, T, C)` **Reshapes back to [B, T, C] after** `return out` **fusing all expert outputs** The softmax, top-k selection, and score normalization all run on the GPU, so only the sparse expert dispatch loop runs in Python. This keeps the routing work where memory bandwidth is plentiful and trims the number of CPU-GPU tensor dispatches per token down to the handful of experts we actually execute. The CUDA routing implementation provides a 1.2 to 1.5× speedup over PyTorch for the MoE model, with the benefits increasing as the number of experts grows. The custom top-k and softmax kernels are particularly effective for this memory-bound routing computation.

###### GEMV vs . GEMM: The fundamental shift

The transition from training to inference represents a fundamental shift in computational patterns. Figure 5.11 illustrates this fundamental difference.

-----

###### 114 Chapter 5 Transformer inference in CUDA GEMM (Prefill)

**[T, C] [C, C] [T, C]**

@ →

###### T tokens processed in parallel (compute-bound)

---

###### GEMV (Decode)

**[1, C] [C, C] [1, C]**

@ →

###### 1 token at a time (memory-bound)

**Figure 5.11 Training GEMM vs. inference GEMV**

The training pattern (GEMM) is as follows:

¡ Process 8 tokens simultaneously: [4,8,4] @ [4,4] = [4,8,4]. ¡ Attention computes full 8 × 8 matrices. ¡ High arithmetic intensity. ¡ All positions are computed in parallel. The inference pattern (GEMV) is as follows:

¡ Process 1 new token: [1,1,4] @ [4,4] = [1,1,4]. ¡ Attention: 1 query against cached keys. ¡ Lower arithmetic intensity per step. ¡ Sequential token generation. ¡ Memory bandwidth is critical for cache access. In our nano example, training attention scores are [4,8,8] matrices (256 elements total). Inference attention scores are [1,1,4] vectors that grow by one element at each step. Training saturates GPU compute with large parallel operations. Inference becomes memory-bound as we repeatedly access the growing KV cache.

### 5.2.2 Deploying the optimized stack

Inference optimizations matter only if the end-to-end experience improves. This section walks through a representative generation session and the measurement harness we use to prove the gains.

-----

###### Inference: Theory and optimization 115

###### Complete generation example: Token by token

Let's walk through generating three new tokens using our nano-GPT setup, showing exactly how the KV cache avoids redundant computation. The following listing shows the complete generation loop with KV caching.

| Listing 5.22 Token generation loop with KV caching |  |
|---|---|
| def run_generation(model, model_name): prompt = "Once upon a time" prompt_tokens = torch.tensor([encode(prompt)], Warms up the dtype=torch.long, device=device) with prefill operations generated = prompt_tokens.squeeze().tolist() initialize the GPU with torch.no_grad(): Decodes step by _, kv_cache = model.prefill(prompt_tokens) using KV cache current = prompt_tokens[:, -1:] efficient for _ in range(max_new_tokens): generation | model to state step for autoregressive |
| logits, kv_cache = model.decode_step(current, kv_cache) |  |

```
current = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
generated.append(current.item())
return generated
```

This benchmarking framework measures two critical inference metrics: time to first token (TTFT) for responsiveness and tokens per second for throughput.

###### Production inference results: PyTorch vs. CUDA

Running our complete benchmarking suite shows the practical effect of CUDA optimization for transformer inference. Figure 5.12 shows the performance comparison.

Time to first token

PyTorch: 12 ms

Naive CUDA: 16 ms (0.75x)

Tokens per second (decode)

PyTorch: 95 tok/s

Naive CUDA: 85 tok/s (0.90x)

**Figure 5.12 Performance comparison: our naive CUDA kernels underperform PyTorch's optimized implementations, demonstrating why production systems use battle-tested libraries.**

The following listing demonstrates the thorough comparison and validation code that verifies correctness while measuring performance.

-----

###### 116 Chapter 5 Transformer inference in CUDA

**Listing 5.23 Performance comparison and validation code**

```python
from collections import OrderedDict
import time
```

```python
def timed(run):
start = time.perf_counter()
tokens = run()
elapsed = time.perf_counter() - start
return torch.tensor(tokens, device=device), elapsed
```

```
results = OrderedDict(
(
("pytorch_dense", timed(run_pytorch_dense)),
("pytorch_moe", timed(run_pytorch_moe)),
("cuda_dense", timed(run_cuda_dense)),
("cuda_moe", timed(run_cuda_moe)),
)
)
```

```python
def flatten_tokens(tokens):
return [t for sub in tokens for t in (sub if isinstance(sub, list) else
[sub])]
```

```python
dense_a = flatten_tokens(results["pytorch_dense"][0].tolist())
dense_b = flatten_tokens(results["cuda_dense"][0].tolist())
assert dense_a == dense_b
moe_a = flatten_tokens(results["pytorch_moe"][0].tolist())
moe_b = flatten_tokens(results["cuda_moe"][0].tolist())
differences = sum(a != b for a, b in zip(moe_a, moe_b))
diff_rate = differences / len(moe_a)
tolerance = 0.50 if len(moe_a) > 20 else 0.10
if diff_rate < tolerance:
print("MoE outputs match within tolerance")
else:
print("MoE outputs differ as expected for sparse routing")
```

Run all four implementations (PyTorch dense/MoE, CUDA dense/MoE). Require exact token-by-token agreement for the dense models. Measure MoE divergence by token difference rate because routing can drift under tiny floating-point perturbations. **NOTE** The 0.50 threshold is deliberately loose, and it is not a claim that the two implementations agree to within 50%. In an autoregressive MoE model, a single early routing flip (top-2 selection is a hard argmax-style choice, so a floating-point difference of one ULP in a gate logit can swap which expert wins) changes that token, which changes the context and cascades into a different downstream sequence. Token-level agreement is therefore a coarse end-to-end

-----

###### Inference kernels 117

sanity check rather than a correctness proof; the tight per-operation checks (torch.testing.assert\_close on the softmax, top-k, and multiply kernels) are what actually verify the CUDA code. For short sequences, the code tightens the bound to 0.10 because there are fewer opportunities for the cascade to compound. The inference.py comparison demonstrates exact token sequence matching for dense models and tolerance-based verification for MoE models. CUDA implementations range from 0.75× to 1.5× relative to PyTorch. Some operations, such as the prefill behind time-to-first-token, actually lose ground to CPU-GPU transfer overhead in our educational implementation, with MoE models benefiting more from the sparse routing optimizations at scale.

The actual model configurations in inference.py are as follows: ¡ *Dense model-95-token printable-ASCII vocabulary, 768 embedding dimensions,*

8 attention heads, 24 layers, ~177M parameters ¡ *MoE model-Same inference-oriented configuration, but with 8 experts using* top-2 routing, ~963M total parameters (only ~283M active per token, since routing selects 2 of the 8 experts in each layer)

###### Next steps: Expanding CUDA kernel coverage

Our inference optimization journey shows a clear progression: ¡ *Phase 1: Prove concept-Start with simple embedding kernels to establish the ker-*

nel binding pipeline and verify correctness. ¡ *Phase 2: Target bottlenecks-Profile to identify which operations take the most time,*

then implement CUDA kernels for those specific operations. ¡ *Phase 3: Optimize memory access-For inference-heavy workloads, focus on optimiz-*

ing memory bandwidth, fusing kernels, and reducing data movement. The foundation we've built with embedding kernels extends naturally to attention, MLP, and layer normalization operations, each providing incremental performance improvements.

## 5.3 Inference kernels

Inference has its own performance profile. These kernels abandon training assumptions and specialize for single-token workloads where latency trumps throughput.

### 5.3.1 Memory layout and contiguity

Our CUDA kernels, for training and inference alike, expect contiguous memory.

**Listing 5.24 Contiguity check in forward pass**

```python
def forward(self, x):
if not x.is_contiguous():
```

-----

###### 118 Chapter 5 Transformer inference in CUDA

`x = x.contiguous()` **Creates a contiguous**

**copy if needed**

```
return CustomOperation.apply(x)
```

| Why contiguity matters |
|---|
| Contiguous memory: |
| x = [[1, 2, 3], Memory layout: [1, 2, 3, 4, 5, 6, 7, 8, 9] |
| [4, 5, 6], |
| [7, 8, 9]] Element x[0][1] is at: base + (0*3 + 1) = base + 1 |
| Noncontiguous after transpose: |
| x_t = x.T Original memory: [1, 2, 3, 4, 5, 6, 7, 8, 9] |
| = [[1, 4, 7], But now "row 0" is: [1, 4, 7] (stride 3 apart!) |
| [2, 5, 8], Element x_t[0][1] is at: base + (0 + 1*3) = base + 3 |
| [3, 6, 9]] |
| Our CUDA kernels assume contiguous layout: |
| int idx = row * num_cols + col; float val = data[idx]; |
| If data is noncontiguous, this indexing breaks; we'd read the wrong memory locations. |
|  |
| The solution is PyTorch's .contiguous(), which creates a new tensor with contiguous memory layout. The cost is one memcpy, but it ensures correctness. |

### 5.3.2 GEMV: The inference workhorse

Inference turns matrix-matrix multiplication into matrix-vector multiplication. Training multiplies a batch of tokens, [B, T, D] @ [D, D] → [B, T, D], while decode multiplies a single token, [1, 1, D] @ [D, D] → [1, 1, D]. That shift from batch processing to singletoken processing requires completely different kernel optimization strategies. The following comparison shows the difference:

```
GEMM (training, T=64 tokens):
Q[64, 128] @ K.T[128, 64] = Scores[64, 64]
-> 64 * 64 * 128 = 524,288 multiply-adds
-> High arithmetic intensity, compute-bound
GEMV (inference, 1 new token):
q[1, 128] @ K.T[128, 64] = scores[1, 64]
-> 1 * 64 * 128 = 8,192 multiply-adds
-> Low arithmetic intensity, memory-bound
```

-----

###### Inference kernels 119

The GEMV kernel implements efficient matrix-vector multiplication for inference.

| Listing 5.25 GEMV kernel signature and coordinates |  |
|---|---|
| __global__ void gemv_kernel( const float* A, const float* x, float* y, int batch, int M, int N) { int batch_idx = blockIdx.z * blockDim.z + threadIdx.z; | One thread per output row (not per output element as in training) |
| int row = blockIdx.y * blockDim.y + threadIdx.y; |  |
| The grid uses blockIdx.z to pick the batch example and a 2D tile thread handles one output row. | to index rows; each |
| Listing 5.26 GEMV guard and pointer setup |  |
| if (batch_idx < batch && row < M) { float sum = 0.0f; const float* A_batch = A + batch_idx * M * N; const float* x_batch = x + batch_idx * N; Once inside bounds, we compute base pointers for the current batch | slice. |
| Listing 5.27 GEMV dot-product loop |  |
| for (int col = 0; col < N; col++) { |  |
| sum += A_batch[row * N + col] * x_batch[col]; } Sequential loop over columns: bandwidth is the The thread walks along the row, accumulating a dot product against | memory bottleneck the input vector. |
| Listing 5.28 GEMV store result |  |
| y[batch_idx * M + row] = sum; } } We write the final scalar into the output vector for this batch item | and row. |
| Design decision: GEMV vs. GEMM kernel strategy |  |
| So why not use the GEMM kernel for everything? |  |
| GEMM kernels are optimized for high arithmetic intensity: |  |

-----

###### 120 Chapter 5 Transformer inference in CUDA

| (continued) |
|---|
| ¡ Using shared memory tiling |
| ¡ Having multiple threads cooperate on one output |
| ¡ Applying complex blocking strategies |
| GEMV is memory-bound: we're bottlenecked by loading A and x, not by computing the |
| dot product. |
|  |
| The GEMM approach for [1, 128] @ [128, 128] is as follows: |
| // Complex: shared memory tiles, thread cooperation |
| __shared__ float tile_A[16][16]; |
| __shared__ float tile_B[16][16]; |
| // Load tiles, sync, accumulate, sync, repeat... |
| // Overhead dominates for 1-row input |
| There is a GEMV approach (simpler is better): |
| // Simple: one thread per output element |
| float sum = 0.0f; |
| for (int col = 0; col < N; col++) |
| sum += A[row * N + col] * x[col]; |
| y[row] = sum; |
| Lastly, consider a performance comparison for attention query projection: |
| ¡ Input-q = [1, 128] |
| ¡ Weight-W_q = [128, 128] |
| ¡ GEMM kernel-~50 μs (overkill for 1 row) |
| ¡ GEMV kernel-~15 μs (optimal for memory bandwidth) |
| The simpler GEMV kernel wins because it matches the computation pattern to the |
| hardware bottleneck. |

###### Top-k for MoE routing

Expert selection via top-k value extraction: MoE routing selects top-k experts from router logits. For k = 2, we need the two highest-scoring experts. This operation differs from the dense linear algebra we've seen so far. One example is selecting experts for a token:

```
Router logits for 8 experts: [0.5, -0.2, 1.3, 0.8, -0.5, 0.9, 0.1, -1.0]
Step 1: Identify top-2 values
Max values: [1.3, 0.9]
Max indices: [2, 5]
Step 2: Normalize routing weights
Raw weights: [1.3, 0.9]
Normalized (softmax over the two): [0.60, 0.40]
Result: Route token to Expert 2 (weight 0.60) and Expert 5 (weight 0.40)
```

-----

###### Inference kernels 121

The top-k kernel identifies the k largest values and their indices for MoE routing. Unlike our previous kernels that parallelize across embedding dimensions or sequence positions, top-k parallelizes across batch elements. Each token in the batch needs its own independent routing decision, so we assign one CUDA block per token. Within each block, a single thread scans through all expert scores and maintains a small sorted buffer of the k best candidates.

This design reflects the problem structure: with only 8 experts and k = 2, there's no benefit to parallel reduction or sorting networks. A single thread can examine all 8 scores and insert the top 2 into a sorted list faster than the overhead of thread synchronization would allow. The following listing shows the kernel signature.

**Listing 5.29 Top-k kernel signature**

```
__global__ void topk_kernel(
const float* input,
float* values,
int* indices,
int batch_size,
int n,
```

| int k) { | One thread per batch element (no |
|---|---|
| int batch_idx = blockIdx.x; | parallelism within the batch) |

The input tensor has shape [batch\_size, n], where n is the number of experts. The kernel writes two output tensors: values contains the k highest scores (for weighting the expert outputs), and indices contains the experts that produced those scores (so we know which experts to run). The following listing shows how each thread computes its row offsets.

**Listing 5.30 Top-k pointer arithmetic**

```
if (batch_idx < batch_size) {
const float* input_row = input + batch_idx * n;
float* values_row = values + batch_idx * k;
int* indices_row = indices + batch_idx * k;
```

Each thread computes pointers to its own slice of the input and output arrays. The input row contains n expert scores, and the output rows will hold the k winners. Because each batch element is independent, threads never conflict on memory accesses. Before scanning the expert scores, we initialize the output buffers. The following listing shows this initialization.

**Listing 5.31 Top-k buffer initialization**

```
for (int i = 0; i < k; i++) {
values_row[i] = -INFINITY;
indices_row[i] = -1;
}
```

-----

###### 122 Chapter 5 Transformer inference in CUDA

Setting values to negative infinity ensures that any real expert score will be larger and can claim a slot. The indices start at -1 as a sentinel value, though in practice, every slot will be filled because we have more experts than k.

Now we can scan through all n expert scores and maintain a sorted list of the k best. The following listing shows the insertion sort logic.

**Listing 5.32 Top-k scanning with a maintained sorted window**

```
for (int i = 0; i < n; i++) {
float val = input_row[i]; O(n × k) algorithm:
```

**examines all n values**

```
for (int j = 0; j < k; j++) {
if (val > values_row[j]) {
for (int m = k - 1; m > j; m--) {
values_row[m] = values_row[m - 1];
indices_row[m] = indices_row[m - 1];
}
```

`values_row[j] = val;` **Inserts in sorted order: shifts** `indices_row[j] = i;` **existing entries right to make room**

```
break;
}
}
}
}
}
```

The algorithm walks through all n expert scores. For each score, it checks whether the score should enter the top-k buffer by comparing it against the current entries from best to worst. If the score beats an entry, we shift everything below that entry down by one position (evicting the smallest value) and insert the new score. The break statement ensures that we don't insert the same score multiple times.

This is classic insertion sort, and its O(n × k) complexity sounds inefficient. But for MoE routing with n = 8 experts and k = 2, that means 16 comparisons total: trivial for a single thread. The simplicity of the code path (no shared memory, no synchronization, no complex indexing) makes this the fastest choice for small problem sizes.

###### Design decision: Why naive insertion sort?

Some alternatives follow. Option 1: Parallel reduction (like finding the max):

```
// Multiple rounds: find max, mask it, repeat k times
__shared__ float smax[256];
for (round = 0; round < k; round++) {
parallel_max_reduce(input, smax);
```

-----

###### Summary 123

|  |
|---|
| top_k[round] = smax[0]; |
| input[argmax] = -INF; // Mask found value |
| } |
| The overhead is thread synchronization and shared memory management. This is good for large k (k > 32). |
|  |
| Option 2: Sorting networks: |
| // Bitonic sort: fixed comparison pattern |
| // Compare-exchange pairs in parallel |
| if ((i & mask) == 0) |
| compare_and_swap(arr[i], arr[i ^ mask]); |
| The overhead is complex indexing; works well for power-of-2 sizes. This is good for k close to n. |
|  |
| Option 3: Insertion sort (our choice): |
| // Simple: one thread scans all values |
| for each value in input: |
| if value > smallest_in_top_k: |
| insert_sorted(value, top_k) |
| It works best for small k (k = 2 in our case). |
|  |
| Reasons this works for MoE are |
| ¡ k is tiny (k = 2 here; DeepSeek-V3 sends each token to 8 of 256 experts). |
| ¡ n is small (n = 8 experts in our example). |
| ¡ Overhead of parallel algorithms > benefit. |
| ¡ Total comparisons: 8 × 2 = 16 (trivial). |
| Performance: For k=2, n=8: |
| ¡ Our kernel: ~2 μs |
| ¡ Parallel reduction: ~15 μs (synchronization overhead) |
|  |

###### Summary

¡ A PyTorch transformer serves as the reference implementation and ground

truth for all CUDA optimizations. Custom kernel outputs are compared against it for correctness, and performance measurements use it as the known baseline. ¡ Each attention head computes query, key, and value projections, then uses dot

products between queries and keys to determine how much each position attends to every other position. The attention weights multiply the values, producing a weighted combination of information from across the sequence.

-----

###### 124 Chapter 5 Transformer inference in CUDA

¡ Multiple attention heads run in parallel, each with its own learned projections,

capturing different types of relationships simultaneously. One head might track nearby words for syntax, while another tracks distant references. The outputs are concatenated and projected back to the embedding dimension. ¡ Transformer blocks pair attention with feed-forward networks. After attention

gathers information across positions, the feed-forward network processes each position independently, expanding to 4× the embedding dimension, applying GELU, and then projecting back down. ¡ Prefill processes the entire prompt in parallel, making it a GEMM-dominated

workload. Decode generates tokens one at a time, where each new token depends on all previous tokens. This sequential dependency makes decode GEMV-dominated and memory-bound. ¡ The KV cache stores key and value projections once and reuses them during

autoregressive generation, transforming O(N2) recomputation into O(N) as sequences grow. ¡ Training and prefill saturate GPU compute with matrix-matrix multiplication.

Decode uses matrix-vector multiplication, where a single query vector multiplies against cached matrices, making memory bandwidth rather than compute the bottleneck. ¡ The GEMV kernel assigns each thread one output element, loads one row of

the weight matrix, and computes a dot product with the input vector. Shared memory tiling doesn't help here because arithmetic intensity is too low: the bottleneck is memory-read speed, not compute. ¡ MoE replaces a single feed-forward layer with multiple parallel networks. A

router scores each expert and selects the top-k experts to execute, so only k/n of the parameters are active per token. This reduces compute while maintaining total model capacity. ¡ Top-k selection for MoE routing uses insertion sort. With small k (typically 2) and

small n (8-64 experts), the O(n × k) insertion sort outperforms parallel algorithms because a single thread scanning 8 values beats the synchronization overhead of parallel reduction. ¡ PyTorch extensions connect CUDA kernels to Python: .cu files contain the kernels, C++ bindings expose functions through pybind11, and setup.py compiles everything into an importable Python module. ¡ CUDA kernels assume row-major packed memory, but PyTorch operations like

transpose create views with non-contiguous strides. Calling .contiguous() before passing tensors to CUDA ensures pointer arithmetic works correctly. ¡ Comparing CUDA outputs against the PyTorch reference catches numerical precision problems and algorithmic bugs early, before they compound in the full model. Match the check to the operation: use a tight numerical comparison (for

-----

###### Summary 125

example, torch.testing.assert\_close) on individual kernel outputs, require exact token-by-token agreement for the dense end-to-end model, and fall back to a difference-rate tolerance for MoE, where hard top-k routing makes token-level divergence expected. ¡ All kernels in this chapter are deliberately naive: simple GEMV loops, explicit

softmax steps, and insertion sort for top-k. Chapter 6 optimizes them with tiling, shared memory, and vectorization.

-----

# 6

## Optimizing our transformer kernels

---

| This chapter covers |
|---|
| ¡ Profiling transformer kernels with Nsight Compute to pinpoint real bottlenecks |
| ¡ Upgrading GEMV, softmax, LayerNorm, top-k, and |
| GEMM through a repeatable optimization playbook |
| ¡ Balancing memory bandwidth, register pressure, |
| and occupancy as first-class design constraints |
| ¡ Deciding when shared memory, vectorization, |
| and warp-level intrinsics actually pay off |
| ¡ Validating every improvement with progressive |
| benchmarks and checks |
| ¡ Packaging the resulting patterns so they transfer |
| cleanly to production transformer stacks |

The gap between "correct" and "fast" can span two orders of magnitude, and closing that gap requires understanding how GPUs actually move and process data. This chapter takes five kernels central to transformer inference: GEMV, softmax, LayerNorm, top-k, and GEMM, and optimizes each one from first principles. We

-----

###### GEMV 127

will profile each kernel, identify its bottleneck, apply a targeted fix, and measure the result.

## 6.1 Preparing to optimize

Before we can effectively optimize our kernels, we need to establish a robust baseline and a reliable way to measure improvements. This section sets up the profiling tools and methods that will guide our optimization decisions throughout the chapter.

### 6.1.1 Profiling CUDA kernels

Before we start optimizing, we need to understand how to measure kernel performance. NVIDIA provides two primary profiling tools: ¡ *Nsight Compute (ncu) is our microscope for individual kernels. It reports occu-* pancy, memory throughput, compute utilization, warp stall reasons, instruction mix, and register pressure, making it the right tool whenever you want to squeeze more performance from one kernel. ¡ *Nsight Systems (nsys)* zooms out to the application timeline. It shows launch sequences, CPU-GPU synchronization, CUDA stream usage, and memory transfers so you can spot coordination problems that hide outside a single kernel. In this chapter, we focus on kernel-level optimizations using Nsight Compute. Each optimization stage includes profiling metrics showing exactly why and how much each change improves performance. Application-level profiling with Nsight Systems is covered in appendix E. The following listing shows the Nsight Compute invocation we use to capture a full kernel profile.

**Listing 6.1 Nsight Compute profile command**

```
ncu --set full -o output_profile ./your_program
```

This generates a detailed report showing memory bandwidth utilization, compute throughput, occupancy, and bottleneck analysis. Throughout this chapter, we'll use these metrics to validate our optimizations and understand GPU behavior at a low level.

## 6.2 GEMV

GEMV calculates the product of a matrix and a vector. For transformer inference, particularly in an autoregressive setting where we generate one token at a time, the batch size is often just one. In this scenario, the highly parallel nature of GEMM is underutilized, and the computation effectively becomes a series of GEMV operations. Optimizing for this specific case is critical for speeding up how fast tokens stream along the user's screen.

-----

###### 128 Chapter 6 Optimizing our transformer kernels

**(M, N) @ (N) = (M)**

Matrix A (M × N) × x = y

**Figure 6.1 Matrix** `A multiplies a`

**length-N input vector to produce an**

```
M-length output vector, illustrating the
```

Dot product: y[i] = A[i,:] • x **shape transformation (M,N) @ (N) = (M).**

Our goal is to compute y = A @ x, where A is an M x N matrix, x is a vector of size N, and y is the resulting vector of size M. Figure 6.1 illustrates this operation conceptually, using a square for the matrix and two rectangles for the input and output vectors so you can

see (M, N) @ (N) = (M) at a glance.

| Memory-bound operations and GEMV performance |
|---|
| Before we begin optimizing, it's crucial to understand what makes GEMV different |
| from GEMM. GEMV is a memory-bound operation, meaning performance is limited by how fast we can move data from global memory, not by how fast we can compute. |
| Consider the computational intensity: the ratio of floating-point operations to bytes transferred. We read M * N elements from matrix A, another N from vector x, and write M results back out. Each output element performs N multiplies plus N-1 adds, for 2 * M * N total FLOPs. Divide those FLOPs by the 4-byte loads and stores, and you land near 0.5 FLOPs per byte of traffic. |
| Compare this to GEMM, which achieves hundreds of FLOPs per byte through data reuse. For GEMV, we read each matrix element exactly once, with minimal opportunity for reuse. |
| Our optimization goal is not to maximize compute utilization (which may stay below 5% of peak) but to maximize memory bandwidth utilization. A well-optimized GEMV kernel should achieve 70% to 90% of the GPU's peak memory bandwidth. |

We will progress through four implementations, comparing them against the cuBLAS

baseline provided in book.cu/4\_optim/gemv/kernels/0\_cublas.cu. The Python

script book.cu/4\_optim/gemv/main.py benchmarks all kernels and verifies correctness. The GEMV, softmax, and LayerNorm kernels in this chapter are based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license, https://github.com/

[Maharshi-Pandya/cudacodes).](https://github.com/Maharshi-Pandya/cudacodes)

| GEMV terminology |
|---|
| GEMV stands for general matrix-vector multiplication: the "G" reminds us that the |
| matrix is arbitrary, the "M" and "V" mark the matrix and vector operands, and the |

-----

###### GEMV 129

linking "E" simply keeps the BLAS naming convention. The full operation is y = alpha \*

```
(A @ x) + beta * y with optional scaling coefficients; here, we pin alpha = 1 and
```

beta = 0 so we can focus on the CUDA mechanics.

The prefix "S" in SGEMV indicates single precision (32-bit floating-point). Other prefixes include D (double precision), C (complex single precision), and Z (complex double precision).

Our first approach will be a naive implementation. The most intuitive way to parallelize this operation on a GPU is to assign one thread to calculate one element of the output vector y. Because each output element y[i] is the dot product of the ith row of matrix A and the vector x, each thread will be responsible for one such dot product. You can find the code for this implementation in book.cu/4\_optim/gemv/kernels/ 1\_naive.cu. Let's break down how it works.

First, each thread calculates the unique row index it will be responsible for:

```
int row = blockDim.x * blockIdx.x + threadIdx.x; The standard CUDA calculation
```

**for a global 1D index**

Next, after a boundary check to make sure the row is within the matrix dimensions, the thread iterates over each column, accumulating the dot product of the matrix row and the input vector: `if (row < M) {` **A boundary check prevents threads from**

`float sum = 0.0f;` **accessing memory outside the matrix.**

```
for (int col = 0; col < N; col++) { The thread iterates through
sum += matd[row * N + col] * vecd[col]; the columns, performing the
```

`}` **dot product.**

```
resd[row] = sum;
```

`}` **Writes the final result for the row to**

**the output vector in global memory**

This implementation is simple and correct, but it has a major performance flaw related to how it accesses global memory. Each thread iterates through a row of the matrix. Let's visualize the memory access pattern.

### 6.2.1 Understanding the memory access problem

The key problem here is memory coalescing. GPUs achieve high memory bandwidth by reading or writing a wide chunk of memory at once. When threads in a warp access contiguous memory locations, the hardware can "coalesce" these accesses into a single transaction.

-----

###### 130 Chapter 6 Optimizing our transformer kernels

| What is a warp? |
|---|
| A warp is the fundamental execution unit on NVIDIA GPUs and consists of exactly 32 |
| threads. These 32 threads execute together in a SIMT (Single Instruction, Multiple Threads) fashion: they all execute the same instruction at the same time, but on different data. AMD GPUs have an equivalent concept called a wavefront, which is typically 64 threads wide. |
| The key properties of warps are |
| ¡ Size-Always 32 threads on current NVIDIA GPUs. |
| ¡ Lane ID-Each thread within a warp has a lane ID from 0 to 31, accessible via |
| threadIdx.x % 32. Lane IDs determine which threads communicate during shuffle operations. |
| ¡ Execution-Threads in a warp execute in lockstep. |
| ¡ Communication-Warp-level primitives like __shfl_down_sync allow threads to |
| exchange data through registers without using shared memory. |
| ¡ Performance-Warp shuffles exchange values directly through registers, avoid- |
| ing both the shared-memory round trip and the __syncthreads() barrier that a shared-memory reduction requires, making them faster for reductions within a warp. |
| Throughout this chapter, we'll use warps for efficient parallel reductions because they |
| provide the fastest way for threads to collaborate within a group of 32. |

Let's examine exactly what happens in our naive kernel. Consider the first warp (threads 0-31) in the first block, where blockIdx.x = 0. When these threads execute the first iteration of the loop, where col = 0, they access

```
Thread 0: row = 0, accesses matd[0 * N + 0] = matd[0]
Thread 1: row = 1, accesses matd[1 * N + 0] = matd[N]
Thread 2: row = 2, accesses matd[2 * N + 0] = matd[2 * N]
Thread 3: row = 3, accesses matd[3 * N + 0] = matd[3 * N]
...
Thread 31: row = 31, accesses matd[31 * N + 0] = matd[31 * N]
```

Adjacent threads access elements that are N addresses apart in memory. For a typical matrix where N = 8192, thread 0 and thread 1 access elements separated by 8,192 floats (32,768 bytes)! These rows are far apart in memory, leading to completely uncoalesced access. The GPU's memory controller fetches data in aligned chunks called cache lines (typically 32-128 bytes). When a warp performs a coalesced access to consecutive addresses, all 32 threads can be satisfied by just a few cache line reads. When access is uncoalesced with large strides, each thread requires a separate memory transaction. The hardware may have to perform up to 32 separate memory transactions for the 32 threads in a warp, leaving most of each fetched cache line unused.

-----

###### GEMV 131

This pattern repeats for every column access, throughout the entire kernel. The result is that we achieve only 10% to 20% of the GPU's peak memory bandwidth. We will fix this in the next section.

| Exercise: Memory coalescing analysis |
|---|
| Modify the naive GEMV kernel in book.cu/4_optim/gemv/kernels/1_naive.cu to |
| track memory addresses. Print the addresses accessed by threads 0, 1, and 2 in the first warp during the first iteration of the loop. Calculate the stride between con- secutive thread accesses. Explain why this stride prevents memory coalescing and estimate the wasted bandwidth. |

### 6.2.2 Achieving coalesced access

To fix the memory access pattern, we need to fundamentally change our parallelization strategy. Instead of assigning one thread to compute an entire output element, we'll assign an entire thread block to compute a single output element. This means that all threads within a block (for example, 32 threads if we launch with a block size of

32) will collaborate on the dot product for a single row. Let's examine the new access pattern. In our implementation in book.cu/4\_optim/ gemv/kernels/2\_coalesced\_warp.cu, we launch M blocks (one per output element), each with 32 threads. Consider block 0, which computes output element y[0]:

```
Block 0 computes y[0] (the dot product of row 0 with vector x)
First iteration (col = tid):
Thread 0: accesses matd[0 * N + 0] and vecd[0]
Thread 1: accesses matd[0 * N + 1] and vecd[1]
Thread 2: accesses matd[0 * N + 2] and vecd[2]
...
Thread 31: accesses matd[0 * N + 31] and vecd[31]
Second iteration (col = tid + 32):
Thread 0: accesses matd[0 * N + 32] and vecd[32]
Thread 1: accesses matd[0 * N + 33] and vecd[33]
...
```

Figure 6.2 illustrates the contrast between these two access patterns.

Now adjacent threads (0, 1, 2, …) access consecutive memory locations! All 32 threads in the warp access elements that are adjacent in memory. The memory controller can coalesce these 32 separate 4-byte reads into just one or two wide memory transactions, dramatically improving bandwidth utilization. Since multiple threads are now working on the same dot product, each thread calculates a partial sum. We then need to "reduce" these partial sums into a single final sum

-----

###### 132 Chapter 6 Optimizing our transformer kernels

###### Uncoalesced (naive) Coalesced (optimized)

|  | Col 0 | Col 1 | Col 2 | Col 3 |  | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|---|---|---|---|---|
| Row 0 | A[0,0] | A[0,1] | A[0,2] | A[0,3] | Row 0 | A[0,0] | A[0,1] | A[0,2] | A[0,3] |
| Row 1 | A[1,0] | A[1,1] | A[1,2] | A[1,3] | Row 1 | A[1,0] | A[1,1] | A[1,2] | A[1,3] |
| Row 2 | A[2,0] | A[2,1] | A[2,2] | A[2,3] | Row 2 | A[2,0] | A[2,1] | A[2,2] | A[2,3] |
| Row 3 | A[3,0] | A[3,1] | A[3,2] | A[3,3] | Row 3 | A[3,0] | A[3,1] | A[3,2] | A[3,3] |

T0 T1 T2 T3 T0 T1 T2 T3

Stride = N between threads Stride = 1 between threads

32 separate memory transactions 1 coalesced memory transaction

**Figure 6.2 Coalesced versus uncoalesced memory access in GEMV. The naive kernel (left) assigns each thread to a different row, creating stride-N accesses that waste cache lines. The optimized kernel (right) assigns all threads to the same row, enabling contiguous reads that coalesce into a single wide transaction.**

for the output element. This reduction can be done efficiently using warp-level primitives, specifically \_\_shfl\_down\_sync, which allows threads within a warp to exchange data without using slow shared memory. The code for this improved kernel can be found in book.cu/4\_optim/gemv/kernels/2\_coalesced\_warp.cu. Listing 6.2 shows the warpReduceSum helper function. It uses a loop to progressively sum values across the warp. In each iteration, a thread retrieves the value from the thread whose lane ID is offset higher than its own (for example, lane 0 gets the value from lane 16 when offset is 16). This process halves the number of active threads carrying partial sums until thread 0 has the total sum.

| Listing 6.2 Warp-level sum reduction __device__ __forceinline__ | Warp-level sum reduction __device__ __forceinline__ float warpReduceSum(float val) { | instruction retrieves |
|---|---|---|
| val += | __shfl_down_sync(0xffffffff, val, offset); |  |
| } |  | a value from a thread with a higher lane ID |

`return val;` **The loop halves the offset in each within the same warp.** `}` **iteration, performing a classic parallel**

**reduction in logarithmic time.**

-----

###### GEMV 133

Listing 6.3 shows the main kernel. Each block is assigned a row, and each thread within the block computes a partial sum by iterating through the columns using a grid-stride loop. This loop structure ensures that the memory accesses are coalesced.

**Listing 6.3 Coalesced partial sum calculation**

```
int row = blockIdx.x;
```

| if (row >= M) return; int tid = threadIdx.x; | Each block is responsible for one row, so blockIdx.x is the row index. | Each thread calculates a partial sum for its portion |
|---|---|---|
| float partial_sum = 0.f; for (int col = tid; col < N; col += blockDim.x) { |  | of the row. |

```
partial_sum += matd[row * N + col] * vecd[col];
}
```

###### The grid-stride loop (col += blockDim.x) ensures adjacent threads access adjacent memory, achieving coalesced reads.

Finally, after each thread has its partial\_sum, the kernel calls our helper to reduce the sums across the warp, and thread 0 writes the final result:

```
float sum = warpReduceSum(partial_sum); All threads in the warp call
```

###### Thread 0 of the warp warpReduceSum, but the final result

`if (tid == 0) {` **is meaningful only in thread 0.**

`resd[row] = sum;` **writes the final sum to** `}` **global memory to avoid**

**a race condition.**

The \_\_shfl\_down\_sync instruction is a powerful tool. It allows a thread to get the value of a variable from another thread in the same warp. In our warpReduceSum, we repeatedly add the value from a thread offset lanes away, so only a handful of shuffle steps are required. While warp-level shuffles live entirely in registers, it's worth grounding the concept with a recap of a shared-memory reduction. Imagine four threads writing their inputs to shared memory, synchronizing, and then letting the first two threads compare (0,2) and (1,3). After another barrier, thread 0 compares the survivors to emit the blockwide max. That's all a reduction is: a tree of pairwise combines with \_\_syncthreads() between each level. We'll use the same tree, just with warp shuffles instead of shared memory when we optimize. This version is significantly faster than our naive kernel because it uses the GPU's memory system correctly. The improvement comes from two sources: **1** Coalesced memory access: We now use 80% to 90% of each fetched cache line

instead of less than 5%. **2** Efficient warp-level reduction: The \_\_shfl\_down\_sync operations execute in a

handful of cycles with no memory overhead The result is roughly a 4 to 5× speedup over the naive implementation, with memory bandwidth utilization jumping from about 12% to about 55% of peak.

-----

###### 134 Chapter 6 Optimizing our transformer kernels

###### Exercise: WARP reduction implementation

Write a warpReduceMax function using \_\_shfl\_down\_sync that finds the maximum value across a warp. Test it by creating a vector of 32 random values and verifying the result matches a sequential scan. Extend it to track both the maximum value and its index within the warp.

### 6.2.3 Profiling the coalescing improvement

The following listing compares the Nsight Compute output for the naive and coalesced kernels so we can quantify the bandwidth gains.

**Listing 6.4 Nsight Compute profiling: Naive vs. coalesced GEMV**

```
cd book.cu/4_optim/gemv
ncu --set full \
--section SpeedOfLight_HierarchicalDoubleRooflineChart \
--section MemoryWorkloadAnalysis \
--section Occupancy \
--kernel-name naive_sgemv_kernel \
--print-summary per-kernel \
./gemv_test
```

The profiling results on the RTX 3090 (M = 16384, N = 8192) tell the story in a few rows. The naive kernel drives the L1/TEX cache pipeline at 96% throughput, which looks like hard work, yet converts that into only 12% of peak DRAM bandwidth. The GPU keeps reading cache lines but uses only a few bytes of each, so nearly all of that memory traffic is wasted. The coalesced kernel touches the cache pipeline far less and turns it into real bandwidth:

| --- Naive GEMV Duration | (uncoalesced) --- 5.04 ms |  |
|---|---|---|
| DRAM Throughput | 11.7 % | Useful bandwidth: almost none |
| L1/TEX Cache Throughput | 95.6 % |  |
| Compute (SM) Throughput | 2.3 % | Cache pipe maxed on wasted traffic |
| Achieved Occupancy | 65.6 % |  |
| --- Coalesced Warp GEMV --- |  |  |
| Duration | 1.08 ms | 4.7x faster |
| DRAM Throughput | 54.9 % |  |
| L1/TEX Cache Throughput 10.9 % Real bandwidth climbs |  |  |
| Compute (SM) Throughput | 10.7 % | Far fewer cache transactions |
| Achieved Occupancy | 32.6 % |  |

The surprise is where the "busyness" lives. At first glance the naive kernel looks busy: its L1/TEX cache pipeline runs at 96%, as if the memory system were fully engaged. But that activity is almost entirely wasted. Because adjacent threads read addresses a full row apart, every warp scatters its accesses across many cache lines and uses only a

-----

###### GEMV 135

fraction of each one, so the pipeline is saturated at 96%, while real DRAM bandwidth sits at just 12% and the compute units idle at 2%. The coalesced kernel inverts this. It touches the L1/TEX pipeline far less (11%) because adjacent threads now read adjacent addresses, and each transaction is fully used. Fewer, fuller transactions translate directly into real bandwidth (55% of peak) and a 4.7× lower runtime (5.04 ms to 1.08 ms). This highlights a crucial profiling lesson: high utilization of one unit does not mean good performance. The naive kernel keeps the cache pipeline saturated with wasteful work; the coalesced kernel does far less cache work but gets the answer faster. But the coalesced warp version is limited because the block size must be exactly

```
warpSize (32). To use more threads per block and gain more parallelism, we would
```

need to combine this warp-level reduction with a shared memory-based reduction at the block level. With only 32 threads working on each output element, we're not extracting as much parallelism as we could. Larger blocks would allow us to process more of the row simultaneously, better hiding memory latency. We'll explore that next.

| Exercise: Occupancy optimization |
|---|
| Your kernel uses 48 registers per thread and 16 KB of shared memory per block, |
| with 256 threads per block. Assume a hypothetical GPU with 65,536 registers per SM, 100 KB of shared memory per SM, a maximum of 2,048 threads per SM, and 32 threads per warp (real parts differ: the RTX 3090 pairs 100 KB of shared memory with a 1,536-thread limit). Propose changes to improve occupancy while maintaining performance. |

### 6.2.4 Scaling up with block-level reduction

Our warp shuffle version is efficient, but it can't use more than 32 threads per block. To increase parallelism and computational density, we want to use larger thread blocks (e.g., 128, 256, or more threads). To do this, we need a way to reduce values across the entire block, not just within individual warps. This is where shared memory comes in. Shared memory is a small, on-chip memory space with much lower latency and higher bandwidth than global memory. It is private to each thread block, and threads within the same block can use it to cooperate and share data.

Our strategy will be a two-stage reduction: **1** Warp-level reduction: First, we perform the same warpReduceSum as before. Each

warp of 32 threads reduces its partial sums to a single value. **2** Shared memory reduction: The first thread of each warp (the "warp leader") writes its warp's sum to a designated spot in shared memory. Then, the first warp of the block performs a final reduction on these values in shared memory to get the total sum for the block.

-----

###### 136 Chapter 6 Optimizing our transformer kernels

Figure 6.3 provides the architectural backdrop for this discussion. It shows a streaming multiprocessor (SM) with its warp schedulers, register file, shared memory/L1 cache, and execution units, emphasizing that shared memory sits alongside the schedulers that coordinate those warp-level reductions. We lean on that fast, block-local memory to bridge the gap between warp-level and block-level collaboration.

|  | L1 | instruction cache |
|---|---|---|
| L0 instr cache Warp scheduler | Ill | L0 instr cache Warp scheduler |
| Dispatch unit Registers | Il | Dispatch unit Registers |

TENSOR

| INT32 | FP32 | INT32 | FP32 |
|---|---|---|---|
| INT32 | FP32 | INT32 | FP32 |

TENSOR

INT32 FP32 CORE INT32 FP32 CORE FP64 FP64 FP64 FP64

| LD/ | LD/ | LD/ | LD/ | LD/ | LD/ | LD/ | LD/ |
|---|---|---|---|---|---|---|---|
| ST | ST | ST | ST | ST | ST | ST | ST |

| L0 instr cache | Ill | L0 instr cache |
|---|---|---|
| Warp scheduler | [Ill | Warp scheduler |
| Dispatch Unit Registers | IL | Dispatch Unit Registers |

| INT32 | FP32 | FP32 |  | INT32 | FP32 | FP32 |  |  |
|---|---|---|---|---|---|---|---|---|
| INT32 | FP32 |  | TENSOR | INT32 | FP32 |  | TENSOR |  |
| INT32 | FP32 |  | CORE | INT32 | FP32 |  | CORE |  |
| FP64 | FP64 |  |  | FP64 | FP64 |  |  | Figure 6.3 GPU SM |
| LD/ | LD/ | LD/ | LD/ | LD/ | LD/ | LD/ | LD/ | architecture showing |
| ST | ST | ST | ST | ST | ST | ST | ST | warp schedulers, register file, shared memory/ |

###### L1 cache, and execution

L1 data cache (shared memory) **units. The shared memory**

\_\_shared\_\_ in CUDA **region is where block-level**

**reductions occur.**

This hierarchical reduction pattern combines the speed of register-based warp shuffles for the bulk of the work with the block-wide communication capabilities of shared memory inside the SM.

-----

###### GEMV 137

The full implementation can be found in book.cu/4\_optim/gemv/kernels/3\_ coalesced\_warpblock.cu. Let's break down the blockReduceSum helper function, which orchestrates this two-stage reduction. Listing 6.5 shows the first stage, which is the same warp-level reduction we've already seen. After that, the "leader" of each warp (the thread with lane ID 0) writes its warp's partial sum to an array in shared memory.

| Listing 6.5 Block reduction: | Stage | 1 |  |
|---|---|---|---|
| int warp_size = 32; val = warpReduceSum(val); if (tid % warp_size == 0) __syncthreads(); A In the second stage, as shown final reduction on the partial | smem[tid ensures shared in the sums | Performs level / __syncthreads() all memory that | the now-familiar warp- reduction on the input val Each warp's leader thread (tid % warp_size warp_size] = val; == 0) writes the result to a unique barrier is critical here; it location in the shared warps have finished writing to memory array. before any thread proceeds. following listing, the block's first warp performs a are now in shared memory. |
| Listing 6.6 Block reduction: | Stage | 2 |  |
| if (tid < val = smem[tid]; } else { val = 0.0f; } if (tid / warp_size == 0) { val = } if (tid == 0) smem[0] = __syncthreads(); | CEIL_DIV(block_size, warpReduceSum(val); val; | Another and | The first k threads, where k is the number of warps in the block, load warp_size)) { the partial sums from shared memory. Other threads load 0. The first warp (tid / warp_size == 0) performs one final warp-level reduction on these values. Thread 0 writes the final, total sum for the block into smem[0]. __syncthreads() ensures this final write is complete visible to all threads before the kernel finishes. |

Listing 6.7 shows the main kernel logic, which is now quite simple. It calculates its partial sum as before, calls our new blockReduceSum helper, and then thread 0 writes the final result from shared memory.

**Listing 6.7 Main kernel with block reduction**

```
extern __shared__ float smem[];
int bid = blockIdx.x;
if (bid >= M) return;
int tid = threadIdx.x;
float partial_sum = 0.f;
```

**extern \_\_shared\_\_ declares a dynamically sized shared memory array; its actual size is specified at kernel launch. The partial sum calculation is identical to our previous kernel.**

-----

###### 138 Chapter 6 Optimizing our transformer kernels

```
for (int col = tid; col < N; col += blockDim.x) {
partial_sum += matd[bid * N + col] * vecd[col]; A single call to our helper
```

`}` **function handles the entire**

**complex reduction.**

```
blockReduceSum(partial_sum, smem, tid, blockDim.x); The block-level reduction
```

**leaves one combined partial**

`if (tid == 0) {` **Thread 0 reads that total sum per row in smem[0].** `float sum = smem[0];` **and stores the row result**

`resd[bid] = sum;` **into resd[bid].**

```
}
```

With this kernel, we can now launch with larger block sizes, allowing the GPU to hide latency more effectively and process the dot product with more parallelism. In our implementation (book.cu/4\_optim/gemv/kernels/3\_coalesced\_warpblock.cu), we use 64 threads per block, but this value could be increased to 128 or 256 depending on the matrix dimensions. Doubling the active threads means we move twice as much data per iteration, keep more memory requests in flight to cover latency, and give the scheduler room to juggle work instead of idling. In practice, the block-level reduction is about 1.7× faster than the warp-only version, pushing memory bandwidth utilization to roughly 85% to 90% of peak and delivering about 8× the throughput of the naive kernel.

### 6.2.5 Maximizing bandwidth with vectorization

We have successfully addressed memory coalescing, but we can still squeeze more performance out of our memory operations. Modern GPUs have wide memory buses, and we can saturate this bandwidth more effectively by issuing wider memory loads. Instead of loading one 32-bit float at a time, we can load a 128-bit chunk, which corresponds to four floats. This is called vectorization. CUDA provides built-in vector types like float2 and float4 that map directly to wide load/store instructions. By reinterpreting our float\* pointers as float4\* pointers, we can direct the compiler to generate these more efficient instructions. This reduces the total number of memory instructions, which can save latency and improve throughput, especially in memory-bound kernels like GEMV. Of course, this technique requires the dimensions of our matrices and vectors to be divisible by the vector size (in this case, 4) and the data to be properly aligned in memory. You can see this implemented in book.cu/4\_optim/gemv/kernels/4\_vectorized.cu. Let's modify our kernel to incorporate float4 loads. First, we reinterpret our input float pointers as float4 pointers. This tells the compiler it's safe to use 128-bit load instructions:

###### Calculates the number of float4

| int n_float4s = N / 4; | chunks in our input vector float4* mat_row = reinterpret_cast<float4*>(matd + bid * N); | reinterpret_cast changes the pointer type |
|---|---|---|
| float4* vec = reinterpret_cast<float4*>(vecd); |  | without changing the underlying address. |

-----

###### GEMV 139

Listing 6.8 shows the main loop, which now iterates over these float4 chunks. Inside the loop, we load four floats from the matrix and four from the vector with just two instructions, then compute their dot product and add the result to our partial sum.

**Listing 6.8 Vectorized dot product calculation**

`float partial_sum = 0.f;` **The loop now strides over float4 elements, so**

**each thread processes four floats per iteration.**

```
#pragma unroll 4
for (int col = tid; col < n_float4s; col += blockDim.x) {
float4 matval = mat_row[col];
```

`float4 vecval = vec[col];` **A single read from mat\_row[col] and**

**vec[col] fetches four floats at a time.**

```
partial_sum += (matval.x * vecval.x +
```

| matval.y * vecval.y + | Calculates the dot product |
|---|---|
| matval.z * vecval.z + | of the four pairs of floats and |
| matval.w * vecval.w); | accumulates the result |

```
}
```

The rest of the kernel, including the block-level reduction and the final write, remains completely unchanged. The vectorized version in book.cu/4\_optim/gemv/kernels/4\_ vectorized.cu represents a mature and highly optimized GEMV kernel.

###### Performance effects of vectorization

Vectorization provides a smaller final boost, on the order of 5% to 10%, because

the block-reduction kernel is already close to the bandwidth ceiling. Even so, using

```
float4 loads still helps: it lowers the instruction count because a single load brings
```

in four floats, keeps the memory fabric busy with wider transactions, and signals the compiler to apply vector-friendly rewrites. Together, those effects push bandwidth utilization to around 97% of peak, matching cuBLAS for GEMV. The total speedup from

naive to vectorized is roughly 8x.

### 6.2.6 Final GEMV performance analysis

The following listing captures the Nsight Compute invocation we use to profile the vectorized GEMV against cuBLAS.

**Listing 6.9 Nsight Compute profiling: Vectorized GEMV vs. cuBLAS**

```
cd book.cu/4_optim/gemv
```

```
ncu --set full \
--section SpeedOfLight_HierarchicalDoubleRooflineChart \
--section MemoryWorkloadAnalysis \
--section Occupancy \
--kernel-name vectorized_sgemv_kernel \
--print-summary per-kernel \
./gemv_test
```

-----

###### 140 Chapter 6 Optimizing our transformer kernels

Running that profile captures the SpeedOfLight breakdown shown next alongside the cuBLAS profile for comparison. Both kernels reach the same bandwidth ceiling, but they get there with very different occupancy:

```
--- Vectorized GEMV ---
```

| Duration | 609 us |
|---|---|
| DRAM Throughput | 97.2 % |
| Compute (SM) Throughput | 5.6 % |

```
Achieved Occupancy 65.2 % (20 registers/thread)
--- cuBLAS GEMV ---
```

| Duration | 615 us |
|---|---|
| DRAM Throughput | 96.0 % |
| Achieved Occupancy | 24.3 % |

These are our key observations:

¡ Our vectorized kernel matches cuBLAS. Both finish in about 610 μs at roughly

97% of peak DRAM bandwidth. ¡ Occupancy is not the deciding factor. cuBLAS saturates bandwidth at just 24%

achieved occupancy, while our kernel reaches the same ceiling at 65%. Once a memory-bound kernel saturates DRAM, extra occupancy buys nothing: both kernels are limited by how fast the card can move the matrix, not by how many warps are resident. ¡ The progression shows diminishing returns. Naive (5.04 ms) → coalesced (1.08

ms) is a 4.7× jump that captures most of the gain; block reduction (1.08 ms → 650 μs) and vectorization (650 μs → 609 μs) add progressively smaller improvements as the kernel approaches the bandwidth ceiling. ¡ The kernel is bound by bandwidth, not registers. The vectorized kernel uses only

20 registers per thread, so occupancy is not register limited; it reaches 65% occupancy and then stalls at the DRAM bandwidth ceiling, exactly where a well-tuned memory-bound kernel should.

### 6.2.7 GEMV optimization summary

We started with a simple, intuitive implementation and progressively identified and eliminated bottlenecks: **1** Coalescing (about 4.7× improvement): Changed from thread-per-row to blockper-row, ensuring adjacent threads access adjacent memory **2** Block-level reductions (about 1.7× improvement): Enabled larger blocks for better parallelism and latency hiding **3** Vectorization (about 1.05× improvement): Used float4 loads to close the last few

percentage points to the bandwidth ceiling

-----

###### Softmax 141

The optimization progression demonstrates a crucial principle: for memory-bound kernels, optimize the memory access pattern first. This pattern (coalescing, parallelism, and vectorization) will reappear in every kernel we optimize in this chapter.

| Exercise: Performance debugging with NCU |
|---|
| You've implemented a vectorized GEMV kernel using float4 loads, but profiling |
| shows it's 10% slower than the nonvectorized block reduction version. Using the NCU profiling techniques from this chapter, list three potential causes for this regression. For each cause, specify which NCU metric section you would examine and what val- ues would confirm your hypothesis. |

You can run python book.cu/4\_optim/gemv/main.py to benchmark all versions and see the progressive improvements on your own hardware.

## 6.3 Softmax

The softmax function is a staple in deep learning, most famously used in the attention mechanism of transformers. Its purpose is to convert a vector of raw scores, or logits, into a probability distribution. Each element in the output vector is between 0 and 1,

and the sum of all elements is 1.

Mathematically, for a given input vector x of size N, the softmax of the ith element is calculated as follows: softmax(x\_i) = exp(x\_i) / sum(exp(x\_j) for j=1 to N).

A crucial detail for a stable implementation is to subtract the maximum value of

the vector from each element before exponentiating. This prevents numerical overflow, which can easily happen with the exp() function, without changing the result.

```
The modified, stable formula is softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j
- max(x))).
```

In a transformer, this operation is applied to each row of the attention score matrix.

| The three-pass problem |
|---|
| Our naive implementation performs three sequential passes over each row: find- |
| ing the maximum value for numerical stability, calculating the normalization fac- tor by summing exponentials, and then normalizing and writing the output. This means each input element is read from global memory three separate times, cre- ating a severe memory bottleneck compounded by uncoalesced memory access patterns. |
| The optimization journey for softmax will therefore mirror the playbook we estab- lished with GEMV. We will first fix the memory coalescing problems by shifting from a thread-per-row to a block-per-row strategy. Then, we'll reduce redundant memory reads through an algorithmic improvement known as online softmax. From there, we will fuse operations using shared memory and apply warp-level optimizations and vec- torization to maximize performance. |

-----

###### 142 Chapter 6 Optimizing our transformer kernels

We'll progress through five implementations, benchmarking them with book.cu/4\_ optim/softmax/main.py. Our first implementation mirrors the logic of our first GEMV kernel: it's simple and naive, and it assigns one thread to process an entire row of the input matrix. This approach requires three separate passes over the data for each row. You can find this implementation in book.cu/4\_optim/softmax/kernels/0\_naive.cu. The kernel logic is split into three distinct loops. The first loop iterates over the row to find the maximum value, which is needed for numerical stability:

```
int row = blockDim.x * blockIdx.x + threadIdx.x; Each thread is
```

**assigned a row.**

```
if (row < M) {
float m = -1 * INFINITY;
for (int col = 0; col < N; col++) { The first pass finds the
```

`int i = row * N + col;` **maximum value in that row.**

```
m = max(m, matd[i]);
}
```

The second loop calculates the denominator for the softmax calculation: the sum of the exponentials of each element minus the maximum value:

```
float L = 0.0f;
for (int col = 0; col < N; col++) {
```

`int i = row * N + col;` **The second pass calculates** `L += expf(matd[i] - m);` **the sum of exp(x - max).**

```
}
```

Finally, the third loop divides each exponentiated element by the sum, writing the final probability to the output matrix. Each of these loops reads the entire row from slow global memory:

```
for (int col = 0; col < N; col++) {
```

`int i = row * N + col;` **The third pass normalizes each**

```
resd[i] = expf(matd[i] - m) / L; element and writes the result.
}
}
```

Each thread reads the same row of data from global memory three times. Global memory is the slowest memory on the GPU, and this repeated access creates a significant memory bottleneck. As with our naive GEMV, the memory accesses are not coalesced when threads from a warp access different rows. The most glaring inefficiency is reading the input row three times. We can do much better by having all threads in a block collaborate on a single row, using shared memory to store the row data once and avoid redundant global memory reads. This concept is often called kernel fusion, where multiple logical steps are combined into a single CUDA kernel.

-----

###### Softmax 143

### 6.3.1 Online softmax: A better sequential algorithm

Before we jump to a fully parallel implementation, we can make a significant algorithmic improvement. The three-pass approach is wasteful. By using a clever numerical stability trick known as the online algorithm, we can compute the max and the sum in a single pass. The idea is to maintain a running maximum m and a running normalization factor L. When we encounter a new input x that is greater than the current maximum, we know our previous running sum is incorrect. Instead of restarting, we can "rescue" it by rescaling it relative to the new maximum.

###### The online algorithm's key insight

The online softmax exploits the multiplication property of exponentials: e^a \* e^b =

```
e^(a + b). When we encounter a new maximum value, we multiply the existing normal-
```

ization sum by a correction factor of e^(old\_max - new\_max). This rescales all previous contributions to be relative to the new maximum without restarting the computation. The correction term ensures numerical correctness while allowing us to fuse two passes into one. This technique is fundamental to Flash Attention, which we'll explore in chapter 8. To show that this works, we can walk through a simple example. The goal here is to verify that the two-pass "online" method is numerically identical to the three-pass naive method. Let's take a simple input vector x = [1.0, 2.0, 3.0, 4.0]. First, the naive three-pass approach:

```
Input: x = [1.0, 2.0, 3.0, 4.0]
Pass 1: Find Max
max_val = 4.0
Pass 2: Calculate Sum of Exponentials
x - max_val = [-3.0, -2.0, -1.0, 0.0]
exp(x - max_val) = [0.0498, 0.1353, 0.3679, 1.0]
sum = 0.0498 + 0.1353 + 0.3679 + 1.0 = 1.553
Pass 3: Normalize
y = [0.0498/1.553, 0.1353/1.553, 0.3679/1.553, 1.0/1.553]
y = [0.032, 0.087, 0.237, 0.644]
```

Now, let's trace the online algorithm with the same input. We'll track the state of the running maximum m and the running sum L at each step of the first pass. As you can see, the final results are identical. The online algorithm correctly computes the necessary statistics in a single pass, which allows us to write a more efficient two-pass kernel. You can find this implementation in book.cu/4\_optim/softmax/kernels/1\_online .cu. The following listing shows the fused kernel.

-----

###### 144 Chapter 6 Optimizing our transformer kernels

| Listing 6.10 Online softmax: Fusing the max and sum passes |
|---|
| float m = -INFINITY; float L = 0.0f; We still use a single thread per row for (int col = 0; col < N; col++) { and iterate through all columns. int i = row * N + col; float curr = matd[i]; If we find a new maximum value… if (curr > m) { L = L * expf(m - curr); …we rescale the existing sum m = curr; L relative to this new max. } L += expf(curr - m); Adds the newly exponentiated } value to the (now correctly scaled) for (int col = 0; col < N; col++) { running sum int i = row * N + col; resd[i] = expf(matd[i] - m) / L; } The online algorithm typically provides about a 20% speedup over the naive three-pass approach by reducing global memory reads from 3N to 2N per thread and improving instruction cache locality with fewer loop structures. However, both naive and online implementations still suffer from uncoalesced memory access with threads in a warp reading from different rows, single-thread-per-row limiting parallelism, and no shared memory reuse. The next optimization addresses these fundamental issues by moving to a block-per-row strategy. |
| Exercise: Online softmax verification |
| Implement both the three-pass naive softmax and the two-pass online softmax |
| using Python and NumPy. Test with the vector [5.0, 2.0, 8.0, 1.0, 6.0]. Ver- ify they produce identical results. Walk through the online algorithm step by step, showing how the running maximum and normalization factor evolve as you process each element. |
|  |
|  |

###### Softmax: A concrete numerical example

The following listing begins the step-by-step Python walk-through that validates the online softmax algorithm on a toy dataset.

**Listing 6.11 Python walk-through for the online softmax example**

```
Input tensor X (batch=2, features=4):
[[1.0, 2.0, 3.0, 4.0],
[5.0, 6.0, 7.0, 8.0]]
```

For row 0, we first find the maximum value for numerical stability:

```
max_val = 4.0
```

-----

###### Softmax 145

Next, we subtract this maximum from each element and exponentiate the result. This step prevents numerical overflow while preserving the relative relationships between values:

```
shifted = [1.0-4.0, 2.0-4.0, 3.0-4.0, 4.0-4.0]
= [-3.0, -2.0, -1.0, 0.0]
exp_vals = [exp(-3.0), exp(-2.0), exp(-1.0), exp(0.0)]
= [0.0498, 0.1353, 0.3679, 1.0000]
```

The next step is to sum all the exponentiated values. This sum serves as the normalization denominator that ensures the final probability distribution sums to 1:

```
sum_exp = 0.0498 + 0.1353 + 0.3679 + 1.0000 = 1.553
```

Finally, we normalize each exponentiated value by dividing by the sum to get probabilities:

```
softmax_row0 = [0.0498/1.553, 0.1353/1.553, 0.3679/1.553, 1.0000/1.553]
= [0.0321, 0.0871, 0.2369, 0.6439]
```

The result sums to 1.0, forming a valid probability distribution. Row 1 follows the same pattern with shifted values:

```
Final softmax output:
[[0.0321, 0.0871, 0.2369, 0.6439],
[0.0321, 0.0871, 0.2369, 0.6439]]
Verification - sum per row: [1.0, 1.0] ✓
```

The naive kernel performs three passes that match our example: finding the max, calculating the sum of exponentials, and normalizing. The online algorithm fuses the first two passes by maintaining running statistics. The fused shared memory kernel further parallelizes the work by having multiple threads collaborate on each row with coalesced access patterns. Profiling the online softmax kernel with NCU reveals a telling result: despite reducing the process from three passes to two, the speedup is modest (around 20%). The occupancy metrics explain why: both the naive and online kernels launch just one thread per row, so the SM stays nearly idle (well under 10% occupancy). The algorithmic improvement is real, but the parallelism bottleneck dominates. Our next step attacks this directly by assigning an entire warp to each row.

### 6.3.2 Parallel softmax with warp shuffles

Following the pattern we established in the GEMV kernel, our first step in parallelizing the row-wise computation is to use a block-per-row strategy. The threads in a block first

-----

###### 146 Chapter 6 Optimizing our transformer kernels

collaborate to find the row's maximum value, and then collaborate again to find the normalization sum.

Instead of immediately jumping to a shared memory implementation, we can first build a kernel that relies only on warp-level primitives (\_\_shfl\_down\_sync). This combines the coalesced access of a block-per-row approach with the high-speed reductions we've already seen. We implement this in book.cu/4\_optim/softmax/kernels/3\_shfl.cu.

Listing 6.12 shows the logic, which is a more complex version of our previous reductions. Each thread computes a local max and sum. Then, we perform a warp-level reduction. If the block has multiple warps, the warp leaders use shared memory to share their results, and the first warp performs a final reduction on those results.

**Listing 6.12 Warp-level max and sum reduction**

###### A standard warp-level reduction for the max

`float val = local_max;` **value is performed by all warps in the block.**

```
for (int offset = warp_size / 2; offset > 0; offset /= 2) {
val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
}
```

`if (blockDim.x > warp_size) {` **If the block is larger than a single**

`if (tid % warp_size == 0) {` **warp, the warp leaders write their**

```
smem[tid / warp_size] = val; results to shared memory.
}
```

`__syncthreads();` **A barrier ensures all warp leaders**

**have written their results.**

```
if (tid < warp_size) {
val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] :
-INFINITY;
for (int offset = warp_size / 2; offset > 0; offset /= 2) {
val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
}
```

`if (tid == 0) smem[0] = val;` **The first warp then reads the** `}` **results from shared memory and** `} else {` **performs a final reduction.**

```
if (tid == 0) smem[0] = val;
}
__syncthreads();
```

###### Understanding hierarchical reductions

The shared memory reduction follows a logarithmic pattern to combine N values into one. Consider reducing eight values initially stored as [3, 7, 2, 8, 6, 4, 5, 1]. In the first iteration with stride = 4, thread 0 compares values at positions 0 and 4, computing max(3, 6) = 6. Similarly, thread 1 computes max(7, 4) = 7, thread 2 computes max(2, 5) = 5, and thread 3 computes max(8, 1) = 8. The result is [6, 7, 5, 8,

```
_, _, _, _].
```

-----

###### Softmax 147

| In the second iteration with stride = 2, thread 0 computes max(6, 5) = 6 and thread 1 computes max(7, 8) = 8, yielding [6, 8, _, _, _, _, _, _]. |
|---|
| In the third iteration with stride = 1, thread 0 computes max(6, 8) = 8, giving the final result [8, _, _, _, _, _, _, _]. |
| This takes only log2(N) steps instead of N-1 sequential comparisons. For 256 threads, that's 8 steps instead of 255. The __syncthreads() barriers are critical because they ensure all threads complete each step before any thread proceeds to the next, pre- venting race conditions on shared memory. |

The shfl\_down\_sync instruction exchanges data between threads in a warp using the register file, bypassing shared memory entirely. This provides several benefits. First, there are no shared memory bank conflicts because each thread reads from its own registers. Second, there are no synchronization barriers within warps because warps execute in lockstep. Third, register file access has much lower latency than shared memory access.

The hierarchical approach, using warp-level reductions followed by block-level reductions, minimizes shared memory usage. With 256 threads organized into 8 warps, we need only 8 values in shared memory for the final reduction. Compare this to a naive block reduction, which would require 256 values in shared memory. This optimization typically improves performance by 15% to 25% over pure shared memory reductions, which is particularly important for small to medium row sizes where shared memory bandwidth can become a bottleneck.

### 6.3.3 Fused softmax with shared memory

To fix the massive inefficiency of the naive kernel, we will adopt the same strategy that worked so well for GEMV: assign one thread block per row. This allows all threads in the block to work together on the softmax calculation for that single row. This layout gives us three wins at once. First, each thread streams a slice of the row from global memory with perfectly coalesced reads. Second, every thread carries the online max-and-norm trick from the previous section, so one read pass produces both statistics. Third, the threads in the block combine their per-thread results through fast shared memory reductions, the same pattern we used for GEMV. This approach effectively "fuses" the three passes of the naive kernel into a single, cohesive unit. The code for this can be found in book.cu/4\_optim/softmax/kernels/2\_ sharedmem.cu. This kernel is significantly more complex than the naive version, but the performance gains are substantial. It streams the row once with an online first pass, performs two block-wide tree reductions in shared memory, and finishes with a normalization pass. Let's examine each step.

-----

###### 148 Chapter 6 Optimizing our transformer kernels

In listing 6.13, each thread streams its slice of the input row directly from global memory, maintaining a running maximum and normalization factor with the online update. Shared memory holds only per-thread partial results for the reductions that follow, never the row itself, so nothing here limits the row size N.

**Listing 6.13 Fused softmax: Online first pass**

```
__shared__ float smem[1024];
```

###### Sizes shared memory for the block's

| int row = blockIdx.x; | Each block handles | 1,024 threads. It holds one partial result |
|---|---|---|
| int tid = threadIdx.x; if (row >= M) return; | one row. | per thread during the reductions, not the row, so N can be arbitrarily large. |

```
float* input_row = xd + row * N;
float* output_row = resd + row * N;
float local_max = -INFINITY;
```

`float local_norm = 0.0f;` **A grid-stride loop: consecutive**

**threads read consecutive elements,**

```
for (int i = tid; i < N; i += blockDim.x) { so the reads are coalesced.
float x = input_row[i];
if (x > local_max) {
```

`local_norm *= expf(local_max - x);` **The online softmax update from the** `local_max = x;` **previous section, now running inside** `}` **each thread, rescales the running norm**

```
local_norm += expf(x - local_max); whenever a new maximum appears.
}
__syncthreads();
```

The following listing shows the next step: each thread publishes its local\_max to shared memory, and the block runs a tree reduction to find the row-wide maximum.

**Listing 6.14 Fused softmax: Parallel max reduction**

`smem[tid] = local_max;` **A classic shared-memory tree reduction: each step halves** `__syncthreads();` **the number of active threads until one value remains.**

```
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
if (tid < stride) {
smem[tid] = max(smem[tid], smem[tid + stride]);
}
```

`__syncthreads();` **The row-wide maximum lands in** `}` **smem[0], where every thread can read it.**

| float row_max = smem[0]; | Another barrier ensures all threads see |
|---|---|
| __syncthreads(); | the correct row_max before shared memory is reused for the next reduction. |

With the row maximum found, the block performs a second, similar reduction to produce the normalization factor. There is one subtlety: each thread's local\_norm was accumulated relative to its own local\_max, so it must first be rescaled relative to the

-----

###### Softmax 149

shared row\_max, using the same correction factor the online algorithm applies within a thread:

###### Rescales each thread's partial norm to the global maximum

**before summing. Without this correction, partial sums computed against different maxima would be meaningless to add.**

```
smem[tid] = local_norm * expf(local_max - row_max);
__syncthreads();
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
if (tid < stride) {
smem[tid] += smem[tid + stride];
}
__syncthreads();
```

| } | A tree reduction identical in shape to the |
|---|---|
| float row_norm = smem[0]; | max reduction, but summing instead |

```
__syncthreads();
```

Finally, with row\_max and row\_norm known to all threads, they perform the final normalization and write the result directly to global memory:

```
for (int i = tid; i < N; i += blockDim.x) {
output_row[i] = expf(input_row[i] - row_max) / row_norm;
}
```

###### A final coalesced grid-stride pass rereads the row and writes each normalized element.

The fused shared memory kernel delivers dramatic improvements. The memory access pattern is now coalesced because consecutive threads read consecutive elements, the row crosses the global memory bus only twice (once for the online statistics pass, once for normalization), and all cross-thread communication happens in fast on-chip shared memory. Because it puts many threads on each row instead of one, this provides a very large speedup over the naive version, roughly 20× on short rows and over 100× on wide ones, with a similar jump over the online algorithm. The warp shuffle and vectorized refinements that follow bring performance to within a few percent of PyTorch, matching or slightly beating it. The shared memory approach trades register usage and synchronization overhead for massive memory bandwidth improvements. For row-wise operations like softmax, this tradeoff is almost always favorable.

### 6.3.4 Vectorized softmax

As a final optimization, just as with GEMV, we can apply vectorization to improve memory bandwidth. By loading data in float4 chunks, we reduce the total number of memory instructions and make more efficient use of the GPU's wide memory bus. The implementation for this optimization can be found in book.cu/4\_optim/softmax/ kernels/4\_vectorized.cu.

-----

###### 150 Chapter 6 Optimizing our transformer kernels

Listing 6.15 shows the logic, which combines our previous techniques. We use a block-per-row strategy with hierarchical reductions, but the initial calculation of each thread's local max and sum is done on float4 chunks.

| Listing 6.15 Vectorized local reduction |  |
|---|---|
| Casts pointers to float4 and loops over the chunks of the input row, striding by blockDim.x chunk is processed by exactly int n_float4s = N / 4; | vectorized so each one thread |
| float4* input_row_vec = reinterpret_cast<float4*>(input_row); |  |
|  |  |
| float maxval = fmaxf(fmaxf(elem.x, elem.y), fmaxf(elem.z, | elem.w)); |

```
if (maxval > local_max) {
local_norm *= __expf(local_max - maxval); Uses the online algorithm
```

`local_max = maxval;` **to update our thread's** `}` **local\_max and local\_norm**

```
local_norm += __expf(elem.x - local_max);
local_norm += __expf(elem.y - local_max);
local_norm += __expf(elem.z - local_max);
local_norm += __expf(elem.w - local_max);
}
```

After this loop, each thread has a local\_max and a local\_norm. The block first reduces the per-thread maxima to find row\_max. Then comes the same subtlety we saw in the shared memory kernel: each thread's local\_norm is relative to its own local\_max, so it must be rescaled before the partial norms can be summed. From here the kernel proceeds exactly as the shfl version, writing the normalized result with the same vectorized float4 stores. This vectorized approach, combined with the fused, parallel reduction strategy, represents a highly optimized softmax kernel.

### 6.3.5 Profiling softmax optimizations

Let's examine how each softmax optimization improves performance using NCU profiling on an RTX 3090 (M = 128, N = 4096). First, profile the naive kernel:

```
cd book.cu/4_optim/softmax
ncu --set full \
--section SpeedOfLight_HierarchicalDoubleRooflineChart \
--section MemoryWorkloadAnalysis \
--section Occupancy \
--kernel-name naive_softmax_kernel \
--print-summary per-kernel \
./softmax_test 128 4096 1
```

The following listing compares the naive kernel directly with the optimized warp shuffle version, highlighting the dramatic difference in occupancy and cache usage.

-----

###### Layer normalization 151

| Listing 6.16 --- Naive Duration | Nsight Compute profiling: Naive vs. warp shuffle softmax Softmax (1 thread/row) --- 1.40 ms | Nsight Compute profiling: Naive vs. warp shuffle softmax |
|---|---|---|
| DRAM Throughput | 0.27 % | Cache pipe busy |
| L1/TEX Cache Throughput | 84.0 % | on wasted work |
| Compute (SM) Throughput | 0.06 % |  |
| Achieved Occupancy 8.4 % (0.75 warps/SM active) |  |  |
| --- Warp Shuffle Softmax (1 warp/row) --- |  |  |
| Duration 9.3 us ~150x faster |  |  |
| DRAM Throughput | 39.6 % |  |
| L1/TEX Cache Throughput | 27.8 % |  |
| Compute (SM) Throughput | 18.0 % |  |
| Achieved Occupancy | 65.7 % |  |

Critical insights are as follows: ¡ Occupancy is the key differentiator: The naive kernel runs one thread per row, so only 0.75 warps are active per SM (8% occupancy) and the SM sits almost idle. The warp shuffle kernel puts 32 threads on each row, lifting occupancy to 66% and cutting runtime from 1.40 ms to 9.3 μs, about 150× faster. ¡ An online algorithm provides minimal benefit alone: going from 3-pass to 2-pass (naive → online) saves only about 20% because both still use a single thread per row with the same occupancy. ¡ Busy cache, idle GPU: Like the naive GEMV kernel, the naive softmax keeps the L1/TEX pipeline at 84% while converting it into just 0.27% of peak DRAM bandwidth and near-zero compute. The warp kernel does far less cache work but lifts real DRAM utilization to 40%t and compute to 18%.

### 6.3.6 Softmax optimization summary

The online algorithm provides about a 1.2× improvement through algorithmic fusion that reduces passes from 3 to 2. Moving to shared memory with a block-per-row layout is the giant step, roughly 90 to 120× at this shape, because it finally parallelizes each row instead of assigning it to a single thread. Warp shuffles add a further ~1.5× by replacing shared-memory reductions with register shuffles, and vectorization adds up to ~15% more at larger N by using float4 loads. The total speedup from naive to vectorized is on the order of 150 to 200× at M = 128,

```
N = 4096 and grows with row width. The pattern is identical to GEMV: fix memory access
```

patterns, maximize parallelism through hierarchical reductions, and saturate bandwidth with vectorization.

You can benchmark all kernels with python book.cu/4\_optim/softmax/main.py.

## 6.4 Layer normalization

Layer normalization (LayerNorm) is another ubiquitous operation in transformers. It normalizes the outputs of a layer across all of its features for a single item in a batch.

-----

###### 152 Chapter 6 Optimizing our transformer kernels

Unlike batch normalization, its calculations are independent of the batch size, making it well suited for transformers and sequence data.

The operation consists of two main steps. For a given input vector x (representing the features for one token): **1** Standardization: Calculate the mean and variance of x. Use them to shift the distribution to have a mean of 0 and a variance of 1. **2** Affine transformation: Apply a learned element-wise scaling (gamma) and shift

(beta) to the standardized vector. The formula is y\_i = gamma\_i \* (x\_i - mean(x)) / sqrt(variance(x) + epsilon) + beta\_i.

At a high level, the computational pattern of LayerNorm is remarkably similar to softmax. Both operations require calculating summary statistics (mean and variance for LayerNorm; max and sum for softmax) across a row of data, and then using those statistics to normalize each element in the row. This structural similarity means that all the optimization principles we just applied to softmax (fusing passes, using a block per row, using shared memory, and performing parallel reductions) are directly applicable here.

| LayerNorm's computational pattern |
|---|
| LayerNorm requires three sequential computations. First, we calculate the mean |
| using mu = (1/n) * sum(x_i). Second, we calculate the variance using sigma2 = (1/n) * sum(x_i - mu)2. Third, we normalize and scale using y_i = gamma_i * (x_i - mu) / sqrt(sigma**2 + epsilon) + beta_i. |
| The naive approach reads each row from global memory three times, mirroring the naive softmax problem. The optimization strategies we developed for softmax apply directly: memory coalescing through block-per-row instead of thread-per-row, shared memory fusion by loading the row once and computing mean and variance in shared memory, warp-level reductions using register-based operations for speed, and vector- ization with float4 loads for bandwidth saturation. |
| The key difference from softmax is that LayerNorm computes the mean and variance, which requires two sum reductions, while softmax computes the max and sum, requir- ing one max reduction and then one sum reduction. |

We'll progress through three implementations and benchmark them with book.cu/4\_ optim/layernorm/main.py.

Our starting point is once again a naive two-pass kernel, where a single thread handles an entire row. You can find this implementation in book.cu/4\_optim/layernorm/ kernels/0\_naive.cu.

The kernel begins by having each thread calculate the mean for its assigned row. This involves a full pass over the row data:

-----

###### Layer normalization 153

```
int idx = blockIdx.x * blockDim.x + threadIdx.x; Each thread is assigned
```

**a row to process.**

```
if (idx < N) {
const float* x = inp + idx * C;
float m = 0.0f;
for (int i = 0; i < C; i++) {
```

`m += x[i];` **The first pass: a loop to sum the** `}` **elements, which is then divided by** `m = m / C;` **the count C to get the mean** Next, the thread makes a second pass over the same data to calculate the variance:

```
float v = 0.0f;
for (int i = 0; i < C; i++) {
float xshift = x[i] - m;
v += xshift * xshift;
```

`}` **The second pass calculates the sum of** `v = v / C;` **squared differences from the mean.** Finally, the thread makes a third pass to apply the normalization, scale, and shift, before writing the output:

```
float eps = 1e-5f;
float s = 1.0f / sqrtf(v + eps); Calculates the reciprocal
```

**standard deviation (s)**

```
float* out_idx = out + idx * C;
for (int i = 0; i < C; i++) {
```

`float n = (s * (x[i] - m));` **The third pass normalizes the**

```
float o = n * weight[i] + bias[i]; data and applies the learnable
```

`out_idx[i] = o;` **affine transformation parameters.**

```
}
}
```

###### Performance issues in naive LayerNorm

This kernel suffers from multiple performance problems. First, uncoalesced memory access occurs because each thread processes a different row, with threads 0 and 1 accessing elements N apart, such as x[0] and x[N], causing strided global memory reads. Second, redundant global memory reads happen because the input row is read three complete times: once for mean calculation, once for variance calculation and once for normalization. For a 1,024-element row, that's 3,072 global memory reads per thread. Third, there's no parallelism within rows, as each thread computes serially through its entire row. With typical row sizes of 768 to 4,096 elements, this leaves massive parallelism untapped. Fourth, the poor arithmetic intensity means the ratio of computation to memory access is extremely low, making the kernel memory-bound but utilizing memory inefficiently.

-----

###### 154 Chapter 6 Optimizing our transformer kernels

For typical transformer shapes, this naive approach achieves under 10% of the GPU's memory bandwidth and runs roughly an order of magnitude slower than the optimized implementations.

###### LayerNorm: A concrete numerical example

The following listing starts the Python verification that checks our LayerNorm computations on a 2 × 4 input.

**Listing 6.17 Python walk-through for the LayerNorm example**

```
Input tensor X (batch=2, features=4):
[[1.0, 2.0, 3.0, 4.0],
[5.0, 6.0, 7.0, 8.0]]
```

For row 0 with values [1.0, 2.0, 3.0, 4.0], we first calculate the mean by summing all elements and dividing by the count:

```
mean = (1 + 2 + 3 + 4) / 4 = 2.5
```

Next, we calculate the variance using the formula sigma2 = (1/n) \* sum(x\_i - mu)2. This involves subtracting the mean from each element, squaring the result, summing these squared differences, and dividing by the count. We then take the square root to get the standard deviation, adding a small epsilon for numerical stability:

```
variance = ((1-2.5)**2 + (2-2.5)**2 + (3-2.5)**2 + (4-2.5)**2) / 4
= (2.25 + 0.25 + 0.25 + 2.25) / 4
= 1.25
std = sqrt(1.25 + 1e-5) = 1.118
```

Finally, we normalize each element by subtracting the mean and dividing by the stan-

```
dard deviation, y_i = (x_i - mean) / std:
Normalized row 0:
[(1.0-2.5)/1.118, (2.0-2.5)/1.118, (3.0-2.5)/1.118, (4.0-2.5)/1.118]
= [-1.342, -0.447, 0.447, 1.342]
```

The same process applies to row 1, yielding identical normalized values since the pattern is the same. The output has mean ≈ 0 and variance ≈ 1 for each row:

```
Normalized output:
[[-1.342, -0.447, 0.447, 1.342],
[-1.342, -0.447, 0.447, 1.342]]
```

###### Mapping to GPU kernel logic

Each thread in our naive kernel performs exactly these three passes. The first loop accumulates the sum and divides by C to get the mean. The second loop accumulates

-----

###### Layer normalization 155

the squared differences to compute the variance. The third loop normalizes using the computed statistics. The inefficiency is that each loop rereads the entire row from global memory, which we address by fusing these operations and using faster memory hierarchies.

###### Exercise: Online algorithm extension

The online softmax algorithm maintains running statistics to combine multiple

passes. Design an "online" version of the LayerNorm mean and variance calculation

that computes both statistics in a single pass. Write pseudocode showing how to

maintain a running mean and running sum of squares. Verify that your algorithm produces correct results.

Profiling the parallel LayerNorm kernel shows that launching separate kernels for mean, variance, and normalization creates unnecessary global memory traffic. Each kernel reads the entire row from HBM, processes it, and writes results back: three round trips where one would suffice. A fused kernel that keeps the row in shared memory throughout all three stages eliminates these redundant transfers.

### 6.4.1 Fused LayerNorm with warp operations

The naive kernel's three separate passes over global memory are the clearest target for improvement. We can attack this problem at two levels: splitting the work across multiple kernel launches, or fusing everything into a single warp-based kernel. Let's start with the multikernel approach before examining the fully fused version.

###### Multikernel parallel approach

Before examining the fully fused warp kernel, note that book.cu/4\_optim/layernorm/ kernels/1\_parallel.cu implements a different strategy by splitting LayerNorm into three separate kernel launches. The first kernel calculates the means across all rows in parallel. The second kernel calculates the variances across all rows in parallel. The third kernel normalizes the values and applies the affine transformation. Each kernel uses block-per-row parallelization with reductions, avoiding uncoalesced access. Although this significantly improves upon the naive version, it requires three separate kernel launches, with their associated overhead. The warp-based approach we'll examine next fuses all three stages into a single kernel launch, eliminating kernel launch overhead and intermediate global memory writes. As the following listing shows, the kernel first calculates the mean by having each lane in the warp calculate a partial sum, followed by a warp reduction.

**Listing 6.18 Warp-based mean calculation**

```
const float* x = inp + idx * C;
```

**idx is now the row index, calculated based**

`float sum = 0.0f;` **on the global warp ID. lane\_id is the**

**thread's index within the warp (0-31).**

-----

###### 156 Chapter 6 Optimizing our transformer kernels

```
for (int i = lane_id; i < C; i += WARP_SIZE) {
sum += x[i];
}
sum = warpReduceSum(sum);
float m = __shfl_sync(0xffffffff, sum, 0) / C;
```

###### __shfl_sync broadcasts the final sum from lane 0 to all other lanes in the warp so they all have the correct mean m.

###### Each thread computes a partial sum in a coalesced manner. warpReduceSum (the same helper from GEMV) sums

**the partial sums. The result is in lane 0.**

| Warp-per-row: The sweet spot for medium-sized | rows |
|---|---|
| Assigning one warp of 32 threads to each row is highly | efficient for typical transformer |
| hidden dimensions like 768, 1,024, and 2,048. The substantial: 32 threads load 32 consecutive elements coalescing, where each warp's memory requests reads. For N = 1024 with 32 threads, only 32 iterations 1,024 serial iterations in the naive version. | memory access benefits are per iteration, achieving perfect combine into one or two cache line are needed, compared with |
| Reduction efficiency is excellent because warp log2(32) = 5 dependent steps, five SHFL instructions with barriers. No shared memory is needed, eliminating zation barriers. __shfl_sync broadcasts the final mean | shuffle reductions complete in instead of a shared-memory tree bank conflicts and synchroni- to all lanes instantly. |
| Arithmetic intensity is optimized by loading the row forming mean reduction with 32 adds, broadcasting forming variance reduction with 32 fused multiply-adds, with one shuffle, and applying normalization in | once from global memory, per- the mean with one shuffle, per- broadcasting the variance registers. |
| For rows of 768 to 2,048 elements, warp-per-row approaches because of lower synchronization | often outperforms block-per-row overhead. |
| The following listing repeats the exact same pattern reciprocal standard deviation (rstd). | to calculate the variance and the |
| Listing 6.19 Warp-based variance and final write |  |
| sum = 0.0f; for (int i = lane_id; i < C; i += WARP_SIZE) { float diff = x[i] - m; sum += diff * diff; } sum = warpReduceSum(sum); float s = rsqrtf(__shfl_sync(0xffffffff, sum, 0) float* o = out + idx * C; for (int c = lane_id; c < C; c += WARP_SIZE) { | Performs a second parallel reduction for the sum of squared differences / C + 1e-5f); The result is broadcast from lane 0, and rsqrtf |

```
float n = s * (x[c] - m);
o[c] = n * weight[c] + bias[c];
}
```

**is computed to get s. Finally, each lane computes and writes its portion of the normalized output.**

-----

###### Layer normalization 157

###### Computing variance efficiently

Note that the variance calculation uses the identity sigma2 = E[x2] - E2. This allows us to accumulate both sum(x) and sum(x2) in a single pass. Each thread accumulates

```
sum_x += val for the mean and sum_x2 += val * val for the variance. Then, after broad-
```

casting the mean, we compute variance = (sum\_x2 / C) - (mean \* mean). This approach is efficient and avoids recomputing (x[i] - mean)\*\*2 for each element. Although mathematically equivalent to the two-pass definition, it can be less numerically stable than Welford's online algorithm when the variance is tiny relative to the square of the mean, that is, when the data is tightly clustered far from zero and the two accumulated terms nearly cancel. Still, for standard normalized transformer activations in FP32, it is generally sufficient and much faster.

###### Performance of warp-based LayerNorm

The warp-per-row strategy typically delivers a speedup of roughly 9 to 10× over the naive implementation. Memory bandwidth utilization reaches 85% to 90% of peak. Performance often matches or exceeds PyTorch for typical hidden dimensions. The approach works best when C, the feature dimension, is between 256 and 4,096 elements. The warp-per-row strategy represents the practical optimum for most transformer workloads. Further optimization would require larger blocks for very wide rows exceeding 4,096 features, vectorized loads using float4 for bandwidth saturation, or multistage pipelining for newest GPU architectures.

The warp-per-row strategy's main limitation is that it can effectively use only 32 threads per row. For larger rows, we must move to a block-per-row strategy.

| Exercise: Block-per-row vs. warp-per-row |
|---|
| For LayerNorm with feature dimension C = 1024, explain why the warp-per-row strat- |
| egy (kernels/2_warp.cu) outperforms a hypothetical block-per-row strategy with 256 threads per block. Consider register usage, synchronization overhead, and memory bandwidth. At what feature dimension would block-per-row become advantageous? |

### 6.4.2 Fused LayerNorm kernel

The optimized LayerNorm kernel follows the same pattern as our fused softmax kernel. A thread block is assigned to each row. The row's data is loaded into shared memory once. Then threads collaborate to perform two parallel reductions: first, to find the mean, and second, to find the variance. Finally, the threads perform the normalization, scaling, and shifting, again reading from shared memory before writing the final result to global memory.

This works because the row is read from global memory once instead of three times, and the reductions read from on-chip shared memory.

The code in book.cu/4\_optim/layernorm/kernels/1\_parallel.cu achieves this same goal, but does so by launching three separate kernels instead of fusing them into one.

-----

###### 158 Chapter 6 Optimizing our transformer kernels

A truly fused kernel, as shown later, is generally more efficient as it avoids the overhead of multiple kernel launches. Let's walk through the stages of a fused LayerNorm kernel. The following listing shows the first stage: loading data into shared memory.

**Listing 6.20 Fused LayerNorm: Loading to shared memory**

```
extern __shared__ float smem[];
```

###### Dynamic shared memory

`int row = blockIdx.x;` **array for the input row** `if (row >= N) return;` **Each block processes**

**one row of the input.**

```
int tid = threadIdx.x;
const float* x = inp + row * C;
```

###### Cooperative coalesced load of the

```
for (int i = tid; i < C; i += blockDim.x) { input row into shared memory
smem[i] = x[i];
```

| } | Barrier to ensure the load is complete |
|---|---|
| __syncthreads(); | before computation begins |

Next, we perform the first of two parallel reductions to calculate the mean of the elements now in shared memory:

###### Each thread computes a

`float local_sum = 0.0f;` **partial sum from the values**

```
for (int i = tid; i < C; i += blockDim.x) { in shared memory.
local_sum += smem[i];
```

| } | Reuses the blockReduceSum helper to find the total |
|---|---|
| blockReduceSum(local_sum, smem); | sum across the block. The result is placed in smem[0]. |

```
float mean = smem[0] / C;
```

`__syncthreads();` **Thread 0 could calculate the mean, but having**

**all threads do it avoids another shared memory read or broadcast. A sync is needed to ensure the smem[0] write is complete.**

Then, we perform a second parallel reduction to find the variance:

```
local_sum = 0.0f;
```

| for (int i = tid; i < C; i += blockDim.x) { float diff = smem[i] - mean; |  | Each thread calculates a partial |
|---|---|---|
| local_sum += diff * diff; | Uses the same blockReduceSum | sum of the squared |
| } | helper again to find the total | differences from |
| blockReduceSum(local_sum, smem); float var = smem[0] / C; | sum of squares | the mean. |
| float rstd = rsqrtf(var + 1e-5f); __syncthreads(); | Computes the variance, var, from the accumulated squared differences |  |

**rstd is the reciprocal square root used to normalize. A sync ensures the smem[0] write is complete before proceeding.**

-----

###### Layer normalization

Finally, the threads use the calculated mean and rstd to normalize the data from shared memory and write the final result to global memory:

```
for (int i = tid; i < C; i += blockDim.x) {
float n = rstd * (smem[i] - mean);
out[row * C + i] = n * weight[i] + bias[i];
}
```

###### In a final grid-stride loop, each thread applies the normalization, scaling, and bias, writing out the final value.

The core optimization pattern (load to shared memory, perform parallel reductions, compute) is a powerful template that applies to a wide range of row-wise operations in deep learning, not just softmax and LayerNorm.

###### Exercise: Kernel fusion design

Design a fused kernel for the composite operation y = ReLU(LayerNorm(x)) where ReLU zeros out negative values. Describe your parallelization strategy. How would you minimize global memory traffic? Which reduction patterns from this chapter would you reuse? Sketch the kernel's structure showing the load, compute, and write phases.

### 6.4.3 LayerNorm optimization summary

Our optimization journey for LayerNorm follows three stages. The naive baseline uses thread-per-row with three passes and uncoalesced access. The multikernel parallel approach uses three kernel launches with block-per-row and coalesced access, delivering about 7× improvement. The warp-based fused kernel uses a single kernel with warp shuffles, optimal for transformer dimensions, providing roughly 9 to 10× improvement over naive in a single launch. LayerNorm has less severe numerical stability concerns because it avoids exponentials. Warp-per-row is often optimal for LayerNorm, while block-per-row is better for softmax with large N. Transformer hidden dimensions of768, 1,024, and 2,048 fit well within 32-thread processing. No shared memory means no bank conflicts and no \_\_syncthreads() barriers, since register-based shuffles stay within the warp. A single kernel launch avoids overhead.

You can benchmark all kernels with python book.cu/4\_optim/layernorm/main.py.

### 6.4.4 LayerNorm profiling results

The following listing records the Nsight Compute invocation for the naive LayerNorm kernel when benchmarking at 4096 × 768.

**Listing 6.21 Nsight Compute profiling: Naive vs. warp LayerNorm**

```
cd book.cu/4_optim/layernorm
ncu --set full \
```

-----

###### 160 Chapter 6 Optimizing our transformer kernels

```
--section SpeedOfLight_HierarchicalDoubleRooflineChart \
--section MemoryWorkloadAnalysis \
--section Occupancy \
--kernel-name layernorm_kernel_0 \
--print-summary per-kernel \
./layernorm_test
```

The resulting capture provides the SpeedOfLight summary and occupancy breakdown shown next. Note how underutilized the naive kernel remains compared with the optimized warp kernel, which keeps many more warps resident:

```
--- Naive LayerNorm (1 thread/row) ---
```

| Duration | 580 us |
|---|---|
| DRAM Throughput | 9.4 % |
| L1/TEX Cache Throughput | 77.3 % |
| Compute (SM) Throughput | 1.4 % |
| Achieved Occupancy | 16.6 % |

| --- Warp LayerNorm (1 | warp/row) --- |  |
|---|---|---|
| Duration DRAM Throughput L1/TEX Cache Throughput | 56 us 87.8 % 17.7 % | ~10x faster |
| Compute (SM) Throughput Achieved Occupancy | 16.9 % 83.5 % |  |

Consider the following analysis: ¡ *Parallelization is everything-Moving from 1 thread per row to 32 threads per row* lifts occupancy from 17% to 84% and DRAM utilization from 9% to 88% cutting runtime from 580 μs to 56 μs, an order-of-magnitude speedup. ¡ *Same pattern as softmax-LayerNorm and softmax follow the same optimization* path (thread-per-row → warp-per-row with parallel reductions), and both are transformed by parallelism far more than by any algorithmic change. ¡ *From wasted cache to real bandwidth-The naive kernel keeps the L1/TEX pipeline* at 77% while achieving only 9% DRAM bandwidth; the warp kernel touches the cache pipeline less (18%) yet reaches 88% of peak bandwidth, the mark of a well-coalesced memory-bound kernel.

## 6.5 Top-k

Our journey now takes us to a kernel that is a cornerstone of mixture-of-experts (MoE) models: top-k. In an MoE transformer, after a "gating" network outputs scores for each available "expert," the top-k kernel selects the K experts with the highest scores. The input token is then processed only by these selected K experts, which is how MoE models achieve massive parameter counts while keeping computation manageable.

The task is simple: for a given input vector of scores, find the K largest values and their original indices.

-----

###### Top-k 161

A naive way to implement this on a GPU is to use a variation of selection sort. We can launch a single thread (or a single block) that iterates K times. In each iteration, it scans the entire input vector to find the current maximum value that hasn't already been chosen, records its value and index, and then marks it as "selected" to avoid picking it again. You can find this implementation in book.cu/4\_optim/topK/kernels/0\_naive.cu. Listing 6.22 shows the kernel, which consists of a main loop that runs K times. Inside this loop, a nested loop scans all N elements to find the largest value that has not yet been selected.

**Listing 6.22 Naive top-k: Finding the next max**

```
for (int k = 0; k < K; k++) {
```

`float max_val = -INFINITY;` **The outer loop runs K times** `int max_idx = -1;` **to find K elements.**

```
for (int i = 0; i < N; i++) { The inner loop performs a
if (!selected[i] && input[i] > max_val) { linear scan to find the
```

`max_val = input[i];` **largest value among the** `max_idx = i;` **unselected elements.**

```
}
}
```

After the inner loop finds the maximum, its value and index are recorded, and the index is marked as selected for the next iteration:

###### Guards against the case where no new max is found

```
if (max_idx >= 0) {
```

| selected[max_idx] = true; | Marks the index as true in a temporary Boolean |
|---|---|
| values[k] = max_val; | array to exclude it from future searches |

```
indices[k] = max_idx;
```

`}` **Stores the value and** `}` **index in the output arrays** This approach is terribly inefficient. The complexity is O(K × N), and it is completely sequential. It uses none of the GPU's massive parallelism. The challenge in parallelizing top-k is that it's not an easily divisible problem. All threads need to agree on the global top K elements. A more sophisticated approach is needed, typically involving parallel sorting or heap-based algorithms distributed across the threads of a block. The following listing walks through a Python toy example that illustrates the top-k selection process.

**Listing 6.23 Python walk-through for the top-k example**

```
Input scores (N=8):
[3.2, 7.1, 2.5, 9.3, 4.8, 1.7, 6.4, 5.0]
Indices: [0, 1, 2, 3, 4, 5, 6, 7]
```

-----

###### 162 Chapter 6 Optimizing our transformer kernels

Using the naive selection sort approach with K = 2, we perform sequential scans to find the top values. The algorithm iteratively finds the maximum value among unselected elements:

```
Iteration 1: Scan all 8 elements
Compare: 3.2, 7.1, 2.5, 9.3, 4.8, 1.7, 6.4, 5.0
Found max=9.3 at index=3
Mark index 3 as selected
Iteration 2: Scan remaining 7 elements (skip index 3)
Compare: 3.2, 7.1, 2.5, [skip], 4.8, 1.7, 6.4, 5.0
Found max=7.1 at index=1
Mark index 1 as selected
```

After K iterations, we extract the final results, which contain the top-k values and their original indices:

```
Top-2 values: [9.3, 7.1]
Top-2 indices: [3, 1]
```

The naive kernel runs K outer-loop iterations, each scanning all N elements to find the next maximum among unselected values. This sequential pattern is inherently difficult to parallelize. The parallel version uses block-wide reductions to find the maximum in each iteration, distributing the search across multiple threads. But the

```
K iterations remain sequential, creating a fundamental bottleneck that limits GPU
```

utilization.

NCU profiling of the naive top-k kernel reveals extremely low occupancy: each thread processes an entire row sequentially, leaving most of the SM idle. The kernel is neither compute-bound nor memory-bound; it is simply doing too little parallel work. A better algorithm is needed before any CUDA-specific optimization can help.

### 6.5.1 A better sequential algorithm: min-heap

Before we can truly parallelize top-k, let's first consider a better sequential algorithm. The selection sort approach is slow because it repeatedly scans the entire list. A much more efficient method uses a min-heap data structure of size K.

The algorithm works as follows: **1** Initialization: Build a min-heap with the first K elements of the input vector. The

smallest of these K elements will be at the root of the heap. **2** Iteration: Iterate through the remaining N-K elements of the input vector. For each new element, compare it to the root of the heap (the smallest element currently in our top-k set). **3** Replacement: If the new element is larger than the root, it belongs in the top-k set. Replace the root with the new element, and then perform a "heapify-down" operation to restore the min-heap property.

-----

###### Top-k 163

**4** Final result: After iterating through all N elements, the heap contains the K largest

values. This algorithm has a much better time complexity of O(N × log(K)), a significant improvement over O(N × K). We can implement it within a single thread block, using shared memory to store the heap. Although this is a better algorithm, implementing it naively in CUDA still leaves performance on the table. In the following kernel, we use the threads of a block to load the initial K elements into a shared memory heap in parallel, but then a single thread (thread 0) does all the work of iterating through the remaining elements and managing the heap. You can find this implementation in book.cu/4\_optim/topK/kernels/1\_ heap.cu. As the following listing shows, the kernel begins by having the first K threads load the first K elements from the input vector into a shared memory array that will serve as our heap.

**Listing 6.24 Min-heap top-k: Parallel load**

`extern __shared__ ValueIndex heap[];` **extern \_\_shared\_\_ declares a dynamically**

**sized shared memory array for the heap.**

```
if (threadIdx.x < K) {
heap[threadIdx.x] = {input[threadIdx.x], threadIdx.x}; The first K
```

`}` **threads perform**

`__syncthreads();` **A barrier ensures the load is complete a parallel load of**

**before the heap construction begins. the initial data.**

Next, a single thread (thread 0) builds the min-heap from the K elements now in shared memory. This is a sequential bottleneck. The same single thread then iterates through the rest of the input array, pushing new elements onto the heap if they are larger than the current minimum:

```
if (threadIdx.x == 0) {
```

`for (int i = K; i < N; i++) {` **This is the main O((N - K) × log(K))**

`if (input[i] > heap[0].value) {` **loop, executed entirely by thread 0.**

```
heap[0] = {input[i], i};
heapify_down(heap, 0, K);
}
}
```

| } | A final barrier ensures all heap updates |
|---|---|
| __syncthreads(); | are complete before writing the results. |

Finally, the first K threads write the results from the shared memory heap back to global memory:

```
if (threadIdx.x < K) {
values[threadIdx.x] = heap[threadIdx.x].value;
indices[threadIdx.x] = heap[threadIdx.x].index;
}
```

###### The K threads that loaded the initial data now perform the final parallel write.

-----

###### 164 Chapter 6 Optimizing our transformer kernels

This version is an improvement, but the single-threaded heap management is a major bottleneck. A truly high-performance top-k kernel requires a parallel heap or another parallel selection algorithm, such as bitonic sort, that can use the full parallelism of the thread block.

### 6.5.2 Parallel top-k with block reduction

Instead of having a single thread manage a data structure, we can parallelize the search for the maximum element at each step of the selection sort. This approach is implemented in book.cu/4\_optim/topK/kernels/2\_warp.cu. The algorithm is a parallel version of the naive selection sort. In each of the K iterations, all threads in the block cooperate to find the single largest value among the currently unselected elements. This uses the same hierarchical block-level reduction pattern we've used before, but this time, the reduction needs to find both the max value and the index of that value. First, we need a reduction helper that can operate on a custom struct ValueIndex that holds both a value and its original index:

```
__device__ __forceinline__ ValueIndex warp_reduce_max_with_idx(
ValueIndex val) { /* ... */ }
__device__ __forceinline__ ValueIndex block_reduce_max_with_idx(
ValueIndex val, ValueIndex* smem) {
}
```

###### This helper function is structured identically to blockReduceSum but uses fmaxf and custom logic inside warp_reduce_max_with_idx

**to correctly track the index of the maximum value.**

Listing 6.25 shows the main kernel loop, which runs K times. In each iteration, every thread scans a portion of the input array to find its local maximum. Then a block-wide reduction finds the global maximum for that iteration.

**Listing 6.25 Parallel top-k main loop**

| for (int k = 0; k < K; k++) { | The outer loop still |  |
|---|---|---|
| ValueIndex local = {-INFINITY, -1}; | runs K times. | Each thread finds the maximum value and index |
| for (int i = tid; i < N; i += blockDim.x) { if (!selected[i] && input[i] > local.value) { local = {input[i], i}; |  | in its assigned slice of the input, ignoring already selected elements. |

```
}
```

`}` **block\_reduce\_max\_with\_idx finds the single**

**maximum among all threads in the block.**

```
ValueIndex global_max = block_reduce_max_with_idx(local, reduce_smem);
```

| if (tid | == 0) { | Thread 0 records the result and |
|---|---|---|
| selected[global_max.index] | = true; | marks the found index as |
| values[k] | = global_max.value; | selected for the next iteration. |

```
indices[k] = global_max.index;
```

-----

###### Top-k 165

```
}
```

`__syncthreads();` **A barrier is crucial to ensure all threads see the** `}` **updated selected array before starting the next search.** This approach fully parallelizes the search at each step, making it much more efficient than the sequential heap for many input sizes.

### 6.5.3 Top-k optimization summary

Top-k presents unique challenges compared to the other kernels we've optimized. Unlike GEMV, softmax, and LayerNorm where we perform the same operation on every element, top-k requires selective processing based on element values. The naive selection sort achieves O(K × N) complexity with no parallelism and executes entirely sequentially. The min-heap improves algorithmic complexity to O(N × log(K)) but remains single threaded, using only one thread to manage the heap structure. The parallel version parallelizes the search at each iteration using block-wide reductions but still requires K sequential iterations. The sequential dependency in finding "the next maximum" after marking previous selections creates a fundamental bottleneck that limits how much parallel hardware can improve performance. The K outer iterations cannot be parallelized because each iteration depends on the results of all previous iterations. Production top-k implementations often use radix sort or other fully parallel algorithms when K is large, or bitonic sort networks, which can find all K elements simultaneously without sequential dependencies. For the small K values typical in MoE models (K = 2 to K = 8 experts), the parallel reduction approach we've shown provides reasonable performance.

### 6.5.4 Top-k profiling analysis

The following listing shows the profiling command we run for the top-k kernels at

```
N = 4096 and K = 128.
```

**Listing 6.26 Nsight Compute profiling: Naive vs. parallel top-k**

```
cd book.cu/4_optim/topK
ncu --set full \
--section SpeedOfLight_HierarchicalDoubleRooflineChart \
--section MemoryWorkloadAnalysis \
--section Occupancy \
--section SchedulerStats \
--print-summary per-kernel \
./topK_test
```

Those commands generate the `SpeedOfLight,` occupancy, and scheduler excerpts shown later, allowing us to contrast the naive and parallel top-k strategies on identical inputs:

-----

###### 166 Chapter 6 Optimizing our transformer kernels

```
--- Naive TopK (selection sort, 1 thread) ---
```

| Duration | 20.4 ms |
|---|---|
| DRAM Throughput | 0.00 % |
| L1/TEX Cache Throughput | 5.7 % |
| Compute (SM) Throughput | 0.07 % |

| Achieved Occupancy --- Parallel TopK (warp | 2.1 % per block) --- | (1 warp/SM active) |
|---|---|---|
| Duration DRAM Throughput L1/TEX Cache Throughput | 231 us 0.01 % 23.1 % | ~86x faster |
| Compute (SM) Throughput Achieved Occupancy | 0.28 % 16.7 % |  |

The key results are as follows:

¡ *Parallel is about 86× faster-The warp kernel finishes in 231 μs versus 20.4 ms for*

the single-thread selection sort. Replacing one thread that scans the whole array

```
K times with a warp-parallel reduction delivers the gain.
```

¡ *Still extremely low utilization-Even the optimized kernel shows near-zero DRAM*

(0.01%) and compute (0.28% throughput. Top-k here is bound by the data dependencies in the K-loop, not by bandwidth or math: the hardware spends most of its time waiting for the previous pick, not working. ¡ *Occupancy improvement helps, but only so much-Moving from 2% to 17% occupancy*

hides some latency, yet both kernels leave most of the GPU idle. ¡ *Sequential bottleneck dominates-The outer K-loop creates serialization* that no

amount of micro-optimization can eliminate.

## 6.6 General matrix-matrix multiplication

We've arrived at the final and most computationally intensive kernel of the chapter: GEMM, or general matrix-matrix multiplication. GEMM is the workhorse of deep learning, forming the backbone of linear layers, convolutions, and many other operations. For transformers, it's used to process entire batches of token embeddings simultaneously. Although GEMV is crucial for single-token generation, GEMM is what gives transformers their high throughput when processing prompts or training in batches.

| Why we switch to FP16 for GEMM |
|---|
| Every kernel so far (GEMV, softmax, LayerNorm, top-k) has used FP32 (32-bit float). |
| For GEMM, we switch to FP16: a 16-bit floating-point format with 5 exponent bits and 10 mantissa bits. FP16 stores roughly 3 to 4 decimal digits of precision, compared to 7 to 8 for FP32. |
| There are two reasons. First, memory bandwidth: FP16 halves the bytes per element, so every load and store moves twice as much data in the same bus cycles. For a |

-----

###### General matrix-matrix multiplication 167

compute-bound kernel like GEMM, feeding the ALUs faster matters more than extra precision. Second, hardware throughput: tensor cores (chapter 7) provide roughly 2× higher TFLOPS for FP16 than for FP32 on GeForce parts, and 15× or more on data center parts like the A100 and H100, so our GEMM kernels are directly preparing for the tensor core chapter ahead. The precision loss is acceptable for most deep learning workloads. Weights and activations in transformer inference are routinely stored in FP16 or BF16 with little accuracy loss for most workloads (the two formats are not interchangeable: BF16 trades mantissa bits for exponent range). Training typically uses mixed precision: FP16 for forward and backward passes, and FP32 for the master weight copy and gradient accumulation. In the following code, the fp16 type is a typedef for CUDA's \_\_half type.

Our goal is to compute C = A @ B, where A is an M x K matrix, B is a K x N matrix, and C is the resulting M x N matrix.

###### GEMM: A compute-bound operation

Unlike GEMV, softmax, and LayerNorm (which are memory-bound), GEMM is fundamentally compute-bound. This dramatically changes our optimization strategy. Consider the following computational intensity analysis. When multiplying an M × K matrix by a K × N matrix in FP16, we need to read MK + KN elements and write MN elements. The computation involves 2MNK FLOPs: MN dot products of length K. This gives an arithmetic intensity of 2MNK / [2(MK + KN + MN)] FLOPs per byte. For a concrete example, consider multiplying two 4096 × 4096 matrices. This requires approximately 67 MB of reads and 34 MB of writes, totaling 101 MB of minimum memory traffic. The computation performs 2 × 4,096³ operations, or roughly 137 billion FLOPs, yielding an arithmetic intensity of approximately 1,365 FLOPs per byte. Compare this with GEMV's approximately 0.5 FLOPs per byte. The massive data reuse in GEMM (each matrix element participates in thousands of multiplications) means we can amortize memory access costs across many computations. The optimization goal for GEMM is to maximize compute utilization (target 80% to 90% of peak TFLOPS), not just memory bandwidth. We achieve this through coalesced memory access as a baseline requirement, followed by shared memory tiling to enable data reuse, register blocking to maximize arithmetic intensity, and finally, vectorization to saturate memory bandwidth when needed.

The most straightforward parallelization strategy, which mirrors our very first naive kernels, is to assign one thread to compute a single element of the output matrix C. Each thread calculates one dot product between a row of A and a column of B. You can find this implementation in book.cu/4\_optim/gemm/kernels/1\_naive.cuh. The

-----

###### 168 Chapter 6 Optimizing our transformer kernels

GEMM kernels in this section are based in part on Simon Boehm's SGEMM tutorial (MIT license, [https://github.com/siboehm/SGEMM\_CUDA). The following listing](https://github.com/siboehm/SGEMM_CUDA) shows the naive kernel.

**Listing 6.27 Naive GEMM kernel Each thread initializes an accumulator. This kernel Uses a 2D grid of threads, where the uses fp16 (half-precision) arithmetic, common for global x and y indices map directly to the accelerating deep learning workloads. rows and columns of the output matrix C**

```
const uint x = blockIdx.x * blockDim.x + threadIdx.x;
const uint y = blockIdx.y * blockDim.y + threadIdx.y;
```

`if (x < M && y < N) {` **The thread iterates along the K**

`fp16 tmp = __float2half(0.0f);` **dimension to compute the dot product**

**for its assigned output element C[x, y].**

```
for (int i = 0; i < K; ++i) {
tmp = __hadd(tmp, __hmul(A[x * K + i], B[i * N + y]));
}
```

`C[x * N + y] = tmp;` **For threads with consecutive x indices, the** `}` **A[x \* K + i] reads are K elements apart (uncoalesced),**

**Writes the final result and B[i \* N + y] is the same address for all of them. to global memory**

This kernel is a good starting point, but it suffers from the same memory access problems we've seen before, only magnified. A warp's 32 threads have consecutive x and the same y. Their reads of A are K elements apart, so they are uncoalesced, and they all read the same element of B.

Profiling the naive GEMM kernel with NCU shows abysmal memory throughput despite the kernel being compute-bound in theory. The problem is memory access patterns: each thread reads elements from a different row of A and a different column of

```
B, creating completely uncoalesced accesses. The memory subsystem is bottlenecked by
```

scattered requests instead of serving efficient burst transfers.

### 6.6.1 Understanding the GEMM memory access problem

Let's analyze the naive kernel's memory access pattern in detail. Consider a 2D block where blockDim.x = blockDim.y = 32, creating 1,024 threads arranged in a 32 × 32 grid.

The access pattern for matrix A is

```
Thread (0,0): A[0 * K + 0], A[0 * K + 1], ..., A[0 * K + K-1]
Thread (1,0): A[1 * K + 0], A[1 * K + 1], ..., A[1 * K + K-1]
Thread (2,0): A[2 * K + 0], A[2 * K + 1], ..., A[2 * K + K-1]
```

Within a warp (threads with consecutive x indices), these accesses are K elements apart, so they are uncoalesced."

-----

###### General matrix-matrix multiplication 169

The access pattern for matrix B is

```
Thread (0,0): B[0 * N + 0], B[1 * N + 0], ..., B[K-1 * N + 0]
Thread (0,1): B[0 * N + 1], B[1 * N + 1], ..., B[K-1 * N + 1]
Thread (0,2): B[0 * N + 2], B[1 * N + 2], ..., B[K-1 * N + 2]
```

Threads with consecutive x values share y, so a warp reads a single element of B per iteration. The uncoalesced traffic comes from A. Performance effect is the following: the naive kernel achieves well under 1% of peak TFLOPS (about 0.3 TFLOPS), moving only around 10 GB/s of useful data out of the 936 GB/s available. Each thread performs K \* 2 operations (K FMAs) but reads 2K values from global memory with terrible locality. Profiling shows that threads spend over 90% of their time stalled, waiting for memory. The solution requires fundamental restructuring through shared memory tiling.

### 6.6.2 Improving coalescing through thread remapping

Before implementing full tiling, we can improve memory access by changing how threads map to output elements. Instead of using 2D threadIdx, we flatten the thread index and recalculate positions. In book.cu/4\_optim/gemm/kernels/2\_gmem\_coalesce.cuh, the thread-to-output mapping becomes

```
const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
```

This helps because, with 1D threadIdx.x running from 0 to 1023 in a block, threads 0 to 31 (the first warp) now have cCol values 0 to 31, meaning they're all working on the same row. These threads access A[cRow \* K + i] with identical cRow values, giving good locality, and they access B[i \* N + cCol] with consecutive cCol values, achieving coalesced reads.

Performance effect is the following: memory throughput climbs sharply once accesses are coalesced, and compute throughput reaches approximately 2.5 TFLOPS, giving us about an 8× speedup over the naive implementation. Still, we're memorybound because each matrix element is read K times from global memory. We need caching.

### 6.6.3 GEMM optimization with tiling and shared memory

The core idea behind tiling is to break the large matrix multiplication into a series of smaller matrix multiplications that a single thread block can handle efficiently. Each thread block calculates a small tile of the output matrix C. To do this, the block first loads a corresponding tile from A and a tile from B into shared memory. It then performs a matrix multiplication using the data in shared memory. Because shared memory is so fast, the threads can read from it repeatedly without

-----

###### 170 Chapter 6 Optimizing our transformer kernels

a significant performance penalty. This process is repeated, moving across the K dimension, loading new tiles of A and B, and accumulating the results.

Figure 6.4 zooms out to compare the naive and tiled schedules. At the top, a single

row/column dot product is computed and written to C before the kernel moves to the next output element. Below it, the tiled kernel cycles through shaded rectangles representing shared-memory tiles, accumulating partial results over several time slices before writing a finished tile back to C. It's intentionally math-free:just a timeline of rectangles

that highlights how tiling reuses data instead of reloading it each time.

| Understanding tiling: The key to GEMM performance |
|---|
| Tiling (also called blocking) transforms GEMM from memory-bound to compute-bound |
| by exploiting data reuse. |
|  |
| The problem without tiling: Each element of A is used N times (once for each column of B), and each element of B is used M times (once for each row of A). Without caching, this means M*N reads from both matrices! |
| The tiling solution: Break the computation into "tiles" small enough to fit in shared memory: |
| For each block computing tile of C[bm:bm+BM, bn:bn+BN]: |
| For each tile along K dimension (bk = 0, BK, 2BK, ...): |
| 1. Load A[bm:bm+BM, bk:bk+BK] into shared memory → BM*BK floats |
| 2. Load B[bk:bk+BK, bn:bn+BN] into shared memory → BK*BN floats |
| 3. Compute partial matrix multiply using shared memory data |
| 4. Accumulate results in registers |
| Data resuse analysis: Without tiling in the naive approach, each thread reads K elements from A and K elements from B, resulting in approximately 2,048 K global memory reads per block of 1024 threads. With tiling using BM = BN = 128 and BK = 8, each tile iteration has the block load BM*BK + BK*BN = 128*8 + 8*128 = 2,048 elements. These 2,048 elements are then reused for BM*BN = 16,384 dot product contributions, giving us a reuse factor of 16,384 / 2,048 = 8x per loaded element. This 8x reuse per tile, across K/8 tile iterations, dramatically reduces global memory pressure. |
| Why shared memory? Shared memory has roughly 19x higher aggregate bandwidth than global memory on the RTX 3090 (~18 TB/s across all SMs vs. 936 GB/s) and more than 10x lower latency (~30 cycles vs. hundreds). By loading tiles once and reusing them many times from shared memory, we shift from memory-bound to compute-bound. |

This strategy solves the memory coalescing problem for both matrices. Consecutive threads load consecutive elements of a tile row from both A and B, so both loads are coalesced. The tiling logic is in book.cu/4\_optim/gemm/kernels/3\_smem\_blocking

.cuh.

-----

###### General matrix-matrix multiplication 171

###### NAIVE MATMUL

2nd row

4 @ 8 =

3rd row 8

i 4

---

###### TILED MATMUL

4

A tile @ B tile = C tile

8 I

I I I

8

Lo

4 2x2

---

4 4

**t = 1** 8 **t = 2** 8

4 4

8 8

4

| t = | 3 | 8 |  |  | t = 4 | 8 |  |
|---|---|---|---|---|---|---|---|
| 4 |  |  |  | 4 |  |  |  |

8 8

###### Dot product one ENTIRE row (A)

**with one ENTIRE column (B). Write the**

###### Accumulate partial dot products (element-wise add) until the C tile is complete.

**Figure 6.4 Naive row/column dot products (top), contrasted with the tiled, partial-accumulation schedule (bottom), for matrix multiplication**

-----

###### 172 Chapter 6 Optimizing our transformer kernels

The following listing shows the kernel setup, which begins by declaring the shared memory tiles and calculating the thread's position within the block.

**Listing 6.28 Tiled GEMM: Setup**

```
__shared__ fp16 As[BLOCKSIZE * BLOCKSIZE];
__shared__ fp16 Bs[BLOCKSIZE * BLOCKSIZE];
const uint threadCol = threadIdx.x % BLOCKSIZE;
const uint threadRow = threadIdx.x / BLOCKSIZE;
fp16 tmp = __float2half(0.0f);
```

The main logic is a loop that iterates through the matrix tiles along the K dimension. In each iteration, the threads of the block cooperate to load one tile of A and one tile of B from global memory into their respective shared memory arrays:

**The loop iterates K / BLOCKSIZE times.**

```
for (int bkIdx = 0; bkIdx < K; bkIdx
As[threadRow * BLOCKSIZE + threadCol]
Bs[threadRow * BLOCKSIZE + threadCol]
__syncthreads();
```

Once the tiles are in shared memory, the threads perform a matrix multiplication on the small, on-chip matrices, as shown in the next listing. Each thread calculates a single dot product, accumulating the results in its private register.

**Listing 6.29 Tiled GEMM: Tile multiplication Each thread iterates BLOCKSIZE times, reading from the shared memory tiles to compute its partial dot product.**

```
for (int dotIdx = 0; dotIdx < BLOCKSIZE;
tmp = __hadd(tmp, __hmul(As[threadRow
}
__syncthreads();
}
C[threadRow * N + threadCol] = tmp;
```

###### After the main loop finishes, each thread writes its final accumulated value to the output matrix C.

###### Declares two shared memory arrays to hold one tile from A

**and one from B. BLOCKSIZE is a template parameter, often 32.**

###### Each thread calculates its 2D position within the tile. Initializes a register tmp to accumulate the

**final result for this thread's output element.**

###### Each thread loads one element from A and one from B

**into the shared memory tiles. Both reads are coalesced, because consecutive threads read consecutive columns.**

```
+= BLOCKSIZE) {
= A[threadRow * K + threadCol];
= B[threadRow * N + threadCol];
```

**The \_\_syncthreads() is absolutely critical. It forces all threads to wait until the shared memory tiles are completely populated before any thread can proceed to the computation phase.**

**Access to As is row-wise, and access to Bs is column-wise. Because this is in shared memory, we don't worry about**

**coalescing, but bank conflicts can be a concern.**

```
++dotIdx) {
* BLOCKSIZE + dotIdx],
Bs[dotIdx * BLOCKSIZE + threadCol]));
```

###### Another __syncthreads() is needed to prevent threads from starting the next outer loop iteration (and overwriting shared

**memory) before all threads have finished their computation on the current tiles.**

-----

###### General matrix-matrix multiplication 173

###### Performance of basic tiling

Basic tiling achieves compute throughput of approximately three TFLOPS because loads are reused from shared memory. This gives us about a 10× speedup over the naive implementation and roughly 1.2× over the coalesced version. The improvement is modest because each thread still computes only one output element. Most of the time is spent loading data into shared memory rather than computing. The instruction mix shows mostly memory loads, not FMAs. Looking at the compiled Parallel Thread Execution (PTX) code for the inner loop reveals the problem. PTX is NVIDIA's intermediate assembly language, which chapter 7 covers in more detail. For now, read it simply as the instructions the compiler actually emitted:

| ld.shared.f32 | %f91, [%r8+3456]; | Loads from shared memory (A and B tiles) |
|---|---|---|
| ld.shared.f32 | %f92, [%r7+108]; | Loads from shared memory (A and B tiles) |
| fma.rn.f32 | %f93, %f92, %f91, %f90; | Single FMA operation |

The instruction sequence is two memory loads followed by one FMA. Memory loads have higher latency than FMAs, and profiling shows threads spending most cycles stalled while waiting for shared memory accesses to complete. The "Stall MIO Throttle" metric means warps are waiting because the MIO (memory input/output) instruction queue is full. The MIO pipelines serve special math instructions, dynamic branches, and shared memory instructions; here the pressure comes from issuing too many shared memory instructions relative to compute.

###### The path forward

Currently we have 1,024 threads, each computing 1 output per thread, yielding 1,024 outputs per block. To maximize compute, we can use 1D blocktiling where each thread computes 8 outputs in a column, giving 8× more FMAs per load. Taking it further, 2D blocktiling has each thread compute 8 × 8 = 64 outputs in a tile, yielding 64× more FMAs per load. This progression increases arithmetic intensity, allowing us to approach compute-bound performance.

| Exercise: Shared memory bank conflicts |
|---|
| The basic tiled GEMM kernel accesses shared memory in patterns that may cause |
| bank conflicts. Examine the inner loop in book.cu/4_optim/gemm/kernels/3_smem_ blocking.cuh. Identify which accesses to As and Bs could cause conflicts when mul- tiple threads in the same warp access shared memory simultaneously. Propose a memory layout modification to eliminate these conflicts. |

This tiled approach is the foundation of all modern high-performance GEMM implementations, including NVIDIA's own cuBLAS library. Although cuBLAS uses far more

-----

###### 174 Chapter 6 Optimizing our transformer kernels

advanced techniques, including the specialized tensor cores that we will discuss in the next chapter, this shared memory tiling strategy is the core idea.

### 6.6.4 1D blocktiling: Computing multiple outputs per thread

Each thread still computes only a single output element. We can do better by having each thread compute multiple output elements. This way, we can reuse loaded data across more computations. In book.cu/4\_optim/gemm/kernels/4\_1d\_blocktiling.cuh, each thread computes

```
TM=8 consecutive elements in a column of the output tile.
```

Memory access analysis: Per tile iteration, each thread loads one element from the

```
B tile at B[dotIdx][threadCol], loads TM = 8 elements from the A tile at A[threadRow *
```

`TM + resIdx][dotIdx] for resIdx from` 0 to 7, and performs 8 FMAs (one per output element).

Arithmetic intensity improvement: Basic tiling with TM = 1 uses two loads per iteration (one from A, one from B) to perform 1 FMA, giving a ratio of 0.5 FMAs per load. With 1D block tiling, where TM = 8, we use nine loads per iteration (eight from `A,` one from B) to perform eight FMAs, giving a ratio of 0.89 FMAs per load. More importantly, the B element is reused across eight computations! The following listing shows this pattern.

**Listing 6.30 1D block-tiling computation**

###### Each thread now has a register array threadResults to hold the TM

`fp16 threadResults[TM];` **output values it's responsible for.**

###### For each step of the inner dot product, the thread loads one

```python
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) { value from the B tile.
fp16 tmpB = Bs[dotIdx * BN + threadCol];
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
threadResults[resIdx] = __hadd(threadResults[resIdx],
__hmul(As[(threadRow * TM + resIdx) * BK + dotIdx], tmpB));
}
```

`}` **It then iterates through its TM results, multiplying the value from B**

**with TM different values from the A tile and accumulating the results.**

Performance effect is the following: The 1D blocktiling approach achieves compute throughput of approximately 9.3 TFLOPS (26% of peak), delivering a 3.1× speedup over basic tiling and roughly 31× over the naive implementation. Warp stall time due to memory drops substantially.

###### Compiler optimizations

An interesting detail: even if we write the code with nested loops in a less optimal order, the compiler is smart enough to optimize it. Consider this less efficient version:

```
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
```

-----

###### General matrix-matrix multiplication 175

threadResults[resIdx] +=

As[(threadRow \* TM + resIdx) \* BK + dotIdx] \* Bs[dotIdx \* BN + threadColl;

This would incur shared memory accesses instead of 72. But examin-

BK \* TH \* 2 128

=

ing the generated PTX shows that the compiler unrolls both loops and eliminates the redundant loads from

Bs:

| Lld.shared.f32 | %f45, | [%r9]; | [%r9]; | DE | First loop loads from the B tile |
|---|---|---|---|---|---|
| 1d.shared.f32 %f46, [%r8]; (values stored in %f45, %f48, etc.) |  |  |  |  |  |
| fma.rn.f32 |  |  |  |  |  |
| 1d.shared.f32 | %f48, | [%r9+256]; |  |  |  |
| 1ld.shared.f32 | %f49, | [%r8+4]; |  |  |  |
| fma.rn. £32 | %FS50, | %F49, | %F48, | %F4T; |  |
| 1d.shared.f32 | %fé68, | [%r8+32]; |  |  | Second loop reuses earlier loaded values |
| fma.rn. £32 | %F69, | %F68, | %F4S, | %F211; | (%f45, %f48) instead of reloading |
| 1d.shared.f32 | %f70, | [%r8+36]; |  |  | Second loop reuses earlier loaded values |
| fma.rn. £32 | %F71, |  |  | %f69; | (%f45, %f48) instead of reloading |

The compiler recognizes the data dependencies, unrolls the loops (because loop counts are compile-time constants), and reuses the values that were loaded in the

Bs

first iteration. When this PTX is compiled to SASS (hardware instructions), the loads from are further vectorized into 128-bit operations:

Bs

###### 32-bit loads from As (scalar)

128-bit vectorized loads from 32-bit loads from As (scalar)

| LDS | R26, | [R35.X4+0x800] | -— |
|---|---|---|---|
| LDS.128 | RS, | [R2] | #2 |
| 1DS.128 | R12, | [R2+6x20] | Bs (four floats at once) |
| LDS | R24, | [R35.X4+0x960] | 128-bit vectorized loads from |
| LDS.128 | R20, | [R2+0x60] | -— Bs (four floats at once) |

This is substantial progress, but we can do better by making threads compute 2D tiles.

### 6.6.5 2D blocktiling: Maximum register reuse

We can take the tiling concept one step further. Instead of having each thread compute

a TM 1 vector of results, we can have each thread compute a small 2D tile of TH

x x TN

results. This is the strategy used in book.cu/4\_optim/gemm/kernels/5\_2d\_blocktiling .cuh.

###### The hierarchy of tiles

To understand 2D block tiling, we need to visualize the data hierarchy. We subdivide each block tile into "micro-tiles," where each micro-tile is managed by a single thread:

-----

###### 176 Chapter 6 Optimizing our transformer kernels

```
Global Matrix C (M x N)
+---------------------------+
```

| \| |  |  |  | \| |
|---|---|---|---|---|
| \| | Block Tile (BM x BN) |  |  | \| |
| \| | +-------------------+ |  |  | \| |
| \| | \| | Thread(0,0) | \| | \| |
| \| | \| | Micro-Tile | \| | \| |
| \| | \| | (TM x TN) | \| | \| |
| \| | +-------------------+ |  |  | \| |
| \| |  |  |  | \| |
| +---------------------------+ |  |  |  |  |

If TM=8 and TN=8, each thread maintains an 8 × 8 grid of accumulators in its registers. This means a single thread computes 64 distinct elements of the output matrix C.

###### Implementation details

In book.cu/4\_optim/gemm/kernels/5\_2d\_blocktiling.cuh, we define the block-tile and micro-tile hierarchy from that diagram in code. The logic can be dense, so let's unpack the indexing carefully.

| Decoding the tile indices |
|---|
| The index calculations in 2D blocktiling often trip up developers. Let's break down the |
| access pattern As[(threadRow * TM + i) * BK + dotIdx]: |
| ¡ threadRow-The vertical coordinate of the thread within the thread block (e.g., |
| 0 to 15). |
| ¡ TM-The height of the micro-tile each thread handles (for example, 8 rows). |
| ¡ threadRow * TM-The global row index (within the shared memory tile) where |
| this thread's work begins. |
| ¡ + i-The offset for the specific row within the thread's micro-tile (0 to 7). |
| ¡ * BK-Stride to the start of that row (since As is row-major with width BK). |
| + dotIdx-The current column we load from in the K-loop. |
| ¡ |
| Essentially, (threadRow * TM + i) selects the logical row of the tile, and dotIdx |
| selects the column. |
| The following listing presents the inner loop strategy with annotated comments. |
| Listing 6.31 2D block tiling computation |

###### Register cache for B-tile values (TN elements) 64 accumulators per thread

```
fp16 threadResults[TM * TN]; (for an 8 × 8 micro-tile)
fp16 regM[TM];
```

`fp16 regN[TN];` **Register cache for A-tile values (TM elements)**

```
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) { Row-major access:
for (uint i = 0; i < TM; ++i) { logical row is
regM[i] = As[(threadRow * TM + i) * BK + dotIdx]; threadRow * TM + i.
}
```

-----

###### General matrix-matrix multiplication

```
for (uint i = 0; i < TN; ++i) {
regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
}
for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
int flatIdx = resIdxM * TN + resIdxN;
threadResults[flatIdx] = __hadd(
threadResults[flatIdx],
__hmul(regM[resIdxM], regN[resIdxN])
);
}
}
}
```

###### Column-major-like access in B-tile logic

###### Outer product update: reuses each loaded regM and regN value multiple times

The outer product structure means each loaded value is reused many times. Think of it as computing a small grid of values:

```
regN[0] regN[1] ... regN[7]
```

| regM[0] | . | . | ... | . |
|---|---|---|---|---|
| regM[1] | . | . | ... | . |
| ... | . | . | ... | . |
| regM[7] | . | . | ... | . |

Each regM[i] value is multiplied by TN = 8 different regN[j] values (a whole row of the grid), and each regN[j] value is multiplied by TM = 8 different regM[i] values (a whole column). In total, TM + TN=16 loads enable TM × TN = 64 FMAs. Arithmetic intensity: With 2D blocktiling where TM = TN = 8, we perform 16 loads per dotIdx iteration (8 from A, 8 from B) to execute 64 FMAs, giving a ratio of 4 FMAs per load: 8× better than basic tiling! Memory access analysis per thread: Per output tile of C across all K/BK iterations, we have approximately 2 \* (K/BK) \* (BM\*BK + BK\*BN) / numThreads ≈ K/(32\*threads) GMEM loads per output. For SMEM loads, we have (K/BK) \* (TM + TN) \* BK = K \* (TM + TN) per thread in total. The thread performs K \* TM \* TN FMAs in total, giving an arithmetic intensity (FMAs per SMEM load) of KxTMxTN / (Kx(TM+TN)) = TMxTN/(TM+TN) = 4. Performance effect is the following: The 2D blocktiling achieves compute throughput of approximately 25.1 TFLOPS (70% of peak), delivering a 2.7× speedup over 1D tiling and roughly 84× over naive. The GPU now spends most of its time computing rather than waiting for memory, with the instruction mix showing roughly 60% FMA operations (compared to less than 10% in the naive kernel). The 2D blocktiling represents the inflection point where GEMM becomes truly compute-bound.

### 6.6.6 Vectorizing memory access for tiling

The final optimization we will apply is vectorization, which we can combine with our 2D tiling strategy. CUDA's int2 and int4 vector types move 8 and 16 bytes, respectively.

-----

###### 178 Chapter 6 Optimizing our transformer kernels

The suffix gives the number of 32-bit lanes: int2 is two 32-bit integers, or 64 bits, not a 2-bit integer. This kernel uses int2 only as a carrier for the bit patterns of four fp16 values. This is implemented in book.cu/4\_optim/gemm/kernels/6\_vectorize.cuh. The following listing shows the vectorized loading pattern.

**Listing 6.32 Vectorized tile loading**

```
if (canVectorize) {
*reinterpret_cast<VecType *>(smemDst) = If the memory is properly
*reinterpret_cast<const VecType *>(gmemSrc); aligned, the thread loads a wide
```

`} else {` **vector in a single instruction.**

```
for (int v = 0; v < VecElems; ++v) {
```

`smemDst[v] = gmemSrc[v];` **If not, it falls back to a scalar loop to handle** `}` **the remaining elements one by one.**

```
}
```

###### Why reinterpret_cast matters

This FP16 kernel uses an aligned vector type to move four values at once. Each participating thread copies four fp16 values, or 8 bytes. VecType aliases int2, an aligned 8-byte carrier for those bits; it does not convert the values to integers or change the arithmetic type. The carrier depends on the element type and the desired transfer size. Four FP32 values fit in a 16-byte float4 or int4; eight FP8 values fit in an 8-byte int2, and 16 FP8 values fit in a 16-byte int4. Production kernels use the same pattern. For example, vLLM uses uint2 to carry four 16-bit lanes, while PyTorch kernels define alignment predicates before issuing wide loads..

```
canVectorize is a Boolean variable defined by this kernel, not a CUDA built-in. It
```

combines the tile-boundary checks with an 8-byte alignment check for both pointers. When the predicate is true, the reinterpret\_cast assignment lets the compiler generate one 64-bit global load and one 64-bit shared memory store per participating thread. When the predicate is false, the scalar path handles the remaining values without reading past the tile boundary. Here, instruction width means the number of bits moved by one load or store that a thread issues. Changing from a scalar fp16 load to an int2 load widens that memory instruction from 16 to 64 bits; it does not change the warp width or the FP16 arithmetic. These per-thread instructions remain coalesced when adjacent threads access consecutive addresses.

###### Why vectorization still matters for compute-bound kernels

Even though we're compute-bound, vectorized global memory access still helps. Each aligned int2 load moves four FP16 values (8 bytes) with one instruction instead of four scalar loads, reducing instruction count and freeing issue slots for FMAs. The alignment check ensures the vector path is used only when both source and destination are 8-byte aligned; otherwise, the scalar fallback preserves correctness.

-----

###### General matrix-matrix multiplication 179

Performance effect is the following: Vectorization pushes compute throughput to approximately 27.8 TFLOPS (78% of peak), yielding a 1.1× speedup over 2D tiling and roughly 93× over naive. Against the RTX 3090's CUDA-core FP16 peak of about 35.6 TFLOPS, that leaves roughly 20% on the table. cuBLAS still reaches 66 TFLOPS on the same problem, but only by using tensor cores, the hardware chapter 7 introduces.

###### Why we don't reach 100% of peak

Several factors prevent us from reaching peak performance. Register pressure limits occupancy, and some shared memory bank conflicts remain. cuBLAS uses tensor cores (about 2× the FP32 CUDA-core rate on the RTX 3090, roughly 15× on an A100 or H100) has autotuned parameters for each matrix size, and implements double buffering with software pipelining.

### 6.6.7 GEMM profiling: Understanding the bottleneck

Profiling the GEMM variants on an RTX 3090 (M = N = K = 4096, FP16) shows the transition from memory-bound to compute-bound performance:

```
cd book.cu/4_optim/gemm
ncu --set full \
--section SpeedOfLight_HierarchicalDoubleRooflineChart \
--section MemoryWorkloadAnalysis \
--section Occupancy \
--print-summary per-kernel \
./gemm_test
```

Listing 6.33 pulls out the most important metrics from that capture. GEMM is computebound, so the number to watch is Compute (SM) throughput and the resulting TFLOPS, not DRAM bandwidth. We show the naive kernel and the fully optimized vectorized kernel side by side.

**Listing 6.33 Nsight Compute profiling: GEMM naive vs. optimized**

| --- Naive GEMM (M=N=K=4096, | fp16) | --- |
|---|---|---|
| Duration Compute (SM) Throughput | 486 12.0 % | ms |
| DRAM Throughput |  | 1.0 % |
| L1/TEX Cache Throughput Achieved Occupancy | 99.0 % 65.4 % | Cache pipe maxed, no reuse |

```
(0.3 TFLOPS)
```

| --- Vectorized GEMM | (register tiling + float4) --- |  |
|---|---|---|
| Duration Compute (SM) Throughput DRAM Throughput | 5.26 ms 76.1 % 9.5 % | ~90x faster |
| L1/TEX Cache Throughput Achieved Occupancy | 81.8 % 59.6 % |  |

```
(26 TFLOPS; cuBLAS with tensor cores: 66 TFLOPS)
```

-----

###### 180 Chapter 6 Optimizing our transformer kernels

Consider this critical analysis: ¡ *Coalescing matters at realistic sizes.* At 4096 × 4096 the matrices no longer fit in cache, so the naive kernel (0.3 TFLOPS) wastes most of its DRAM traffic on uncoalesced accesses. Simply fixing the access pattern lifts it to 2.5 TFLOPS, an 8× jump, before any tiling. ¡ *Shared memory tiling is a modest step (2.5 → 3.0 TFLOPS). Basic tiling adds reuse, but* each thread still computes only one output, so the math units stay mostly idle. The big gains come from computing more outputs per thread. ¡ *Block tiling is the real game-changer (3.0 → 25.1 TFLOPS). 1D tiling (8 outputs/* thread) reaches 9.3 TFLOPS, and 2D tiling (64 outputs/thread) reaches 25.1 TFLOPS, an 8× jump over shared memory tiling. This is the transition to compute bound: the vectorized kernel runs at 76% compute throughput and just 1% to 10% DRAM. ¡ *The full ladder needs realistic sizes. From naive to vectorized is about 90× (0.3 → 27.8* TFLOPS) at 4096 × 4096. At tiny matrices, the whole problem fits in cache and hides these differences, so always profile at sizes you actually care about.

### 6.6.8 GEMM optimization summary

Our GEMM optimization journey shows the progression from memory-bound to compute-bound. Regarding performance progression (M = N = K = 4096, FP16, RTX 3090), the naive implementation achieves around 0.3 TFLOPS with uncoalesced access and no data reuse. Coalescing pushes us to roughly 2.5 TFLOPS. Shared memory tiling adds basic reuse, reaching 3 TFLOPS. 1D blocktiling increases compute per thread by 8× and lands at 9.3 TFLOPS. 2D blocktiling delivers 64× compute per thread and marks the transition to compute-bound performance at 25 TFLOPS, about 70% of the card's FP16 CUDA-core peak. Vectorization pushes to about 27 TFLOPS (77%). cuBLAS reaches 66 TFLOPS by using tensor cores, the hardware the chapter 7 material targets. The progression falls into three phases: fixing memory access, staging data in shared memory, and raising the work each thread does. Phase one is dominated by memory bottlenecks; remapping threads for coalescing brings the first ~8× speedup but leaves the kernel bandwidth-limited. Phase two introduces shared memory tiles so each load serves multiple FMAs, yet the math units still idle between transactions. Phase three adopts block tiling so every thread computes a TM × TN tile in registers, lifting arithmetic intensity high enough to keep the SMs busy. Moving from a TM × 1 tile to TM × TN gives every element of A and B many more reuse opportunities. Each B load feeds TM outputs, each A load feeds TN outputs, so TM + TN loads feed TM × TN FMAs (16 loads and 64 FMAs at TM = TN = 8). Table 6.1 compares arithmetic intensity and the strategy that pays off across the four kernels.

-----

###### Summary 181

**Table 6.1 Arithmetic intensity and the optimization strategy that pays off for each of the four kernels in this chapter**

| Kernel | Arithmetic Intensity | Optimal Strategy |
|---|---|---|
| GEMV Softmax LayerNorm GEMM | ~0.5 FLOPs/byte ~1 FLOPs/byte ~1 FLOPs/byte ~1,365 FLOPs/byte | Coalescing + vectorization Shared memory fusion Warp shuffles Register blocking + tiling |

---

GEMM's massive arithmetic intensity makes it uniquely amenable to compute optimization. The other kernels remain memory-bound no matter how we optimize.

Treat memory-bound kernels with coalescing, fusion, and vectorization. Treat compute-bound kernels with higher arithmetic intensity, smarter register reuse, and enough occupancy to hide latency. Vendor libraries build on the same foundation with autotuning, tensor cores, and deeper pipelining.

You can benchmark all kernels with python book.cu/4\_optim/gemm/main.py.

###### Summary

¡ Nsight Compute profiling reveals occupancy, bandwidth, and stall metrics that

identify kernel bottlenecks and quantify optimization gains of10 to 100×. ¡ Memory-bound kernels (GEMV, softmax, LayerNorm) follow a consistent optimization ladder: fix coalescing, fuse passes in shared memory, apply warp and block reductions, and then vectorize loads. ¡ GEMV optimization progresses from naive thread-per-row (10% to 20% bandwidth) through coalesced block-per-row (40% to 60%) to vectorized loads (80% to 95%), matching cuBLAS performance. ¡ Warp-level reductions using \_\_shfl\_down\_sync operate entirely in registers,

avoiding shared memory overhead and executing in logarithmic time. ¡ Block-level reductions combine warp shuffles with shared memory to scale

beyond 32 threads, enabling larger thread blocks that better hide memory latency. ¡ Softmax and LayerNorm share the same block-per-row fusion pattern: load once

to shared memory, perform parallel reductions for statistics, normalize in a single kernel. ¡ Online softmax fuses the max and sum passes using exponential rescaling, reducing global memory reads from 3N to 2N per row. This technique reappears in Flash Attention. ¡ Top-k demonstrates that not all algorithms parallelize equally: the sequential

dependency in finding successive maxima creates a fundamental bottleneck that limits GPU utilization.

-----

###### 182 Chapter 6 Optimizing our transformer kernels

¡ GEMM transitions from memory-bound to compute-bound through shared

memory tiling (data reuse), 1D block tiling (8× FMAs per load), and 2D block tiling (64× FMAs per load), reaching 70% to 78% of peak TFLOPS. ¡ Algorithmic improvements (online softmax, heap-based top-k, register tiling

GEMM) often deliver larger gains than micro-optimizations, showing that math changes and kernel tuning work in tandem. ¡ The optimization patterns established here carry over to other memory-bound

and compute-bound GPU workloads.

-----

# 7

## Tensor cores

---

| This chapter covers |
|---|
| ¡ Tensor core architecture and when to use tensor cores versus CUDA cores |
| ¡ WMMA fragments, tiling, and the high-level |
| tensor core API (Volta and newer) |
| ¡ WGMMA asynchronous operations with inline |
| PTX on Hopper |
| ¡ Tensor Memory Accelerator producer-consumer |
| pipelines and circular buffering |
| ¡ Progressive optimization from 71 TFLOPS |
| (WMMA) to 618 TFLOPS (WGMMA) |
| ¡ Mapping manual CUDA core patterns to their |
| tensor core equivalents |

Every major deep learning workload spends most of its GPU time performing matrix multiplications. Tensor cores exist to accelerate exactly this bottleneck. A cluster of H100s trains a 70B-parameter model in weeks rather than months (Llama 3.1 70B took 7.0 million H100-hours).

-----

###### 184 Chapter 7 Tensor cores

Our best manually optimized GEMM from chapter 6 reaches 43 TFLOPS on an H100 (27.8 TFLOPS on the RTX 3090). By the end of this chapter, we will reach 618 TFLOPS, a 14× improvement. The jump comes from tensor cores, fixed-function matrix units that trade the flexibility of CUDA cores for massive throughput on a single operation:

```
D = A × B + C.
```

## 7.1 Tensor core fundamentals

To use tensor cores effectively, we must understand their distinct architecture and execution model. Unlike general-purpose CUDA cores, tensor cores impose strict requirements on data layout and thread cooperation.

### 7.1.1 What are tensor cores?

Tensor cores are specialized hardware units within each streaming multiprocessor (SM) that execute matrix multiply-accumulate operations at significantly higher throughput than CUDA cores. Tensor cores use mixed precision: they multiply at low precision (FP16 or BF16) but accumulate at higher precision (FP32), combining high throughput with numerical stability. Although CUDA cores are general-purpose scalaror vector-processing units, tensor cores are fixed-function units optimized for a specific operation: D = A @ B + C, where A, B, C, and D are matrix fragments. A CUDA core performs one fused multiply-add (two FLOPs) per clock cycle. A tensor core, by contrast, performs an entire 16 ×16 × 16 or larger matrix operation in a single instruction. Because the accumulator runs at FP32 while the inputs are FP16, training remains numerically stable despite the lower-precision multiplications: rounding errors don't drift and compound across thousands of gradient steps. On an H100, this translates to dramatic performance differences: 67 TFLOPS for FP32 on CUDA cores versus 989.5 TFLOPS for dense FP16 on tensor cores, roughly a 15× improvement. (NVIDIA's headline figure of 1,979 TFLOPS additionally assumes 2:4 structured sparsity, which doubles the number but no longer compares like for like.) But this speedup comes with constraints. Tensor cores require specific matrix dimensions, data layouts (often column-major), and precision formats. They work best when your computation naturally decomposes into matrix multiply-accumulate patterns. Operations like element-wise additions, activations, or irregular memory accesses see no benefit and may even be slower because of the overhead of managing tensor core fragments. Figure 7.1 shows the full cuBLAS-style GEMM: C = alpha \* (A @ B) + beta \* C, with scalar scaling and accumulation into an existing matrix. Both cuBLAS and tensor cores support various precision formats (FP16, BF16, TensorFloat-32 [TF32, which we enabled back in chapter 4], FP8, and FP4 on newer architectures), with the accumulator typically at FP32 regardless of input type. Tensor core Matrix Multiply-Accumulate (MMA) instructions strip this down to D = A \* B + C, where C is simply the accumulator from the previous iteration, with no scalar weighting. The alpha/beta scaling that cuBLAS provides is handled as a separate epilogue step, not by the tensor core hardware itself.

-----

###### Tensor core fundamentals 185

**cuBLAS GEMM Tensor core MMA**

| alpha * | A FP16 | @ | B FP16 | + beta * | C FP32 |
|---|---|---|---|---|---|

C = alpha \* (A @ B) + beta \* C

**Full matrix, any size. Scalars handled by library.**

**Figure 7.1 The full GEMM equation with scalars alpha and beta. Tensor core MMA instructions skip the scalar scaling, computing D = A × B + C directly on small matrix fragments. Input precision varies by generation: FP16, BF16, TF32, FP8, or FP4.**

| A | * | B | + | C |
|---|---|---|---|---|
| FP16 |  | FP16 |  | accumulator |

D = A \* B + C

**16 x 16 fragments. No scalars. Hardware-managed.**

### 7.1.2 When to use tensor cores

Tensor cores shine whenever the workload is dominated by large matrix multiplications. Think about transformer QKV projections, batches of attention GEMMs, convolution layers re-expressed through im2col, or any large language model training run: once the math reduces to dense matrix multiply-accumulate, the tensor hardware takes over. They struggle when the tiles are too small, when the work is element-wise, or when memory addresses wander. Matrices below roughly 64 × 64 rarely amortize launch overhead, activations and normalizations see no benefit, and irregular access patterns or non-MMA primitives, such as reductions, are better left to CUDA cores. The break-even point varies by GPU and matrix size, but a useful heuristic is this: if your kernel spends more than 50% of its time in matrix multiplies, tensor cores will likely provide significant speedup. In chapter 6, our best manually optimized GEMM achieved about 27.8 TFLOPS on the RTX 3090. This chapter moves to an H100, where the same CUDA-core approach tops out around 43 TFLOPS thanks to the bigger chip. As we will see, even a basic tensor core implementation on that H100 can reach 71 TFLOPS, and advanced WGMMA kernels push beyond 600 TFLOPS.

### 7.1.3 A brief history of tensor core instructions

Tensor core programming has evolved through several generations, each adding new capabilities and improving performance: ¡ *MMA (Volta hardware2017; PTX instruction 2019)-Volta's V100 introduced tensor* cores in 2017, initially reachable only through WMMA. The lower-level mma.sync PTX instruction arrived with PTX ISA 6.4 in CUDA 10.1 (2019). MMA operates at the PTX level and requires explicit register management. Although powerful, its low-level nature and limited tooling made it challenging to use. MMA supported

-----

###### 186 Chapter 7 Tensor cores

FP16 input with FP32 accumulation, targeting mixed-precision training workloads. We will not implement MMA in this chapter (WMMA supersedes it for most use cases), but it remains historically important as the first hardware-accelerated matrix operation. ¡ *WMMA (Volta, 2017)-Warp Matrix Multiply Accumulate shipped in CUDA 9.0*

(2017) providing a higher-level C++ API for the V100's tensor cores. WMMA abstracts away register layouts through "fragment" types and provides clean load, mma, and store operations. It operates at warp granularity (32 threads) and supports 16 × 16 × 16 matrix tiles. WMMA made tensor cores accessible to a much broader audience, and subsequent architectures (Turing, Ampere) extended it with new precisions such as TF32 and BF16. This is where we will begin our hands-on implementation. ¡ *WGMMA (Hopper, 2022)-Warp Group Matrix Multiply Accumulate on the*

H100 operates at warp group granularity (128 threads = 4 warps). It introduces asynchronous tensor core execution, allowing compute to overlap with memory movement performed by the separate Tensor Memory Accelerator (TMA) engine. WGMMA supports larger tiles (up to 64 × 256 × 16) and includes hardware-managed barriers for producer-consumer patterns. We will spend most of this chapter on WGMMA. ¡ *TCGen05 (Blackwell, 2024)-The fifth-generation* tensor core instruction on

the B200 further improves efficiency and adds support for FP4 precision. B200 GPUs are now available on major cloud providers, and CUTLASS has supported TCGen05 since version 3.8, and the 4.x line extends that support across the Blackwell family. We discuss TCGen05 conceptually in this chapter's conclusion but defer implementation to chapter 11 (CUTLASS), where library abstractions make it more accessible.

### 7.1.4 Tensor core performance across GPU generations

Understanding tensor core performance requires examining the instruction sets and precision formats together. Our pedagogical approach pairs each instruction type with specific precision formats that best demonstrate its capabilities. Table 7.1 summarizes tensor core performance across GPU generations.

**Table 7.1 Tensor core performance (dense TFLOPS, FP32 accumulate)\***

| GPU | Architecture | FP32 (CUDA) | TF32 | FP16 | BF16 | FP8 | FP4 |
|---|---|---|---|---|---|---|---|
| A100 80GB H100 80GB B200 RTX 3090 | Ampere Hopper Blackwell Ampere | 19.5 67 75 35.6 | 156 495 1,125 35.6 | 312 990 2,250 71 | 312 990 2,250 71 | - 1,979 4,500 - | - - 9,000 - |

-----

###### WMMA: The high-level tensor core API 187

**Table 7.1 Tensor core performance (dense TFLOPS, FP32 accumulate)\* (continued)**

| GPU | Architecture | FP32 (CUDA) | TF32 | FP16 | BF16 | FP8 | FP4 |
|---|---|---|---|---|---|---|---|
| RTX 4090 RTX 5090 | Ada Lovelace Blackwell | 82.6 105 | 82.6 105 | 165 210 | 165 210 | - 838 | - 1,676 |

\* Data center values are from NVIDIA data sheets. Consumer values are from the appendixes of the Ampere GA102, Ada, and RTX Blackwell architecture whitepapers. Structured sparsity (2:4) doubles the tensor core numbers but is not shown.

For our instruction-precision pairing strategy, we anchor MMA to TF32 and FP16 on A100 so you can see the PTX fundamentals on hardware many labs already own; we pair WMMA with FP16 and BF16 on RTX 3090/4090 cards to keep the exercises accessible; we reserve FP8 experiments for WGMMA on H100, where asynchronous execution matters; and we treat TCGen05 with FP4 as forward-looking context for B200 and the next RTX generation. Note that dropping precision roughly doubles throughput at each step (TF32 to FP16 to FP8 to FP4). Hopper delivers about a 3× jump over Ampere at the same precision, and Blackwell's dual-die B200 roughly doubles Hopper again. Consumer RTX boards expose many of the same tensor-core instructions but at lower throughput due to fewer SMs and tighter power envelopes. For example, the RTX 3090's tensor core TF32 throughput equals its CUDA core FP32 throughput (35.6 TFLOPS), while the data center A100 delivers over 4× that (156 TFLOPS) at the same precision. This creates an accessibility ladder: most readers can practice on RTX cards today, while A100, H100, and B200 illustrate where the hardware is heading. This performance matrix guides our chapter structure: we begin with accessible WMMA implementations on widely available hardware, then progress to advanced WGMMA concepts that readers can understand conceptually even without H100 access.

**TIP** You don't need to own data center hardware to run the code in this chapter. WMMA works on any GPU from Volta onward, including every RTX consumer card from the 2000 series up, so your local machine likely supports it. For WGMMA (Hopper), H100 instances are available for under $2/hour on cloud providers like Voltage Park, which I use and partner with for this book. For TCGen05 (Blackwell), B200 access is available through Prime Intellect and Modal. Even if you can't run the WGMMA or TCGen05 code yourself, every kernel in this chapter includes output and profiling results so you can follow the optimization journey without the hardware.

## 7.2 WMMA: The high-level tensor core API

WMMA provides the most straightforward entry point to tensor core programming. It exposes matrix operations through three key functions: load\_matrix\_sync, mma\_sync, and store\_matrix\_sync. These functions operate on "fragments," or opaque types that

-----

###### 188 Chapter 7 Tensor cores

represent a thread's portion of a larger matrix tile. Under the hood, fragments are distributed across the 32 threads of a warp, but the programmer works with them as abstract matrix slices. The basic WMMA workflow mirrors the tiling patterns from chapter 6. Each thread block claims a tile of the output matrix, stages the matching input tiles in shared memory, and lets its warps pull fragments with load\_matrix\_sync. Those fragments feed

```
mma_sync, the results land back in shared memory via store_matrix_sync, and the block
```

finally writes the accumulated tile to global memory. Let's examine the implementation in book.cu/5\_tensor\_cores/kernels/8\_wmma .cuh.

### 7.2.1 WMMA fragments and tiling strategy

The kernel uses template parameters to control tiling at multiple levels, as shown in the following listing.

**Listing 7.1 WMMA template parameters**

```
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16,
int WMMA_TILE_M = 4, int WMMA_TILE_N = 2, int WARP_TILE_M = 2,
int WARP_TILE_N = 4>
__global__ void gemm_wmma_tiled(...) WMMA_M, WMMA_N, WMMA_K define the
```

**dimensions of a single WMMA operation (always 16 × 16 × 16 for FP16).**

To understand WMMA's tiling hierarchy, picture a large grid representing the output matrix, as shown in figure 7.2. The thread block claims a 128 × 128 region of that grid. Within that region, we assign each warp a smaller rectangle (32 × 64 in our case). Each warp then tiles its rectangle with 16 × 16 WMMA operations, stepping through it systematically. This hierarchy maps naturally to the GPU's execution model: warps execute independently, but all threads within a warp operate in lockstep.

**Block tile One warp's tile grid One WMMA op**

|  |  |
|---|---|
| zoom | 16x16 16x16 16x16 16x16 D = A x B + C 32 zoom |
| 128 x 128 | 16 x 16 x 16 16x16 16x16 16x16 16x16 4,096 FMAs WARP_TILE_M=2, WARP_TILE_N=4 |

BM=128, BN=128

**Figure 7.2 WMMA tiling hierarchy: A 128 x 128 block tile decomposes into warp regions (32 x 64), each containing a 2 x 4 grid of 16 x 16 WMMA operations.**

-----

###### WMMA: The high-level tensor core API 189

WMMA distributes each 16 × 16 tile across all 32 threads of a warp through "fragments." A fragment is one thread's share of the 16 × 16 tile, held in its registers. load\_ matrix\_sync fills every thread's fragment, and the threads never need to know which elements they hold. When you call mma\_sync, all 32 threads simultaneously feed their fragment data to the tensor core, which computes the full 16 × 16 × 16 operation in hardware. This is fundamentally different from chapter 6, where each thread explicitly managed every value it computed.

Figure 7.3 visualizes this distribution pattern. Each shade of gray represents a group of four threads, showing how the 256 elements of a 16 × 16 tile are scattered noncontiguously across the warp. The exact mapping is hardware-specific and opaque to the programmer.

**Figure 7.3 WMMA fragment distribution across a 16 x 16 tile. Each of the 32 warp threads holds eight noncontiguous elements. The exact mapping is hardware managed and opaque to the programmer.**

The following listing shows how block-level tile dimensions are computed.

-----

###### 190 Chapter 7 Tensor cores

| Listing 7.2 Block tile size calculation |
|---|
| Block height: constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; 16 × 4 × 2 = 128 rows constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; Block width: constexpr int BK = WMMA_K; 16 × 2 × 4 = 128 columns K dimension: always 16 for WMMA (we iterate across K) Each warp computes a WARP_TILE_M * WARP_TILE_N grid of WMMA operations. Each WMMA operation produces a 16 × 16 output tile, so each warp is responsible for a 32 × 64 region of the output. Before the warp can use WMMA operations, we need to load tiles from global memory into shared memory. This stage is still manual: WMMA automates the matrix multiply, not the memory movement. The key challenge is distributing work across 256 threads (our block size) to load a 128 × 16 tile of A and a 16 × 128 tile of B. Each thread should load eight FP16 values (one int4 vector) to maximize memory bandwidth. The arithmetic for computing each thread's load position is designed around this constraint. For matrix A (128 rows × 16 columns), we want adjacent threads to load adjacent columns for coalescing. If each thread loads eight columns, we need two threads per row (16 / 8 = 2). This is why tid / 2 gives us the row index-every two threads advance to the next row. The column offset (tid % 2) * 8 alternates between 0 and 8, giving the two threads in each row their nonoverlapping eight-element chunks. For matrix B (16 rows × 128 columns), we need 16 threads per row (128 / 8 = 16). Thus, tid / 16 gives us the row index (every 16 threads start a new row), and (tid % 16) * 8 gives each thread its column offset within that row. The following listing shows this vectorized loading pattern. |
| Listing 7.3 Vectorized tile loading |

`__shared__ fp16 sA[BM][BK];` **Shared memory tiles for A and B**

```
__shared__ fp16 sB[BK][BN]; (128 × 16 and 16 × 128)
```

`const int load_smem_a_m = tid / 2;` **For A: 2 threads per row, so tid / 2 gives the**

`const int load_smem_a_k = (tid % 2) * 8;` **row and (tid % 2) \* 8 gives the column offset.**

```
const int load_smem_b_k = tid / 16;
```

`const int load_smem_b_n = (tid % 16) * 8;` **For B: 16 threads per row, so tid / 16 gives the**

**row and (tid % 16) \* 8 gives the column offset.**

```
if (a_vec_valid) {
*reinterpret_cast<int4 *>(smem_a_ptr) = Vectorized load using int4
*reinterpret_cast<const int4 *>(gmem_a_ptr); (128 bits = 8 FP16 values)
}
```

-----

###### WMMA: The high-level tensor core API 191

Once data is in shared memory, warps load fragments using load\_matrix\_sync. This is where the "magic" of WMMA begins: we're no longer thinking about individual elements but about entire 16 x 16 tiles. Each warp owns multiple output tiles (in our case, a 2 × 4 grid of tiles, so eight 16 × 16 tiles in total). For each tile, the warp must load the corresponding fragments from A and B. The fragment type declares the thread's role: is this thread carrying a piece of the

```
A matrix, the B matrix, or the accumulator? The hardware uses this type information to
```

distribute the 16 × 16 matrix correctly across the 32 threads. When we call load\_matrix\_sync, something interesting happens under the hood. The hardware examines the shared memory pointer and the leading dimension, then automatically gives each of the 32 threads its assigned portion of the 16 × 16 tile to hold in registers. Thread 0 doesn't get elements [0-7], and thread 1 doesn't get elements [8-15]; the distribution is more sophisticated, optimized for the tensor core's internal data flow. We don't need to know the distribution pattern. We declare the fragment, call the load function, and trust the hardware to handle the rest. The following listing demonstrates this fragment loading.

| Listing 7.4 Fragment loading from shared memory |
|---|
| Each warp declares an array of matrix_a fragments. wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, fp16, wmma::row_major> A_frag[WARP_TILE_M]; for (int i = 0; i < WARP_TILE_M; ++i) { int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M; |
| wmma::load_matrix_sync(A_frag[i], &sA[warp_smem_a_m][0], BK); |

```
}
```

###### Computes the starting row for this warp's Loads one 16 × 16 fragment from shared tile within the block's shared memory

**memory. BK is the leading dimension (stride).**

The hardware distributes the 16 × 16 matrix across all 32 threads, but the distribution pattern is opaque and hardware-specific.

### 7.2.3 Matrix multiply with WMMA

The actual matrix multiply-accumulate happens through nested loops over the warp's tiles. This is the payoff: after all the setup with loading fragments, we issue a single

```
mma_sync call per tile, and the tensor core performs 4,096 FMA (fused multiply-add)
```

operations (16 × 16 × 16) in hardware. Think of mma\_sync as a synchronized team maneuver. All 32 threads in the warp call it simultaneously. Each thread already holds its fragment pieces from the earlier load\_

```
matrix_sync calls. When the function executes, all threads feed their data to the tensor
```

core unit in the SM, which performs the entire 16 × 16 × 16 matrix multiply-accumulate

-----

###### 192 Chapter 7 Tensor cores

in a handful of clock cycles. The result fragments are distributed back across the same 32 threads, ready for the next accumulation or final storage. The nested loop structure mirrors chapter 6's manual tiling, but at a much coarser granularity. Instead of iterating over individual elements, we iterate over 16 × 16 tiles. Instead of computing one FMA per iteration, we compute 4,096 FMAs per tensor core instruction. You might notice that the 128 × 128 block tile divides into only eight warp regions (four along M, two along N), which means eight warps per block, or 256 threads. A block can hold up to 32 warps (1,024 threads), but fewer warps give each warp more work. Each of our eight warps handles a 2 × 4 grid of 16 × 16 WMMA operations, performing eight tensor core instructions per K-tile iteration. If we split the same 128 × 128 block across 32 warps, each warp would handle just a 2 × 1 grid: only two tensor core instructions per iteration. The fragment load overhead (loading A and B fragments from shared memory) would dominate the actual compute. Eight warps is the sweet spot where each warp does enough work to amortize that overhead while still staying within SM register and shared memory limits.

The following listing shows this pattern.

**Listing 7.5 WMMA matrix multiply**

```
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
C_frag[WARP_TILE_M][WARP_TILE_N];
```

###### Accumulator fragments are

```
for (int i = 0; i < WARP_TILE_M; ++i) { declared once at the start.
for (int j = 0; j < WARP_TILE_N; ++j) {
wmma::fill_fragment(C_frag[i][j], zero); Initializes accumulator
```

`}` **fragments to zero**

```
}
for (int tile_k = 0; tile_k < numKTiles; ++tile_k) {
for (int i = 0; i < WARP_TILE_M; ++i) {
for (int j = 0; j < WARP_TILE_N; ++j) {
wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
}
```

`}` **mma\_sync performs** `}` **D = A @ B + C for 16 × 16 × 16 tiles.** The mma\_sync operation is where hardware acceleration happens. This single function call triggers the tensor core to perform 16 × 16 × 16 = 4,096 multiply-accumulate operations. All 32 threads in the warp participate, but the hardware manages the details of how the work is distributed. Note a critical detail in the fragment types: inputs A and B use half (FP16) for throughput, but the accumulator fragment uses float (FP32) to prevent rounding errors from accumulating across iterations. This mixed-precision design is what makes tensor cores practical for training: without the higher-precision accumulator, gradient updates would drift, and training would become unstable.

-----

###### WGMMA: Asynchronous warp group operations 193

### 7.2.4 Storing results

After accumulation, fragments are stored back to shared memory before being written to global memory. The following listing shows the storage pattern.

**Listing 7.6 WMMA result storage**

```
for (int i = 0; i < WARP_TILE_M; ++i) {
for (int j = 0; j < WARP_TILE_N; ++j) {
int store_row = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
int store_col = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
wmma::store_matrix_sync(&sC[store_row][store_col], C_frag[i][j], BN,
wmma::mem_row_major);
```

`}` **Stores each fragment to**

| } | The barrier ensures all warps have finished storing before | shared memory. BN is the leading dimension; |
|---|---|---|
| __syncthreads(); | global write begins. for (int idx = tid; idx < BM * BN; idx += blockDim.x) { | mem_row_major specifies layout. All threads |

```
int row = idx / BN; cooperate to
```

`int col = idx % BN;` **write the block's**

```
int globalRow = blockRow + row; result to global
int globalCol = blockCol + col; memory.
if (globalRow < M && globalCol < N) {
C[globalRow * N + globalCol] = sC[row][col];
}
}
```

This WMMA kernel achieves 71 TFLOPS on an H100 for 4096 × 4096 matrices: a 1.7× improvement over our best manual CUDA core kernel (43 TFLOPS) from chapter 6.

## 7.3 WGMMA: Asynchronous warp group operations

Instead of operating at warp granularity (32 threads), WGMMA operates on warp groups (128 threads = 4 warps). Instead of using synchronous operations that block until complete, WGMMA uses asynchronous instructions that issue operations and return immediately. And instead of manually loading data from global to shared memory, WGMMA integrates with the Tensor Memory Accelerator (TMA), a dedicated direct memory access (DMA) engine that handles data movement independently of the threads. Let's unpack why each of these changes exists. Why four warps? Each Hopper SM contains four warp schedulers, each capable of issuing instructions to one warp per cycle. A warp group aligns exactly with this hardware: all four schedulers issue the same WGMMA instruction on the same cycle, feeding 128 threads' worth of register data to the tensor core in a single coordinated step. Blocks do not execute in lockstep (their warps can be at different program counters), so a block is too loose a grouping. Individual warps are too narrow to saturate the wider tensor core units on Hopper. Why TMA? "TMA" stands for Tensor Memory Accelerator, and the name is precise. It is a dedicated DMA engine on each SM, meaning that it moves data between

-----

###### 194 Chapter 7 Tensor cores

global memory and shared memory without any thread intervention. What makes it a tensor memory accelerator rather than a plain DMA engine is that it understands multidimensional layouts: you give it a descriptor that encodes the tensor's base address, strides, and tile shape, and the hardware walks the multidimensional index space on its own. Threads never compute source addresses or issue load instructions for TMA transfers. This frees all 128 threads to compute while data arrives in the background. Finally, why asynchronous? WMMA's mma\_sync blocked all 32 threads until the multiply-accumulate finished. WGMMA's `mma_async` issues the instruction and returns immediately. The tensor core works in the background while threads move on to other tasks (prefetching the next tile, for example). Explicit fence instructions (warpgroup\_arrive, warpgroup\_wait) let you control exactly when threads must wait for results. This overlap of compute and memory is where most of the performance gain comes from. Our WMMA kernel achieved 71 TFLOPS. Our final WGMMA kernel will reach 618 TFLOPS: an 8.7× improvement and 87% of cuBLAS performance. We will build this through progressive optimizations, starting with a basic WGMMA kernel and advancing to larger tiles, asynchronous loads, and producer-consumer pipelining.

### 7.3.1 Basic WGMMA: The first asynchronous tensor cores

Our first WGMMA kernel (book.cu/5\_tensor\_cores/kernels/9\_wgmma.cuh) introduces the core programming model. Unlike WMMA's high-level C++ API, WGMMA requires PTX inline assembly and explicit management of shared memory descriptors. **NOTE** WGMMA programming requires PTX inline assembly, which is a significant jump in complexity from WMMA's C++ API. We walk through PTX step by step in this section so you understand exactly what the hardware is doing. If you prefer a higher-level approach, chapter 11 covers CUTLASS, which wraps WGMMA (and TCGen05) behind clean C++ template abstractions. You don't need to write PTX to use Hopper tensor cores in production, but understanding it gives you a mental model that makes debugging and optimization much easier. In WMMA, a single warp (32 threads) operates as an independent unit. In WGMMA, four warps execute together as a warp group, all reaching the WGMMA instruction at the same time. This is stricter than normal thread cooperation: these 128 threads must be at the same program counter when the instruction fires. WGMMA operates on 64 × 64 tiles instead of16 × 16 tiles. A single m64n64k16 instruction performs 65,536 FMAs, and four of them complete a full 64 × 64 × 64 tile: 262,144 FMAs in total. The tensor core needs data from all 128 threads simultaneously to fill its wider execution units. That's 4× the threads, 16× the tile area, and 16× the FMAs per instruction: 4× the work per thread. Figure 7.4 shows this tiling hierarchy.

-----

###### WGMMA: Asynchronous warp group operations 195

Global memory (HBM)

cooperative thread load

###### Shared memory (block tile 64x64)

Warp group (128 threads) processes entire 64x64 tile

WGMMA (m64n64k16)

###### Registers (accumulators)

Warp group (128 threads) 64x64 output distributed across threads

**Figure 7.4 WGMMA tiling hierarchy (basic kernel)**

The following listing shows the WGMMA instruction format with PTX inline assembly.

**Listing 7.7 WGMMA instruction format Output: 32 FP32 registers (64 × 64 = 4,096 Encodes shared memory elements distributed across 128 threads) addresses as 64-bit descriptors**

```
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], fp16* sA, fp16* sB) {
uint64_t desc_a = make_smem_desc(&sA[0]);
uint64_t desc_b = make_smem_desc(&sB[0]);
asm volatile( WGMMA
"wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " instruction: 64 ×
```

**64 × 16 tile, FP32**

```
"{%0, %1, %2, ..., %31}," accumulate, FP16
```

`" %32,"` **Input A descriptor inputs** `" %33,"` **Input B descriptor**

```
" %34, %35, %36, %37, %38;\n"
: "+f"(d[0][0]), "+f"(d[0][1]), ..., "+f"(d[3][7]) Register
: "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), constraints:
"n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)), +f = read/write
"n"(int32_t(TransA)), "n"(int32_t(TransB))); float, l = 64-bit
```

`}` **operand,**

**n = compile-time Scaling and transpose flags constant**

### 7.3.2 Understanding PTX inline assembly

Listing 7.7 contains PTX (parallel thread execution) inline assembly, and if you haven't seen it before, it looks like hieroglyphics. PTX is NVIDIA's virtual instruction set architecture: it sits between your C++/CUDA code and the hardware-specific machine code (called SASS). Think of PTX as the GPU equivalent of LLVM IR or x86

-----

###### 196 Chapter 7 Tensor cores

assembly: a low-level, portable representation. ptxas normally compiles it to SASS ahead of time and embeds that in your binary; the driver JIT-compiles the embedded PTX at load time only when the binary carries no SASS for the GPU it finds. Every CUDA kernel you write passes through PTX on its way to the hardware, and understanding it gives you direct control over what the GPU executes. Let's walk through the WGMMA instruction from listing 7.7, character by character. The wrapper starts with asm volatile(…). The asm keyword tells the C++ compiler that the string inside contains assembly instructions, not C++ code. The volatile qualifier prevents the compiler from reordering, optimizing away, or caching the result of this assembly block. Without volatile, the compiler might decide the instruction has no observable side effects and remove it entirely. Inside the asm block, the instruction string reads: "wgmma.mma\_async.sync.aligned

```
.m64n64k16.f32.f16.f16". Every dot-separated token has a specific meaning:
¡ wgmma identifies the instruction family (warp group matrix multiply-accumulate).
¡ mma_async specifies that this is an asynchronous matrix multiply-accumulate: the
```

instruction issues and returns immediately. ¡ `sync` means the executing thread waits until all threads in its warp reach the same instruction before resuming. This is the lockstep requirement we discussed.

```
¡ aligned means every thread in the warp group must execute this same instruc-
```

tion (uniform control flow); anything else is undefined behavior.

```
¡ m64n64k16 specifies the tile dimensions: M = 64 rows, N = 64 columns, and K = 16
```

reduction dimension.

```
¡ f32 is the accumulator (output) precision: 32-bit float.
¡ f16.f16 specifies the input precisions for matrices A and B: both are 16-bit floats.
```

The output operands {%0, %1, …, %31} are placeholders that map to C++ variables through the register constraint list at the bottom. Each %N corresponds to the Nth operand in the constraint list. The 32 output registers hold the 64 × 64 = 4,096 result elements, distributed across the 128 threads of the warp group (32 floats per thread). The %32 and %33 operands are the shared memory descriptors for matrices A and B, and %34 through %38 are compile-time constants controlling scaling and transposition. The constraint strings after the colon control how the compiler maps C++ variables to registers. "+f"(d[0][0]) means "Use a 32-bit float register (f), and this operand is both read and written (+)." "l"(desc\_a) means "Use a 64-bit register (l for "long")."

```
"n"(ScaleD) means "This is an immediate constant (n), baked into the instruction
```

encoding at compile time."

###### Seeing PTX yourself

You can compile any CUDA kernel to PTX with nvcc -ptx. The following listing shows a vector addition kernel and its PTX output.

-----

###### WGMMA: Asynchronous warp group operations 197

**Listing 7.8 Vector addition kernel**

```
__global__ void vecadd(
float* a, float* b, float* c, int
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
c[i] = a[i] + b[i];
}
}
```

Compile with nvcc -ptx -arch=sm\_80 vecadd.cu to produce the following PTX. The key lines, stripped of boilerplate, are

| ld.param.u64 ld.param.u64 ld.param.u64 | %rd1, [param_0]; %rd2, [param_1]; %rd3, | %rd3, [param_2]; |
|---|---|---|
| ld.param.u32 | %r2, | [param_3]; |
| mov.u32 | %r3, | %ctaid.x; |
| mov.u32 | %r4, | %ntid.x; |
| mov.u32 | %r5, | %tid.x; |

```
mad.lo.s32 %r1, %r3, %r4, %r5;
threadIdx.x
```

| setp.ge.s32 | %p1, %r1, %r2; |
|---|---|
| @%p1 bra | $L__BB0_2; |
| ld.global.f32 | %f1, [%rd8]; |
| ld.global.f32 | %f2, [%rd6]; |
| add.f32 | %f3, %f2, %f1; |
| st.global.f32 | [%rd10], %f3; |

```
$L__BB0_2:
ret;
```

Every line maps directly to your C++ code. Registers prefixed with %r hold 32-bit integers, %rd holds 64-bit integers (pointers), %f holds 32-bit floats, and %p holds predicates (Boolean flags). The %ctaid.x, %ntid.x, and %tid.x registers are built-in PTX registers for blockIdx.x, blockDim.x, and threadIdx.x. The mad.lo.s32 instruction is a fused multiply-add on integers, exactly our blockIdx.x \* blockDim.x + threadIdx.x index cal-

```
culation. The @%p1 bra instruction is a predicated branch: if the predicate %p1 is true,
```

jump to the label.

```
n) {
// load pointer 'a'
// load pointer 'b'
// load pointer 'c'
// load int 'n'
// blockIdx.x
// blockDim.x
// threadIdx.x
// i = blockIdx.x * blockDim.x +
// if (i >= n)
// skip to end
// load b[i]
// load a[i]
// a[i] + b[i]
// store to c[i]
```

###### Exercise: Handwrite PTX

Write the PTX for a vector addition kernel from memory. Use ld.param to load the four parameters, compute the thread index with mad.lo.s32, add a bounds check with setp and a predicated branch, load two floats from global memory, add them

with add.f32, and store the result. Then compile nvcc `-ptx and compare your ver-`

sion with the compiler output. This exercise builds the muscle memory you'll need

when reading and writing inline PTX for tensor core instructions on Hopper and Blackwell.

-----

###### 198 Chapter 7 Tensor cores

Shared memory descriptor encoding is critical for WGMMA. Instead of fetching data element by element (or fragment by fragment), threads build a 64-bit descriptor that holds the starting address, the stride between rows (leading dimension), and the stride between tiles. Why go through this complexity? In WMMA, threads explicitly loaded fragments into registers before the tensor core operation. In WGMMA, the tensor core reads directly from shared memory using the descriptor. This eliminates a register-to-register copy and gives the hardware more control over the timing of data movement, enabling the asynchronous execution we'll explore soon. The encoding packs multiple pieces of information into specific bit ranges of the 64-bit descriptor. The address goes in the low bits, strides in the middle bits, and control flags in the high bits. Once encoded, this single 64-bit value becomes the "input" to the WGMMA instruction. The following listing shows how the descriptor is constructed.

**Listing 7.9 Shared memory descriptor encoding**

###### Encodes the address in bits [13:0]: where the tile starts Converts the generic pointer

**to a shared memory address**

```
__device__ uint64_t make_smem_desc(fp16* ptr) {
uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
uint64_t desc = 0x0000000000000000;
desc |= matrix_descriptor_encode(addr);
```

| desc \|= matrix_descriptor_encode((uint64_t)16) << 16; desc \|= matrix_descriptor_encode((uint64_t)1024) << 32; desc \|= 1llu << 62; |  | Encodes the leading dimension |
|---|---|---|
| return desc; | Encodes the stride dimension | byte offset in |
| } | byte offset in bits [45:32]: | bits [29:16]: |
| Swizzle mode in bits [63:62]: 1 selects | for iterating across K | row-to-row |

**the 128-byte swizzle pattern. distance**

This descriptor format allows the hardware to understand the memory layout without additional instructions from threads. The performance benefit: our basic WGMMA kernel achieves 318 TFLOPS with 64 × 64 × 64 tiles, already 4.5× faster than WMMA.

### 7.3.3 WGMMA with larger tiles

The next optimization is straightforward: increase tile sizes. Larger tiles improve arithmetic intensity (compute-to-memory ratio) and reduce the overhead of loading tiles. Think of it this way: every time we load a tile from global memory, we pay a fixed latency cost. With a 64 × 64 tile, we perform 262,144 operations (64 × 64 × 64 with K = 64) per load. With a 128 × 128 tile, we perform 1,048,576 operations per load, four times the computation for twice the bytes loaded. Table 7.2 shows the comparison.

-----

###### WGMMA: Asynchronous warp group operations 199

| Table 7.2 | Tile size progression |  |  |
|---|---|---|---|
| Kernel | Tiles (BM x BN x BK) | Performance | Improvement |
| K9 (basic) | 64 x 64 x 64 | 318 TFLOPS | Baseline |
| K10 (larger tiles) | 128 x 128 x 64 | 433 TFLOPS | 1.36x |

The implementation in book.cu/5\_tensor\_cores/kernels/10\_wgmma\_larger\_tiles .cuh simply changes the tile-size template parameters while keeping the same basic structure. The 128 × 128 tiles double the work per tile load, reducing the relative cost of memory operations. But we are still loading tiles synchronously, blocking the warp group until data arrives from global memory. This leaves the tensor cores idle during loads. The next optimization addresses this.

### 7.3.4 WGMMA with asynchronous memory operations

The TMA is Hopper's dedicated DMA engine for tensor data movement. TMA allows threads to issue a load operation and immediately return to computation while the hardware manages the data transfer in the background.

In chapter 6's tiled GEMM, every thread loaded its share of a tile, waited at

```
__syncthreads(), and then computed, so loading and math took turns. Our WGMMA
```

kernels so far copy tiles with TMA, but every thread still waits for each copy before computing. The fix is the one PyTorch's DataLoader uses, where worker processes prepare the next batch while the GPU trains on the current one. In K11, a producer thread asks TMA for upcoming tiles while the consumer threads run WGMMA on tiles that have already arrived. Here's how the pattern looks in code: a tiny producer set (often just thread 0) issues the TMA copies that move data from global to shared memory, the remaining threads consume that data via WGMMA, and CUDA barriers keep both sides in lockstep so the tensor cores never read half-delivered tiles. The following listing demonstrates the producer-consumer coordination.

**Listing 7.10 Producer-consumer with barriers**

```
__shared__ barrier barA;
```

`__shared__ barrier barB;` **CUDA barriers for producer-**

**consumer coordination**

```
if (threadIdx.x == 0) {
```

`init(&barA, blockDim.x);` **Initializes barriers with the**

```
init(&barB, blockDim.x); expected thread count
```

`}` **Only thread 0 issues TMA** `__syncthreads();` **loads (producer role).**

```
for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
if (threadIdx.x == 0) {
cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA,
```

-----

###### 200 Chapter 7 Tensor cores

```
block_k_iter*BK, num_block_m*BM, barA);
tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
} else {
```

```text
tokenA = barA.arrive(); Non-producer Arrives at the barrier
} threads arrive with a transaction count
barA.wait(std::move(tokenA)); without a (indicates how many
__syncthreads(); transaction. bytes are being loaded)
warpgroup_arrive(); Warp group fence
wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]); before WGMMA
```

```
}
```

###### Performs asynchronous All threads wait for matrix multiplication the load to complete.

Let's trace through one iteration to understand the coordination. Thread 0 (the producer) calls the TMA load function, which dispatches the request to the hardware DMA engine and returns immediately; thread 0 doesn't wait for the transfer to complete. Instead, it "arrives" at a barrier, announcing that it has initiated a transaction. All other threads (consumers) also arrive at the barrier, but with no transaction. The barrier tracks: "I'm expecting 128 arrivals (one per thread) and 1 transaction (the TMA load)." Only when both conditions are met (all threads have arrived and the TMA load has finished) does the barrier release all threads. This is fundamentally different from

```python
__syncthreads(),which waits only for threads, not for background hardware operations.
```

The TMA descriptor encodes the entire tensor layout. The following listing shows how tensor maps are created for TMA operations.

**Listing 7.11 TMA tensor map creation**

```
template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap *tma_map, fp16* gmem_ptr,
int blocks_height, int blocks_width) {
uint64_t gmem_prob_shape[5] = {
(uint64_t)BlockMinorSize*blocks_width, Total tensor dimensions
(uint64_t)BlockMajorSize*blocks_height, in global memory
1, 1, 1
};
uint64_t gmem_prob_stride[5] = { Strides between
```

`sizeof(fp16),` **elements and rows**

```
sizeof(fp16) * BlockMinorSize*blocks_width,
0, 0, 0
```

```
};
uint32_t smem_box_shape[5] = {
uint32_t(BlockMinorSize),
uint32_t(BlockMajorSize),
1, 1, 1
};
```

###### Tile dimensions to load per TMA operation

```
cuTensorMapEncodeTiled(tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
gmem_address, gmem_prob_shape, gmem_prob_stride + 1, smem_box_shape,
```

-----

###### WGMMA: Asynchronous warp group operations 201

```
smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
CU_TENSOR_MAP_SWIZZLE_128B, ...); Hardware swizzling for
```

`}` **bank conflict avoidance** This implementation (book.cu/5\_tensor\_cores/kernels/11\_wgmma\_async\_loads.cuh) achieves 504 TFLOPS, a 1.16× improvement over synchronous loads.

### 7.3.5 WGMMA with maximum tiles

The final optimization pushes tile sizes to the hardware limits and introduces multiple consumer warp groups. We run a 128 × 256 × 64 tile (doubling the N dimension), keep one warp group busy issuing loads while two consume, launch 384 threads per block (128 in each warp group), and cycle data through a three-stage circular buffer. The following listing shows the maximum tile configuration.

**Listing 7.12 Maximum tile configuration Block M: 128 (fits two WGMMA**

###### M = 64 operations) Block N: 256 (fits four

`constexpr int BM = 128;` **WGMMA N = 64 operations)**

```
constexpr int BN = 256;
```

`constexpr int BK = 64;` **Block K: 64 (fits four WGMMA**

```
constexpr int NUM_THREADS = 128 * 3; K = 16 operations)
```

###### 1 producer warp group + 2 consumer warp groups

Each consumer warp group computes a 64 × 256 half of the output tile, so the two groups split the tile along M. The producer keeps all consumers fed with fresh data through the circular buffer. With several buffers, the producer fills one while the consumers compute on another, so the tensor cores rarely wait for a load In the code, this pattern manifests as multiple regions in shared memory, each with its own barrier. The producer issues TMA loads to all three regions, staggered across loop iterations. Consumers process whichever region has completed its load. The barriers ensure correctness: no consumer touches a region until its TMA load finishes, and no producer overwrites a region until consumers are done with it. K11 already ran this pipeline with five buffers and one consumer warp group. The jump from 504 to 618 TFLOPS comes from the second consumer warp group and the wider 128 × 256 tile. Figure 7.5 visualizes this producer-consumer pipeline. The producer warp group (WG0) issues TMA loads into a rotating set of three shared memory buffers. Two consumer warp groups (WG1, WG2) run WGMMA on whichever buffer has finished loading. After the initial fill, all three warp groups are active at every time step: the producer always loads while both consumers compute, and barriers ensure that no warp group touches a buffer that isn't ready.

-----

###### 202 Chapter 7 Tensor cores

Producer (WG0) TMA load TMA load TMA load TMA load TMA load TMA load

buf 0 buf 1 buf 2 buf 0 buf 1 buf 2

Consumer (WG1) compute compute compute compute compute buf 0 buf 1 buf 2 buf 0 buf 1

Consumer (WG2) compute compute compute compute buf 0 buf 1 buf 2 buf 0 time

**Figure 7.5 Producer-consumer pipeline with three-stage circular buffer. The producer warp group loads data via TMA while consumer warp groups run WGMMA on completed buffers, overlapping memory and compute.**

This kernel (book.cu/5\_tensor\_cores/kernels/12\_wgmma\_max\_tiles.cuh) achieves 618 TFLOPS: 87% of cuBLAS performance and a 1,234x improvement over the naive CUDA core implementation from chapter 6.

## 7.4 Performance analysis: CUDA cores vs. tensor cores

Let's consolidate our performance journey across chapters 6 and 7. Table 7.3 summarizes our optimizations.

**Table 7.3 Complete GEMM optimization progression (4096 x 4096 FP16)**

**Kernel Technique Time (ms) TFLOPS Speedup vs. naive**

Chapter 6: CUDA cores

| K1 | Naive (1 thread = 1 output) | 273.9 | 0.5 | 1x |
|---|---|---|---|---|
| K2 | Global memory coalescing | 24.4 | 5.6 | 11x |
| K3 | Shared memory blocking | 12.8 | 10.7 | 21x |
| K4 | 1D register tiling | 5.4 | 25.5 | 51x |
| K5 | 2D register iling | 4.0 | 34.5 | 69x |
| K6 | Vectorized loads | 3.2 | 42.9 | 86x |

Chapter 7: Tensor cores

| K8 (WMMA) | Warp-level tensor cores | 1.9 | 71.0 | 142x |
|---|---|---|---|---|
| K9 (WGMMA basic) | Warp group asyncltensor cores | 0.43 | 318 | 633x |
| K10 (WGMMA larger) | 128 x 128 tiles | 0.32 | 433 | 864x |
| K11 (WGMMA async) | TMA producer-consumer | 0.27 | 504 | 1,004x |

-----

###### Performance progression analysis 203

**Table 7.3 Complete GEMM optimization progression (4096 x 4096 FP16) (continued)**

| Kernel | Technique | Time (ms) | TFLOPS | Speedup vs. naive |
|---|---|---|---|---|
| Chapter 6: CUDA cores |  |  |  |  |
| K12 (WGMMA max) | Maximum tiles (128 x 256) | 0.22 | 618 | 1,234x |
| Baseline |  |  |  |  |
| cuBLAS | NVIDIA library | 0.19 | 713 | 1,479x |

## 7.5 Performance progression analysis

The following are some key insights from this progression:

¡ *CUDA core phase (K1-K6)-Manual optimizations* achieved an 86× speedup

through memory hierarchy optimization. The largest jump came from memory coalescing (11×), which fixed how threads access global memory. Shared memory added another 2×, register tiling squeezed out 3.3×, and vectorization gave a final 1.2×. Each optimization attacked a different bottleneck: first memory bandwidth, then reuse, then arithmetic intensity, and then transaction efficiency. ¡ *The WMMA jump (K6 → K8)-Moving from manual CUDA cores (43 TFLOPS)*

to WMMA (71 TFLOPS) gave us a 1.7× speedup. The algorithm is the same, but with fixed-function units performing the multiply-accumulate instead of generalpurpose ALUs. The 1.7× speedup may seem modest compared with marketing claims of "10 to 20× speedups," but that's because our chapter 6 kernel was already highly optimized. Against a naive implementation, WMMA shows a 142× speedup. ¡ *The WGMMA jump (K8 → K9)-The leap from WMMA (71 TFLOPS) to basic*

WGMMA (318 TFLOPS) is where we see a 4.5× improvement. This comes from two sources: larger tiles (64 × 64 instead of16 × 16) mean more work per instruction, and warp group execution (128 threads instead of 32) provides better occupancy and latency hiding. The hardware can keep more tensor cores busy simultaneously. ¡ *WGMMA refinements (K9 → K12)-From basic WGMMA to maximum-size tiles,*

we gained another 1.9× through three incremental optimizations. Moving to 128 × 128 tiles gave 1.36×. K11's producer-consumer pipeline added 1.16× by overlapping loads with compute, and K12's second consumer warp group and 128 × 256 tile added 1.23×. Figure 7.6 illustrates the dramatic scale difference between instruction generations. Each rectangle is drawn to relative scale, showing how a basic WGMMA 64 × 64 × 64 tile covers 64 times more FMAs than a single WMMA operation (the tile takes four m64n64k16 instructions), and how the maximum tile configuration pushes this even further.

-----

###### 204 Chapter 7 Tensor cores

| WMMA | WGMMA Basic | WGMMA Max |
|---|---|---|
| (Volta+, 32 threads) | (Hopper, 128 threads) | (Hopper, 128 threads) |

---

16x

16 x 16 4K FMAs

8x

---

64 x 64 128 x 256 262K FMAs / tile 2M FMAs / tile

(4 instructions)

**Figure 7.6 Tile sizes across tensor core generations. Rectangles are drawn to relative scale: a WGMMA Basic tile covers 64x more FMAs than a single WMMA op (using four m64n64k16 instructions), and the maximum WGMMA tile pushes this to 512x.**

¡ *The final gap (87% of cuBLAS)-We achieved* 618 TFLOPS compared with cuBLAS's 713 TFLOPS. The remaining 13% comes from microoptimizations beyond the scope of educational code: hand-tuned PTX instruction scheduling, aggressive register allocation, bank conflict elimination through swizzling patterns, and multikernel strategies that dispatch different kernel variants based on matrix dimensions.

### 7.5.1 Decision framework: When to use tensor cores

Use tensor cores when the matrices are large enough to keep them busy (roughly 64 or more in every dimension), when the profile shows that you are compute-bound, when you can supply the column-major layouts WGMMA expects, and when most of the runtime falls inside matrix multiplies. Stay on CUDA cores when dimensions shrink below that range, when memory access patterns wander unpredictably, when the kernel is dominated by element-wise work or reductions, or when you need the full flexibility of general-purpose ALUs. For practical implementation, your choice depends on your goals and constraints. If you're building a production system and don't need custom kernels, start with cuBLAS. It achieves 713 TFLOPS and requires zero custom code. Only write custom kernels when you have fusion opportunities (like Flash Attention in chapter 8) or unusual memory layouts that don't fit library assumptions. If you're learning tensor core programming or need custom kernels on Ampere hardware, use WMMA. The high-level API makes fragment management straightforward,

-----

***From CUDA cores to tensor cores: Pattern mapping*** **205**

and the code works on widely available GPUs (A100, RTX 30/40 series). You'll achieve respectable performance (71 TFLOPS) without the complexity of PTX assembly. If you need maximum performance on H100 and are comfortable with low-level programming, implement WGMMA with TMA. The PTX inline assembly and producerconsumer patterns have a steep learning curve, but the 618 TFLOPS result justifies the effort for latency-critical applications. For long-term maintainability, especially when targeting multiple GPU generations, use CUTLASS (chapter 11). CUTLASS provides high-level abstractions over WMMA and WGMMA, handles edge cases, and includes highly optimized templates for common patterns. Hand-tuned kernels are specialized. Change the architecture, the SM count, or the matrix shapes, and a kernel can lose much of its speed. Our K12 kernel does not run on Blackwell at all, because wgmma requires sm\_90a (section 7.7). Before putting weeks into a kernel, consider where else it will run.

## 7.6 From CUDA cores to tensor cores: Pattern mapping

The optimization patterns we carefully crafted in chapter 6 have direct analogs in tensor core programming. Understanding these mappings helps build intuition for when and how to use tensor cores.

### 7.6.1 Manual tiling becomes hardware tiling

In chapter 6, we discovered that keeping partial results in registers was the key to high performance. Each thread maintained a small 2D array of accumulator values (TM × TN), loaded small vectors from shared memory into temporary registers, computed outer products, and accumulated the results. This register tiling pattern emerged from first principles: registers are the fastest memory, so we maximize the work done per register load. Tensor cores encode this exact pattern in hardware. The following listing shows the manual 2D register tiling pattern from chapter 6.

**Listing 7.13 Chapter 6: Manual 2D register tiling**

```
fp16 threadResults[TM * TN]; Each thread declares its
```

**own accumulator array.**

```
for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) { Nested loops iterate
for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) { element by element.
threadResults[resIdxM * TN + resIdxN] =
__hadd(threadResults[resIdxM * TN + resIdxN],
__hmul(regM[resIdxM], regN[resIdxN])); Explicit FMA using
```

`}` **intrinsics: one** `}` **operation per iteration** With WMMA, this entire nested loop is replaced by a single function call, as shown in the following listing.

-----

###### 206 Chapter 7 Tensor cores

**Listing 7.14 Chapter 7: WMMA automatic tiling**

```
wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;
wmma::mma_sync(C_frag, A_frag, B_frag, C_frag); Hardware-managed
```

**fragment: distribution A single call performs 4,096 FMAs across 32 threads (16 × 16 × 16) in tensor core hardware. is implicit.**

The hardware distributes the 16 × 16 output tile across 32 threads, performs all multiply-accumulates, and manages register allocation automatically.

With WGMMA, the scale increases further, as the following listing demonstrates.

**Listing 7.15 Chapter 7: WGMMA large-scale tiling**

`float d[4][8];` **32 accumulators per thread**

**(64 × 64 / 128 threads)**

`wgmma64(d, sA, sB);` **One m64n64k16 instruction: 65,536 FMAs, reading**

**directly from shared memory. The kernel issues four of these to cover a full 64 × 64 × 64 tile (262,144 FMAs).**

A handful of WGMMA instructions replace hundreds of manual multiply-accumulate operations.

### 7.6.2 Vectorized loads become TMA

In chapter 6, we spent considerable effort on vectorized memory transfers. We carefully cast pointers to float4 types, checked alignment, handled edge cases when matrix dimensions weren't divisible by 4, and manually unrolled loops to ensure the compiler generated wide load instructions. This was tedious but necessary: wide loads maximize memory bandwidth by moving 128 bits at a time instead of 32 bits. With TMA, the hardware handles all of this automatically. The following listing shows our manual approach from chapter 6.

**Listing 7.16 Chapter 6: Manual vectorized loads**

```
float4* mat_row_vec = reinterpret_cast<float4*>(matd + bid * N);
```

| float4 matval = mat_row_vec[col]; | Each thread computes | Manual pointer casting for |
|---|---|---|
| partial_sum += (matval.x * vecval.x + matval.y * vecval.y + | its own load address. | 128-bit alignment |
| matval.z * vecval.z + matval.w * vecval.w); | Manual unpacking and element-wise math |  |

This required careful pointer arithmetic, alignment checks, and manual unrolling. The following listing shows how WGMMA and TMA reduce the entire tile load to a single operation.

-----

***From CUDA cores to tensor cores: Pattern mapping*** **207**

**Listing 7.17 Chapter 7: TMA automatic loads**

```
if (threadIdx.x == 0) { A single thread issues
cde::cp_async_bulk_tensor_2d_global_to_shared( the entire tile transfer.
&sA[0], tensorMapA,
block_k_iter*BK, num_block_m*BM, barA);
}
```

###### Descriptor encodes shape, strides, alignment, and swizzling: no manual pointer math

The single TMA call replaces all the manual vectorization logic. The descriptor we created earlier already encodes the tensor shape, strides, alignment requirements, and swizzling patterns, so cp\_async\_bulk\_tensor\_2d\_global\_to\_shared computes the addresses, issues the widest possible memory transactions, applies the swizzle that avoids bank conflicts, overlaps the transfer with ongoing compute, and trips the barrier once the data arrives.

### 7.6.3 Shared memory becomes implicit

In chapter 6, shared memory was an explicit intermediate stage. We manually loaded from global to shared memory, synchronized threads with \_\_syncthreads(), and then loaded from shared memory to registers before computing. Each stage was visible in the code. Tensor cores still use shared memory (it's still the fastest on-chip memory for interthread communication), but the shared-to-register transfer becomes implicit. Listing 7.18 shows the manual shared memory orchestration from chapter 6.

**Listing 7.18 Chapter 6: Manual shared memory management**

`__shared__ fp16 sA[BM][BK];` **Explicit shared memory declarations with** `__shared__ fp16 sB[BK][BN];` **programmer-chosen dimensions**

```
for (int i = tid; i < BM * BK; i += blockDim.x) { Manual cooperative loading:
```

`sA[...] = A[...];` **each thread handles a subset**

| } __syncthreads(); | Explicit synchronization before shared memory reads | of elements. |
|---|---|---|
| fp16 regA = sA[...]; | Explicit shared-to-register transfer before compute |  |

```
result = __hadd(result, __hmul(regA, regB));
```

With tensor cores, the load from shared memory to registers becomes implicit. The following listing shows the equivalent WGMMA code.

**Listing 7.19 Chapter 7: Implicit shared-to-register loads**

`__shared__ alignas(128) fp16 sA[BM*BK];` **Single flat array with alignment constraint**

###### (swizzling handles bank conflicts)

`wgmma64(d, &sA[0], &sB[0]);` **Shared-to-register transfer fused into**

###### WGMMA: no explicit load, no syncthreads

-----

###### 208 Chapter 7 Tensor cores

The WGMMA instruction includes the shared memory read, multiply-accumulate, and register update in a single instruction. The intermediate step (loading from shared to registers) has disappeared from the programmer's view. The hardware still performs this transfer, but it's fused into the WGMMA instruction itself. This reduces instruction count, register pressure, and synchronization overhead.

TCGen05 also moves the accumulator out of registers. A single thread issues tcgen05. mma, and the result accumulates in tensor memory (TMEM), a dedicated on-chip memory, instead of in the registers of a 128-thread warp group.

### 7.6.4 What you still optimize manually

Despite the automation, several critical optimizations remain the programmer's responsibility. You must still choose grid dimensions, block dimensions, and tile sizes for kernel launches, as these decisions dramatically affect occupancy and performance. Tensor cores handle individual tiles, but you decide how to partition the full matrix across blocks and how to iterate across the K dimension. For WGMMA, you must manage memory layout: the instruction reads operands from shared memory in either K-major or MN-major layout, selected per operand by descriptor immediates, and the declared layout must match how you staged the data. Our kernels stage column-major tiles; converting a 4096 × 4096 matrix host-side can cost around 30 ms, more than the GEMM itself, so you choose whether to preconvert, store matrices natively in the matching layout, or stage the layout change during the shared-memory load.

With TMA and producer-consumer patterns, you design the pipeline architecture: decide how many producer threads to use, how many stages to include in the circular buffer, and which barriers coordinate producers and consumers. For epilogue operations beyond basic matrix multiplication, such as adding bias terms or applying activation functions, you implement these separately, though libraries like CUT- LASS provide abstractions for common patterns. Finally, although wgmma.mma\_async is asynchronous, you still place warpgroup\_arrive(), warpgroup\_commit\_batch(), and `warpgroup_wait<N>() instructions` manually. Incorrect synchronization causes race conditions or deadlocks, so understanding the async programming model remains essential.

###### 7.7 TCGen05: Blackwell's tensor cores

Before we conclude, let's briefly examine TCGen05, the fifth-generation tensor core instruction set introduced with NVIDIA's Blackwell architecture (B200). Although we won't implement TCGen05 kernels in this chapter, understanding its capabilities provides context for where tensor core programming is headed. B200 GPUs are broadly available as of early 2026: the major cloud providers (AWS P6-B200 instances, Google Cloud A4 VMs, Azure ND B200 v6) all offer Blackwell access, along with GPU-focused providers like CoreWeave, Lambda Labs, RunPod, and Modal. CUTLASS 4.2+ includes comprehensive TCGen05 examples.

-----

###### TCGen05: Blackwell's tensor cores 209

### 7.7.1 TCGen05 overview

TCGen05 builds on WGMMA's asynchronous programming model while adding several key improvements. The progression from WMMA to WGMMA to TCGen05 follows a consistent theme: each generation increases parallelism, reduces instruction overhead, and tightens the integration between compute and memory hardware. The headline feature is native FP4 (4-bit floating point) tensor core operations. FP4 allows approximately 9,000 TFLOPS of dense theoretical throughput on a single B200, more than 4× the dense FP8 throughput of H100. This extreme quantization is primarily useful for inference, where activation magnitudes are more stable than during training. FP4 represents a fundamental tradeoff: you get 4× more operations per second than FP16, but each operation uses 4 bits instead of16. For models where quantization accuracy is acceptable, this translates to both faster compute and reduced memory bandwidth pressure. There is also improved efficiency. TCGen05 refines the WGMMA instruction encoding, reducing instruction overhead and improving register allocation. Even for FP16 and FP8 operations that were already supported on Hopper, TCGen05 provides measurable performance improvements. The instruction set is leaner, allowing the compiler to generate tighter loops and reduce the latency between consecutive tensor operations. Blackwell's second-generation Tensor Memory Accelerator works more efficiently with TCGen05, providing better overlap between memory movement and compute. The hardware coordination that required careful manual barrier management in WGMMA becomes more automated in TCGen05, reducing the programmer's synchronization burden. WMMA code carries forward to Blackwell unchanged. WGMMA does not: wgmma requires the architecture-specific sm\_90a target, which runs only on compute capability 9.0, so Hopper WGMMA kernels must be ported to the tcgen05 instruction family for Blackwell. The concepts you learn here (async tensor cores, descriptors, producer-consumer staging) transfer directly.

### 7.7.2 Challenges and considerations

TCGen05 has matured significantly since Blackwell's launch in the following ways: ¡ *Hardware availability-B200 GPUs are broadly available. Beyond the hyperscalers* (AWS, GCP, Azure, Oracle Cloud), GPU-focused providers such as CoreWeave, Lambda Labs, RunPod, and Modal offer B200 instances on demand at roughly $3 to $6 per GPU-hour. ¡ *Tooling maturity-CUDA Toolkit 12.8 is the minimum version with full Black-* well support, including TCGen05 instructions and the sm\_100 target architecture. CUTLASS 4.2+ includes Blackwell-specific examples for FP4, FP8, and mixed-precision GEMM operations. Profiling with Nsight Compute and Nsight

-----

###### 210 Chapter 7 Tensor cores

Systems works reliably for TCGen05 kernels. Most cloud provider images ship with CUDA 12.8+ by default on B200 instances. ¡ *FP4 calibration-Making FP4 work effectively requires sophisticated quantiza-*

tion techniques. Unlike FP8, which many models tolerate with minimal accuracy loss, FP4 demands finer-grained block scaling, careful calibration, outlier handling via rotations such as the Hadamard transform, and sometimes quantization-aware training or distillation to recover accuracy. For these reasons, we defer TCGen05 implementation to chapter 11 (CUTLASS), where the CUTLASS library provides higher-level abstractions that handle many of these complexities. By the time you reach that chapter, you will have built intuition from WMMA and WGMMA that transfers directly to TCGen05 programming.

## 7.8 Looking ahead

The tensor core patterns from this chapter (fragments, tiling, warp-level synchronization) become building blocks for chapter 8's Flash Attention. We'll use WMMA for that implementation, keeping it accessible to Ampere GPUs while focusing on the fusion algorithm itself: tiling across the sequence dimension, online softmax, and keeping intermediate data in shared memory rather than writing back to global memory between operations.

###### Summary

¡ Tensor cores provide fixed-function matrix units with massive throughput advantages: 989.5 dense FP16 TFLOPS on H100 versus 67 FP32 TFLOPS on CUDA cores, roughly 15×. ¡ The instruction set has gone from WMMA and MMA on Volta to WGMMA on

Hopper and TCGen05 on Blackwell ¡ WMMA's fragment model mirrors chapter 6's tiling strategy. A warp loads 16 ×

16 × 16 fragments, calls mma\_sync, and stores the results. Basic WMMA reaches 71 TFLOPS on H100, 1.7× faster than our best CUDA-core kernel. ¡ WGMMA operates on warp groups (128 threads) and requires inline PTX,

shared-memory descriptors, and the TMA. ¡ Progressive optimization takes WGMMA from 318 TFLOPS to 618 TFLOPS (87%

of cuBLAS): larger tiles, overlapped loads, and producer-consumer pipelines. ¡ Hardware automation eliminates manual work: register tiling becomes wgmma

`.mma_async, vectorized loads collapse` into TMA calls, and shared-to-register transfers disappear into the instruction. ¡ Programmer responsibilities remain: launch geometry, tile shapes, shared

memory layout staging (a host-side 4096 × 4096 conversion can exceed the GEMM itself), producer-consumer staging, epilogue math, and warpgroup\_\* synchronization.

-----

###### Summary 211

¡ Tensor cores excel for matrices larger than 64 × 64 in compute-bound workloads.

CUDA cores remain appropriate for small tiles, irregular memory patterns, or element-wise operations. ¡ TCGen05 (Blackwell) adds FP4 support and improved TMA integration. Chapter 11 covers TCGen05 programming through CUTLASS.

-----

# 8

## Flash Attention

---

| This chapter covers |
|---|
| ¡ Why standard attention becomes memory-bound and how to spot that shift |
| ¡ Core ideas behind Flash Attention: tiling, online |
| softmax, and SRAM residency |
| ¡ Restructuring loops and work distribution so tiles |
| cooperate efficiently |
| ¡ Building a fused Flash Attention kernel with |
| WMMA tensor cores |
| ¡ Evaluating Flash Attention across versions and |
| hardware generations |
| ¡ Deciding when Flash Attention is the right |
| optimization for your model |

Tensor cores gave us raw compute throughput. But attention has a different bottleneck: the sheer volume of data shuffled between on-chip SRAM and off-chip high-bandwidth memory (HBM).

-----

###### From compute-bound to memory-bound 213

## 8.1 From compute-bound to memory-bound

In the last chapter, we explored tensor cores and took GEMM from 43 to 618 TFLOPS on an H100. Our focus was almost entirely on optimizing raw arithmetic, ensuring that every GPU cycle was spent doing useful math. But as we move into optimizing complex operations like the attention mechanism, we encounter a different kind of performance barrier.

Standard self-attention's problem is memory traffic. It requires computing a large intermediate matrix, often called the attention score matrix, which must be written out to and then read back from HBM. As the context length of our models grows, this matrix becomes enormous, and the cost of moving that data completely overshadows the cost of the actual computation. Our optimization problem shifts from compute-bound to memory-bound.

This chapter introduces Flash Attention, a technique that redesigns the attention algorithm to eliminate this HBM bottleneck. The algorithm violates several intuitions that experienced GPU programmers have built up from working with simpler operations. These are the most common misconceptions: that tiling attention should work the same way as tiling GEMM, that partial softmax across tiles requires global communication between thread blocks, and that the familiar "inching inwards" loop from matrix multiplication has no place in the algorithm.

We will confront these misconceptions directly, building the algorithm up by identifying each hurdle and working through it. We start by dissecting the standard attention implementation to pinpoint the memory bottleneck, then work through the core insights that unlock the algorithm, and finally build a Flash Attention kernel from the ground up using WMMA tensor cores for the matrix multiplications.

### 8.1.1 The HBM bottleneck in standard attention

To understand why Flash Attention is necessary, we must first analyze the standard implementation of the attention mechanism. The core of attention is a sequence of three operations. First, we calculate the dot product of the query (Q) and key (K) matrices to produce a score matrix S with S = Q @ K.T. Next, we apply a softmax function to the score matrix S to get the attention weights P, computing P = softmax(S). Finally, we multiply the attention weights P by the value (V) matrix to get the final output O with O = P @ V.

Let's consider the shapes of these matrices for a typical language model. If we have a sequence of length N and a head dimension d, then Q, K, and V each have shape

```
(N, d). The intermediate matrix S (and P) will have shape (N, N). For a context length
N of 8,192, this S matrix would be 8192 × 8192. If we use 32-bit floats, this single matrix
```

requires 8192 \* 8192 \* 4 bytes, which is 268 megabytes. For a modern model with a 128K context length, this grows to over 65 gigabytes for a single attention head.

The problem is not the computation itself. The matrix multiplications are exactly what tensor cores are designed to accelerate. The problem is that the entire S matrix must be written to HBM after the first matrix multiplication, and then the entire matrix must be read back from HBM to perform the softmax, which then writes its result P back

-----

###### 214 Chapter 8 Flash Attention

to HBM, which is then read again for the final matrix multiplication. This data movement between the fast on-chip SRAM and the slower, off-chip HBM is the bottleneck. The GPU is left waiting for data, a situation illustrated in figure 8.1.

| Q @ K.T Softmax | Write S (NxN matrix) Read S (NxN matrix) Write P (NxN matrix) | Write S (NxN matrix) Read S (NxN matrix) Write P (NxN matrix) | Write S (NxN matrix) Read S (NxN matrix) Write P (NxN matrix) | HBM (slow memory) |
|---|---|---|---|---|
| P @ V GPU SM (fast SRAM) | Read | P (NxN | matrix) |  |

###### The N x N intermediate matrices are too large for SRAM and

**must be stored in HBM. The constant read/write cycles to slow HBM dominate the runtime, making the operation memory-bound.**

**Figure 8.1 Standard attention memory bottleneck**

Memory bus

## 8.2 Designing Flash Attention

Now that we see why standard attention is memory-bound, we can design a solution. The core ideas are tiling (processing the attention matrix in blocks that fit in SRAM) and online softmax (maintaining running statistics so we never need the full row of scores at once). We'll build intuition for why these ideas work before writing any code.

### 8.2.1 Building intuition: Why this problem is harder than it looks

Before we examine the Flash Attention solution, we need to confront a fundamental tension. If you already understand how tiled matrix multiplication works on GPUs, you might be thinking, "This seems straightforward. Just break the computation into tiles that fit in SRAM, as we do for GEMM." But attention has a property that makes it fundamentally different from matrix multiplication, and this difference is the source of the most common misconceptions about Flash Attention.

The challenge is the softmax operation. In a standard matrix multiplication C = A @ B, each element of the output C[i,j] depends only on row i of A and column j of B. The computation is embarrassingly parallel. You can compute any output tile independently, without needing information from other tiles. Different thread blocks can work on different output tiles simultaneously, each loading just the data it needs, computing its result,

-----

###### Designing Flash Attention 215

and writing back to global memory. Softmax breaks this independence. To compute the softmax probability for a single element in a row, you need the maximum value and the sum of exponentials across the entire row. If you have a sequence length of8,192 tokens, computing the attention weight for token 0 attending to token 1 requires knowing the scores for token 0 attending to all 8,192 tokens.

We want to process attention in small tiles that fit in SRAM, but softmax seems to require the full row of scores. How can we compute a "partial softmax" on a small tile when the mathematical definition of softmax requires global information? If we try to compute partial results tile by tile, where would we store the intermediate state as we march across a long sequence? Must those intermediate values live in slow global memory, destroying our performance gains? And how does the familiar "inching inward" pattern from GEMM fit into an algorithm that must also march horizontally across the sequence?

### 8.2.2 The Flash Attention solution: Tiling and online softmax

Flash Attention elegantly solves the memory bottleneck by restructuring the computation to avoid writing the large N x N intermediate matrices to HBM. It achieves this through two key techniques: tiling and online softmax. Instead of computing the full attention matrix at once, we process it in smaller blocks, or tiles that are small enough to fit in the GPU's fast SRAM. But as we just discussed, the softmax operation seems to require global information. The breakthrough is that we can compute softmax iteratively, maintaining running statistics that allow us to process one tile at a time while still producing mathematically exact results.

### 8.2.3 Tiling: Breaking the problem down

The core idea behind tiling is to break down the large Q, K, and V matrices into smaller blocks. We then compute the attention output block by block, loading only the necessary tiles of Q, K, and V into SRAM for each step. For each block of the output matrix

```
O, we iterate through blocks of the K and V matrices, compute the attention scores that
```

contribute to that output block, and accumulate the results.

This means we never need to form the full S or P matrices in memory. We compute a small submatrix of scores, apply softmax, and multiply by the corresponding block of V, all within the fast on-chip memory of a single streaming multiprocessor (SM). The intermediate results are kept in registers and SRAM until they are accumulated into the final output block, which is then written to HBM. This process is shown in figure 8.2.

### 8.2.4 Online softmax: Computing global statistics iteratively

Tiling is a great start, but it introduces a new challenge. The softmax function,

```
softmax(x_i) = exp(x_i) / sum(exp(x_j)), requires summing over all elements in a row
```

to normalize each element. If we have only a small block of the scores S in SRAM at any given time, how can we compute the correct softmax without knowing the sum over the entire row?

-----

###### 216 Chapter 8 Flash Attention

**Q K V**

[Fe **O**

(N, d) (N, d) (N, d)

Load tiles (N, d)

Write result

Qi Kj Vj

###### SRAM (on-chip)

S and P never leave SRAM. No N x N in HBM.

**Figure 8.2 Flash Attention loads small tiles of Q, K, and V from HBM into fast on-chip SRAM. All intermediate results (the score matrix S and probability matrix P) stay in SRAM and are never written to HBM. Only the final output O is written back.**

This is where the first major misconception arises. You might think we would compute a "local" softmax on each tile, creating independent probability distributions that somehow get combined later. This would be mathematically incorrect. We don't compute multiple local softmaxes. We compute a single, exact softmax iteratively using an algorithm called online softmax. Here's how it works: as we process each tile of scores, we maintain two running statistics in registers or SRAM-a running maximum m and a running sum of exponentials l. When we process the first tile and compute scores for columns 0 through 127, we find the maximum value in those scores and store it as m. We compute the sum of exponentials sum(exp(score - m)) and store it as l. When we process the second tile, with scores for columns 128 through 255, we might find a new, larger maximum value. Let's call this new local maximum m\_new. This new maximum might be larger than our running maximum m. If the new maximum is larger, we face a problem. All our previous exponentials were computed relative to the old maximum m. To maintain mathematical correctness, we

-----

###### Implementing Flash Attention 217

need to "rescale" our previous running sum to be consistent with the new maximum m\_

```
new. The correction factor is exp(m_old - m_new). We multiply our running sum l by this
```

correction factor, then add the new partial sum computed with the new maximum. This ensures that when we finally divide by l at the end, we get exactly the same result as if we had computed the softmax over the entire row at once. In floating point, it matches this up to the usual rounding differences that come from reordering additions. What it never does is materialize the full row of scores in memory.

Figure 8.3 traces this process for a single query row. Read it from left to right: the dashed strip at the top is the full row of scores that never exists in memory, and each step below it pulls one 128-column tile into SRAM and folds it into the three running values `m, l,` and O. The arrows between steps carry the correction factor

```
exp(m_old - m_new), which rescales the old state whenever a new tile raises the maxi-
```

mum. After the last tile, dividing O by l yields exactly the result a full-row softmax would have produced.

one query row of attention scores (length N) - never stored in full

tile 1 (cols 0-127) tile 2 (cols 128-255) tile 3 (cols 256-383) ...

| Step 1 | Step 2 | Step 3 |
|---|---|---|
| Tile 1 scores in SRAM | Tile 2 scores in SRAM | Tile 3 scores in SRAM |

rescale old l, O: m = max(tile 1) c = exp(m\_old - m\_new) m = max(m, tile 2) l = sum(exp(s - m)) l = l x c + new sum O = exp(s - m) @ V1 O = O x c + exp(s - m) @ V2 m = running max l = running denominator (sum of exponentials) O = running output accumulator rescale old l, O: c = exp(m\_old - m\_new) m = max(m, tile 3)

l = l x c + new sum O = O x c + exp(s - m) @ V3

after the last tile: O / l = exact softmax result, as if the whole row was used

**Figure 8.3 Online softmax streams one query row's scores tile by tile. Each step keeps only the current tile in SRAM plus three running values: the max m, the denominator l, and the output O. When a new tile raises the max, the old l and O are rescaled by exp(m\_old - m\_new), so the final O / l equals the exact full-row softmax result.**

## 8.3 Implementing Flash Attention

With the algorithm designed, we move to code. We start by examining a naive attention implementation that exposes the memory bottleneck, then build a fused Flash Attention kernel that eliminates it. Along the way, we'll see how the work distribution strategy, loop hierarchy, and WMMA tensor cores all fit together.

-----

###### 218 Chapter 8 Flash Attention

### 8.3.1 Dissecting the naive implementation

Before we build our optimized kernel, let's examine a straightforward, "naive" implementation of the attention mechanism in CUDA, which you can find in book.cu/ 6\_flash/kernels/naive.cu. This code serves as our baseline. It correctly computes the attention output, but in a way that perfectly demonstrates the memory bottleneck. It uses three separate kernel launches, which we will examine one by one. First is the kernel in listing 8.1, which computes the S = Q @ K.T matrix multiplication. Each thread in this kernel is responsible for computing a single element of the output matrix S.

**listing 8.1 Naive QK.T matrix multiplication kernel**

```
__global__ void naive_qk_matmul_kernel(
float* Q, float* K, float* S,
int N, int d, float scale
) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < N && col < N) {
float sum = 0.0f;
for (int k = 0; k < d; k++) {
sum += Q[row * d + k] * K[col * d + k];
}
S[row * N + col] = sum * scale;
}
}
```

Next, a separate kernel applies the softmax function to the S matrix in place. Each thread in this kernel handles one full row of the S matrix.

**Listing 8.2 Naive row-wise softmax kernel**

```
__global__ void naive_softmax_kernel(float* S, int
int row = blockIdx.x * blockDim.x + threadIdx.x;
if (row < N) {
float* row_ptr = S + row * N;
float max_val = -INFINITY;
for (int i = 0; i < N; i++) {
max_val = fmaxf(max_val, row_ptr[i]);
}
float sum = 0.0f;
for (int i = 0; i < N; i++) {
row_ptr[i] = expf(row_ptr[i] - max_val);
sum += row_ptr[i];
}
```

###### Each thread computes the coordinates of one output element in the S matrix. The thread iterates through the head dimension d to

**compute the dot product.**

###### Writes the final scaled result to the S matrix in HBM

```
N) {
```

###### Assigns each thread to a single row of the S matrix

###### The thread first finds the maximum value in the row for numerical stability.

###### It then computes the exponent of each element (minus the max) and accumulates the sum.

-----

###### Implementing Flash Attention 219

`for (int i = 0; i < N; i++) {` **Finally, it normalizes each**

`row_ptr[i] /= sum;` **element by the sum.**

```
}
}
}
```

The third kernel performs the final P @ V matrix multiplication. This kernel is structurally very similar to the first one.

**Listing 8.3 Naive P@V matrix multiplication kernel**

```
__global__ void naive_sv_matmul_kernel(
float* S, float* V, float* O,
int N, int d
) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < N && col < d) {
float sum = 0.0f;
for (int j = 0; j < N; j++) {
sum += S[row * N + j] * V[j * d + col];
}
```

`O[row * d + col] = sum;` **Writes the final result for one element** `}` **of the output matrix O to HBM**

```
}
```

Finally, in listing 8.4, the host-side C++ function orchestrates these three separate kernel launches. This function clearly shows the allocation of the intermediate S matrix in HBM and the sequence of kernel calls that read from and write to it, causing our performance bottleneck.

**Listing 8.4 Host logic for the naive attention implementation**

```
float* S;
cudaMalloc(&S, N * N * sizeof(float)); Allocates memory on the GPU (HBM)
```

**for the large N × N score matrix**

```
const int BLOCK_SIZE = 16;
dim3 block_qk(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid_qk((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) /
BLOCK_SIZE);
naive_qk_matmul_kernel<BLOCK_SIZE><<<grid_qk, block_qk>>>(
Q_ptr, K_ptr, S, N, d, scale);
```

###### Launches the first kernel Launches the

`dim3 block_softmax(256);` **to compute S = Q @ K.T second kernel to**

```
dim3 grid_softmax((N + 255) / 256); apply softmax to
naive_softmax_kernel<<<grid_softmax, block_softmax>>>(S, N); S in place
dim3 block_sv(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid_sv((d + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) /
BLOCK_SIZE);
```

-----

###### 220 Chapter 8 Flash Attention

```
naive_sv_matmul_kernel<BLOCK_SIZE><<<grid_sv, block_sv>>>(
S, V_ptr, O_ptr, N, d);
```

###### Launches the third kernel

`cudaFree(S);` **to compute O = P @ V** This sequence of separate kernels, each requiring HBM access, is precisely what Flash Attention aims to eliminate through kernel fusion.

### 8.3.2 Understanding the work distribution strategy

Before we look at the fused Flash Attention kernel, we need to address the second major misconception: Where do the running statistics live? If we compute softmax iteratively across tiles while GPUs parallelize work across many thread blocks, you might reasonably conclude that different thread blocks must be working on different tiles of the same output row. To share the running maximum m and running sum l, those statistics would need to live in global HBM, where all thread blocks can access them. This would destroy the performance gains we're trying to achieve. The flaw is in the assumption about how work is distributed. In a standard tiled GEMM, we parallelize across both output dimensions. If we're computing a matrix

```
C with shape (M, N), we might launch a grid of thread blocks, with each thread block
```

computing a tile of the output. Block (0,0) computes C[0:32, `0:32], block (0,1)` computes C[0:32, `32:64],` and so on. All these blocks work independently and simultaneously.

Flash Attention uses a different parallelization strategy. Each thread block is assigned a small "brick" of complete output rows. For example, thread block 0 might be responsible for output rows 0 through 31, and thread block 1 might be responsible for output rows 32 through 63. To compute its assigned output rows, each thread block must perform a sequential loop across the entire sequence length. Thread block 0 loads its tile of Q (rows 0-31), then loops through all tiles of K and V, processing columns 0 to 127, then 128 to 255, then 256 to 383, and so on until it has moved across the entire sequence. Figure 8.4 illustrates this work distribution: each thread block owns a complete horizontal slice of the output matrix and moves sequentially across the K/V tiles, keeping all running statistics in its own registers and shared memory. This means the running statistics m and l for rows 0-31 never need to leave thread block 0. They can live in that thread block's registers or shared memory throughout the entire sequential loop. Thread block 1 simultaneously makes its own independent march across the sequence for rows 32-63, maintaining its own separate running statistics in its own registers. There is no communication between thread blocks during the main computation. Because each thread block does a long sequential march rather than processing a single tile and finishing, and because all the intermediate state is kept in on-chip SRAM and registers, the memory footprint stays constant regardless of sequence length. A thread

-----

###### Implementing Flash Attention 221

###### Output O Block 0's sequential march across K/V

| Block 0 rows 0-31 |
|---|
| Block 1 rows 32-63 |
| rows 64-95 |
| rows 96-127 |

d

**Figure 8.4 Each thread block owns complete output rows and marches sequentially across K/V tiles. Running statistics (m, l) stay in registers, with no cross-block communication.**

block processing a sequence of length 8,192 uses the same amount of SRAM as one processing a sequence of length 128,000. The loop is longer, but the memory usage per iteration remains fixed.

### 8.3.3 The three-level loop hierarchy

Now we need to address the third major misconception-one that often troubles programmers who have a deep understanding of tiled GEMM. If you've built a high-performance matrix multiplication kernel, you know that computing an output tile requires inching inward over the shared inner dimension. To compute `C[i,j] = A[i,:] @ B[:,j], you load` a slice of A and a slice of B, multiply them, load the next slices, multiply, and accumulate until you've traversed the full inner dimension. So when you learn that Flash Attention has a "sequential loop across the sequence length," you might try to reconcile this with your GEMM intuition and become deeply confused. How can we compute a full tile of scores without looping over the head dimension d? Where does the inching inward pattern fit into the "horizontal march" across the sequence? Figure 8.5 shows this three-level nesting. The answer is that Flash Attention has a three-level loop hierarchy, and your GEMM intuition is not wrong. It's just operating at a different level. At the grid level, we have parallelism across different output row tiles. Different thread blocks compute different bricks of output rows independently. At the thread block level, we have the sequential horizontal march across the sequence length. This is the main Flash Attention loop, where we iterate through tiles of K and V. At the warp and thread level, we have the inching inwards loop over the head dimension d.

-----

###### 222 Chapter 8 Flash Attention

This is a standard mini-GEMM kernel that computes a tile of scores by accumulating partial products.

These loops nest. The innermost inching inwards loop is completely contained within a single iteration of the middle horizontal march loop. When thread block 0 processes K and V tiles corresponding to sequence positions 0 to 127, it performs a complete GEMM-style computation to produce a full tile of scores for those positions. This tile is consumed immediately by the online softmax update, and then the middle loop advances to the next K and V tiles for positions 128 to 255. The inner GEMM computation runs again, producing another complete tile of scores, which is again consumed immediately. The scores are never accumulated across iterations of the middle loop. Each iteration of the horizontal march produces a complete partial score tile, applies online softmax, updates the output accumulator, and then discards the scores before moving to the next iteration.

###### Grid level: parallel across output row tiles

Block (i) owns output rows [i\*Br .. (i+1)\*Br]

###### Block level: sequential march across sequence length

for j in 0..Tc: load Kj, Vj tiles

###### Warp level: GEMM "inching inwards" over head dim d

for k in 0..d/16: WMMA(Qi\_slice, Kj\_slice)

Q @ K.T softmax P @ V

**Figure 8.5 Flash Attention's three nested loops: Grid-level parallelism across output rows, block-level sequential march across the sequence, and warp-level GEMM over the head dimension**

This nested structure explains why Flash Attention doesn't violate your understanding of matrix multiplication. The matrix multiplications (Q @ K.T and P @ V) are performed using exactly the techniques you know, with inner loops accumulating over the shared dimension. Flash Attention adds an outer sequential loop around these GEMM operations and uses online softmax to combine the results in a way that maintains mathematical correctness without materializing the full attention matrix.

### 8.3.4 The fused Flash Attention kernel

The fused Flash Attention kernel can be found in book.cu/6\_flash/kernels/ fa.cu. Having worked through the conceptual hurdles, we can approach this

-----

###### Implementing Flash Attention 223

implementation with a clear mental model. This single, monolithic kernel replaces the three separate kernels from the naive implementation. By keeping the entire computation on chip and processing the matrices in small tiles, we avoid the costly round trips to HBM. Our implementation uses WMMA tensor core instructions for the heavy lifting of matrix multiplication, which requires us to use half precision (FP16) for the inputs. As you read through the kernel code, keep in mind the four key insights we developed. First, we're computing an exact, iterative softmax using correction factors, not approximate local softmaxes. Second, each thread block owns its output rows completely and maintains all state in private SRAM. Third, the familiar GEMM inching-inwards loop is nested inside the horizontal march loop. Fourth, the output accumulator is a single running total that gets updated in place, not a growing list of partials. We will build the kernel piece by piece. First, let's look at the following listing, which shows the kernel signature and the shared memory setup that is so crucial to this technique.

**Listing 8.5 Flash Attention kernel signature and shared memory layout**

```
template <const int Br, const int Bc>
__global__ void flash_attn_2_5_kernel(
```

`half* Q, half* K, half* V,` **Dynamically allocates a block of**

| int N, int d, | shared memory. The host determines |
|---|---|
| float scale, | the size when launching the kernel. |

`float* O` **Partitions the raw** `) {` **shared memory block**

```
extern __shared__ char smem_raw[]; by creating pointers for
half* Qi = reinterpret_cast<half*>(smem_raw); each tile and variable
half* Kj = Qi + Br * d; we need to store
half* Vj = Kj + Bc * d;
float* Sij_fp32 = reinterpret_cast<float*>(Vj + Bc * d);
half* Sij_fp16 = reinterpret_cast<half*>(Sij_fp32 + Br * Bc);
float* Oi = reinterpret_cast<float*>(Sij_fp16 + Br * Bc);
float* temp_pv = Oi + Br * d;
```

`float* mi = temp_pv + Br * d;` **Scratch space for the Pij @ Vj** `float* mi_new = mi + Br;` **result before it is folded into**

```
float* li = mi_new + Br; the accumulator Oi
}
```

###### The online softmax needs both the previous maximum (mi) and the updated one (mi_new), because the output correction in listing 8.10 is

**computed from their difference.**

The kernel's logic is built around a nested loop. The outer loop iterates over blocks of the output matrix O (and therefore, blocks of Q). The first step inside this loop, shown in the following listing, is to load the corresponding tile of Q from HBM into shared memory and initialize the output and softmax statistics.

-----

###### 224 Chapter 8 Flash Attention

| Listing 8.6 Flash Attention outer loop and Q-tile loading |
|---|
| int Tc = CEIL_DIV(N, Bc); int Tr = CEIL_DIV(N, Br); Threads in the block cooperatively for (int i = 0; i < Tr; i++) { load a Br × d tile of Q from HBM int row_offset = i * Br; into the Qi shared memory buffer. for (int idx = tx; idx < Br * d; idx += blockDim.x) { int r = idx / d; int c = idx % d; if (row_offset + r < N) { Qi[r * d + c] = Q[qkv_off + (row_offset + r) * d + c]; } else { Qi[r * d + c] = __float2half(0.0f); } } for (int idx = tx; idx < Br * d; idx += blockDim.x) { Oi[idx] = 0.0f; } Threads also initialize the output tile Oi if (tx < Br) { to zeros and the softmax statistics mi, mi[tx] = -INFINITY; mi_new, and li to their starting values. mi_new[tx] = -INFINITY; li[tx] = 0.0f; } __syncthreads(); // outer loop continues in the following listings Note that the outer for (int i = 0; i < Tr; i++) loop is still open at the end of listing 8.6: everything from here through the write-back in listing 8.11 executes inside it, once per Q tile. The inner loop iterates through blocks of the K and V matrices. In each iteration, we load tiles of K and V into shared memory and then perform the three core attention operations: S = Q @ K.T, P = softmax(S), and O += P @ V. The following listing shows the start of this inner loop and the cooperative loading of K/V tiles. |
| Listing 8.7 Flash Attention inner loop and K/V-tile loading |

```
for (int j = 0; j < Tc; j++) {
int col_offset = j * Bc; Threads
```

**cooperatively load**

```
for (int idx = tx; idx < Bc * d; idx += blockDim.x) { Bc × d tiles of K and
```

`int r = idx / d;` **V from HBM into the**

###### Kj and Vj shared

`int c = idx % d;` **memory buffers.**

```
if (col_offset + r < N) {
Kj[r * d + c] = K[qkv_off + (col_offset + r) * d + c];
Vj[r * d + c] = V[qkv_off + (col_offset + r) * d + c];
} else {
Kj[r * d + c] = __float2half(0.0f);
Vj[r * d + c] = __float2half(0.0f);
}
```

-----

###### Implementing Flash Attention 225

```
}
```

`__syncthreads();` **A barrier synchronization**

```
// inner loop continues through listing 8.10 ensures that all data is loaded
```

Like the outer loop, this j loop stays open: the score computation, online softmax, and accumulation in listings 8.8 through 8.10 all run once per K/V tile inside the Q-tile iteration. This double nesting is the whole point of the tiling structure: each Q tile accumulates attention across every K/V tile before its output is written. Now comes the first major computation in listing 8.8: the Q @ K.T matrix multiplication. We use WMMA for this, defining fragments and calling the mma\_sync intrinsic.

**into shared memory before any computation begins.**

| Listing 8.8 QK.T matrix multiplication with WMMA |
|---|
| wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag; wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag; wmma::fill_fragment(c_frag, 0.0f); The core tensor core for (int k = 0; k < d; k += 16) { operation. Each warp wmma::load_matrix_sync(a_frag, Qi + k, d); multiplies its 16 × 16 wmma::load_matrix_sync(b_frag, Kj + k, d); fragments and wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); accumulates the result. } wmma::store_matrix_sync(Sij_fp32, c_frag, Bc, wmma::mem_row_major); __syncthreads(); Stores the result tile, Sij, from the warps' registers back into shared memory Listing 8.8 walks through the WMMA API in order, but the picture is easier to read as geometry. Figure 8.6 lays out how each warp's 16 × 16 fragments cover a tile of Q and Kj, where partial products accumulate before store_matrix_sync writes scores back to shared memory for the softmax stage. With a tile of scores in shared memory, we perform the online softmax update shown in listing 8.9. This step involves finding the new maximum, computing the exponen- tials, and updating the running statistics. The kernel keeps two max arrays: mi holds the maximum from the previous K/V tile (the "old" max), and mi_new receives the updated one. Both must survive this step because the output correction in listing 8.10 is com- puted from their difference. |
| Listing 8.9 Online softmax calculation |

| mi[s_row] = mi_new[s_row]; | Promotes the previous iteration's maximum into mi before computing this tile's. After this line, mi is the |
|---|---|
| float row_max = -INFINITY; | old max and mi_new is free to receive the new one. |

-----

###### 226 Chapter 8 Flash Attention

```
for (int c = 0; c < Bc; c++) {
row_max = fmaxf(row_max, Sij_fp32[s_row * Bc + c]);
}
float new_max = fmaxf(mi[s_row], row_max);
mi_new[s_row] = new_max;
float row_sum = 0.0f;
for (int c = 0; c < Bc; c++) {
float exp_val = expf(Sij_fp32[s_row * Bc + c] - new_max);
Sij_fp32[s_row * Bc + c] = exp_val;
row_sum += exp_val;
}
float correction = (mi[s_row] == -INFINITY)
? 0.0f
: expf(mi[s_row] - new_max);
li[s_row] = correction * li[s_row] + row_sum;
__syncthreads();
```

###### The updated maximum goes into mi_new, leaving mi intact for the output correction

**in listing 8.10. Only at the top of the next iteration does it become the "old" max.**

###### Rescales the running sum by

**exp(old\_max - new\_max). The guard handles the first tile, where the old max is still negative infinity.**

(d x N)

K.T

Q S

EN

(N x d)

**(16, 16) @ (16, 16) -> (16, 16)**

| Tensor core instruction as partial GEMM | Figure 8.6 | WMMA tiled |
|---|---|---|
| instead of slow CUDA core block-tiling | matrix | multiplication |

The final computational step is to multiply the partial softmax matrix Pij (which is stored in Sij\_fp32) by the value tile Vj. We convert Sij to FP16 and use WMMA again. The result is corrected using the online softmax statistics and accumulated into the output tile Oi. This accumulation step addresses our fourth and final conceptual hurdle: understanding the nature of the output accumulator. You might worry that as we march across a long sequence, we're creating a growing list of partial results that will exhaust our SRAM. After processing tiles for positions 0 to 127, 128 to 255, 256 to 383, and so on, don't we have dozens of partial output tiles that

-----

###### Implementing Flash Attention 227

need to be stored? The answer is no, and understanding why requires a shift in how you think about accumulation. The Oi buffer in SRAM is not one of many partial tiles. It is a single running total, like a bank account balance or a scratchpad where you update your exam score as you grade each question. When we compute Pij @ Vj for the first tile, we add it to Oi, which starts at zero. When we process the second tile and compute a new Pij @ Vj, we don't store it separately. We scale the existing value in Oi by the correction factor (to account for the updated maximum), then add the new contribution, and overwrite the old value in

```
Oi, as shown in listing 8.10. The information from the first tile is not lost. It has been
```

absorbed into the updated total. No matter how many tiles we process, we always have exactly one Oi buffer in SRAM, and it always holds the sum of all corrected contributions seen so far.

```
Listing 8.10 P@V matmul and output accumulation
for (int idx = tx; idx < Br * Bc; idx += blockDim.x) {
Sij_fp16[idx] = __float2half(Sij_fp32[idx]); Converts the computed Pij tile
```

`}` **back to FP16 to be used in the** `__syncthreads();` **next WMMA operation**

```
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);
for (int k = 0; k < Bc; k += 16) {
wmma::load_matrix_sync(a_frag, Sij_fp16 + k, Bc); Temporarily stores
wmma::load_matrix_sync(b_frag, Vj + k * d, d); the result of Pij @ Vj
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); in shared memory
}
wmma::store_matrix_sync(temp_pv, c_frag, d, wmma::mem_row_major);
__syncthreads();
for (int idx = tx; idx < Br * d; idx += blockDim.x) {
int r = idx / d;
float correction = (mi[r] == -INFINITY || mi_new[r] == -INFINITY)
? 0.0f
: expf(mi[r] - mi_new[r]);
Oi[idx] = correction * Oi[idx] + temp_pv[idx];
}
```

`__syncthreads();` **In-place accumulation: recovers each element's row index, scales the**

**running total by exp(old\_max - new\_max) using the mi and mi\_new values maintained in listing 8.9, and adds the new contribution**

Finally, after the inner loop completes, the fully computed output tile Oi is normalized and written from shared memory back to HBM, as shown in the following listing.

-----

###### 228 Chapter 8 Flash Attention

| Listing | 8.11 Normalization and | write-back to HBM |  |  |
|---|---|---|---|---|
| for (int int if (s_row } } __syncthreads(); With the 8.1 shows sequence Table 8.1 | col = s_col; col < d; global_row = < Br && global_row O[qkv_off + s_row * d + col] / kernel complete, we can timing results on an lengths. Flash Attention benchmark | col += Bc) { row_offset + s_row; < N) { global_row * d + col] = li[s_row]; measure what RTX 3090 with B results on RTX 3090 | Each the final, Oi[ from output all this work = 16, H = 8, and (B = 16, H = 8, D | thread writes its portion of normalized output tile shared memory to the global matrix O in HBM. actually gives us. Table D = 64 across three = 64)* |
| Seq Len | Naive CUDA (ms) | Our FA (ms) | FA vs. naive | PyTorch Flash (ms) |
| 256 512 1,024 | 33.8 69.2 156.5 | 2.18 7.93 32.0 | 15.5x 8.7x 4.9x | 0.12 0.28 0.75 |

\* The naive baseline runs three FP32 CUDA-core kernels per head, while our FA kernel uses FP16 tensor cores, so roughly 2x of the measured speedup comes from the precision drop rather than fusion. The following paragraphs unpack this.PyTorch's production Flash Attention is another order of magnitude ahead; ours is a pedagogical kernel, not a production one.

Our WMMA-based Flash Attention kernel is up to approximately 15× faster than the naive three-kernel implementation (15.5× at sequence length 256, tapering to 4.9× at 1,024). All outputs pass correctness checks against the PyTorch reference (max absolute difference < 0.001), confirming that online softmax is mathematically faithful to the standard computation, with only the expected floating-point rounding differences. These speedups deserve a careful caveat. Our comparison bundles three separate advantages into one number. First, the Flash Attention algorithm itself: tiling and online softmax eliminate the O(N2) HBM round trips for the intermediate score matrix. Second, the FA kernel uses FP16 tensor cores for matrix multiplication, while the naive kernel uses FP32 CUDA cores. Tensor cores are fundamentally faster at matrix math, so part of the speedup comes from the precision difference, not the algorithm. Third, the FA kernel launches once for all 128 heads via a 2D grid, while the naive kernel loops over heads on the host and launches 384 separate kernels (128 heads × 3 kernels each). Kernel launch overhead and serialization account for some of the gap, especially at shorter sequences where per-kernel compute is small.

-----

###### The evolution of Flash Attention 229

A fairer algorithmic comparison would use the same precision and launch strategy for both kernels. We chose not to do that here because the whole point of Flash Attention is that it enables fusion: you can't fuse three separate kernels that each need the full N × N matrix, but you can fuse tiled operations that need only small on-chip buffers. The fusion advantage and the tensor core advantage are part of the story, not confounds to be eliminated.

Our kernel also wastes significant compute. The block has 256 threads (eight warps), but each WMMA operation only requires one warp to produce the 16 × 16 result. All eight warps execute the same WMMA instructions and race to store identical results to the same shared memory location. Seven out of eight warps are doing redundant work. Similarly, the online softmax uses only 16 of 256 threads (one per row of the tile). A production kernel assigns different warps to different output tiles or pipeline stages, which is one reason PyTorch's Flash Attention is roughly 30× faster than ours.

The gap to PyTorch comes from these inefficiencies plus several optimizations we omitted: larger tile sizes (128 × 128 vs. our 16 × 16), warp specialization for overlapping compute with memory loads, asynchronous memory pipelines, and architecturespecific tuning. The next section traces exactly how production kernels evolved from the same starting point we just built.

## 8.4 The evolution of Flash Attention

The Flash Attention kernel we built represents the foundational ideas behind this optimization technique. Since the original Flash Attention paper (Dao, Fu, Ermon, Rudra, and Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," https://arxiv.org/abs/2205.14135), there have been three major iterations, each bringing substantial performance improvements and new optimization techniques.

### 8.4.1 Flash Attention 1 and 2

The original Flash Attention established the core algorithmic foundation we explored in this chapter. The key innovation was recognizing that the memory bottleneck, not computation, was the limiting factor for attention at long sequence lengths. By applying classical techniques from high-performance computing, tiling and recomputation, to the specific structure of the attention mechanism, the algorithm eliminated the need to materialize the massive N × N attention matrices in HBM. Instead, it processed attention in blocks small enough to fit in SRAM, so the intermediate score and probability matrices never touch slow memory. This approach reduced memory usage from O(N²) to O(N) and achieved 2 to 4× speedups over optimized baselines.

Flash Attention 1 parallelized over batch size and the number of attention heads. Each thread block handled one attention head, computing the full output for that head by iterating through tiles. Although this was a breakthrough, it left performance on the

-----

###### 230 Chapter 8 Flash Attention

table. The forward pass reached only 30% to 50% of theoretical maximum throughput, and the backward pass was even slower, at 25% to 35%. In contrast, optimized matrix multiplications routinely achieve 80% to 90% utilization. The bottleneck had shifted from memory bandwidth to suboptimal parallelism and work distribution.

Flash Attention 2 (Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," https://arxiv.org/abs/2307.08691) addressed the performance gap through three focused improvements: algorithmic tweaks to reduce non-matmul operations, better parallelism across sequence lengths, and smarter work partitioning within thread blocks.

The first change targets non-matmul work. Data center GPUs have specialized tensor core units that make matrix multiplication roughly 15× faster than general arithmetic operations (H100: 989.5 dense FP16 tensor TFLOPS versus 67 FP32 TFLOPS; on consumer parts the gap is closer to 2×). Every cycle spent on non-matmul work (computing exponentials, finding maximums, normalizing) is many cycles not spent on matmuls. Flash Attention 2 introduced deferred scaling and logsumexp storage to minimize these operations while maintaining correctness.

The parallelism and work partitioning improvements were more fundamental. Flash Attention 2 added parallelization over the sequence-length dimension. The outer loop over output blocks became embarrassingly parallel, with different thread blocks handling different output tiles independently. This kept the GPU busy even when processing single long sequences. The work partitioning changes targeted shared memory traffic. Flash Attention 1 split K and V across warps using a "split-K" scheme. Each warp computed part of QK^T, wrote intermediate results to shared memory, synchronized, and then combined the results. This created unnecessary shared memory reads and writes. Flash Attention 2 instead split Q across warps while keeping K and V accessible to all warps. After computing its slice of QK^T, each warp independently multiplied it with

```
V to get its output slice. No synchronization was needed. This reduced shared memory
```

traffic and yielded a measurable speedup.

Together, these improvements delivered roughly a 2× speedup over Flash Attention 1, reaching 50% to 73% of theoretical throughput on A100 GPUs. End-to-end training of GPT-style models achieved 225 TFLOPs/s per A100, a 1.3× improvement over Flash Attention 1 and 2.8× over the non-Flash baseline.

Our implementation in this chapter captures the essential ideas from Flash Attention 1 and 2: tiling to fit data in SRAM, online softmax for numerically stable streaming computation, and kernel fusion to eliminate memory bottlenecks. As we saw in table 8.1, our WMMA-based kernel achieves up to approximately 15× speedup over the naive baseline. But production implementations push much further, reaching 75% of theoretical hardware throughput on modern GPUs.

### 8.4.2 Flash Attention 3: Asynchrony and low precision

Flash Attention 3 (Shah, J. et al., "Flash Attention-3: Fast and Accurate Attention with Asynchrony and Low-Precision," https://arxiv.org/abs/2407.08608) targeted Hopper

-----

###### The evolution of Flash Attention 231

(H100), where Flash Attention 2 achieved only 35% utilization despite optimized GEMMs reaching 80% to 90%. The gap came from ignoring Hopper's asynchronous hardware: TMA for overlapped memory copies and WGMMA for nonblocking matrix multiplies.

The fix was warp specialization. Producer warps stream data via TMA into circular buffers while consumer warps run WGMMA on completed tiles, coordinated by lightweight barriers. Intra-warpgroup pipelining hides softmax latency (3.9 TFLOPs/s for special functions) under GEMM execution (989 TFLOPs/s), and FP8 tensor cores double throughput with block quantization to control precision loss.

Flash Attention 3 achieved a 1.5 to 2× speedup over v2, reaching 740 TFLOPs/s (75% utilization) with FP16 and close to 1.2 PFLOPs/s with FP8.

### 8.4.3 Flash Attention 4: Complexity for Blackwell

Flash Attention 4 targets Blackwell with two algorithmic changes and a more complex pipeline. First, software exponentials replace hardware SFUs (only 4 per SM vs. 128+ CUDA cores), using a cubic polynomial approximation via Horner's method that matches SFU precision for BF16 while avoiding the bottleneck. Second, smarter rescaling skips the softmax correction when the running maximum changes by less than a stability threshold, reducing rescaling operations by 10×.

The pipeline expands from two warpgroups to five specialized warp types: Load, MMA, Softmax, Correction, and Epilogue warps, all running concurrently on persistent kernel blocks scheduled by a tile dispatcher. This delivers a reported 20% speedup over NVIDIA's cuDNN attention kernels but at a significant cost in complexity: the kernel is highly architecture-specific and requires substantial rewriting for each GPU generation, pushing framework developers toward abstractions like CUTLASS and CuTe. (Our understanding of Flash Attention 4 is based on the reverse-engineering work by Frye, Wang, and Feng at Modal: https://modal.com/blog/reverse-engineer-flash-attention-4. The Flash Attention implementations are open source: https://github.com/Dao-AILab/ [flash-attention.)](https://github.com/Dao-AILab/flash-attention)

### 8.4.4 What this evolution teaches us

The progression from Flash Attention 1 to 4 reveals several lessons about GPU optimization. First, algorithmic innovations matter most early on. The shift from materializing attention matrices to tiling and online softmax delivered the largest single speedup. Later improvements, while substantial, were incremental.

Second, hardware-software co-design is increasingly important. Flash Attention 3 and 4 are inseparable from Hopper and Blackwell architectures. Their performance gains come from exploiting specific hardware features like TMA, WGMMA, and FP8 tensor cores. A kernel tuned that tightly to one chip can lose its edge on the next architecture, at other shapes, or when fewer SMs are free. Before putting that kind of work into a kernel, consider everywhere it will run, not just the one case you care about.

-----

###### 232 Chapter 8 Flash Attention

Third, there's a fundamental tension between simplicity and performance. Flash Attention 1's implementation, similar to what we built in this chapter, is relatively straightforward. Flash Attention 4's five-warp, multibuffered, asynchronous pipeline is substantially more complex. Each generation traded simplicity for percentage points of throughput. For production systems at scale, those percentage points translate to millions of dollars in hardware costs, making the complexity worthwhile. For research and learning, simpler implementations often suffice. Finally, this evolution shows where GPU programming is heading. Modern kernels are becoming asynchronous, pipelined, and warp specialized. The programming model is shifting from "write a function that a thread executes" to "write a state machine where warps handle different pipeline stages." This shift is powerful but challenging, requiring new abstractions and tools. The CUDA ecosystem is responding with libraries like CUTLASS for tensor core programming and domain-specific languages like Triton for simpler kernel authoring. Production kernels start from the same tiling, online softmax, and tensor core matmuls, then add larger tiles, asynchronous pipelines, and warp specialization.

###### Summary

¡ Standard attention is memory-bound: the N × N score and probability matrices require multiple HBM round trips that dominate execution time at long context lengths. ¡ Online softmax maintains running maxima and normalization sums across tiles, producing mathematically exact results without ever materializing a full row of scores in memory. ¡ Each thread block owns a complete set of output rows and moves sequentially across all K/V tiles. Running statistics stay in registers and shared memory, with no cross-block communication. ¡ The fused Flash Attention kernel combines Q@K.T, softmax, and P@V into a single WMMA-powered kernel. Data stays on chip throughout, reducing HBM traffic from O(N2) to O(N). ¡ On an RTX 3090, the fused kernel achieves up to an approximately 15× speedup over the naive three-kernel baseline. The 30× gap compared with PyTorch's production Flash Attention comes from larger tile sizes, warp specialization, and asynchronous pipelines that the pedagogical kernel omits. ¡ The three-level loop hierarchy nests a standard GEMM "inching inward" loop (over head dimension d) inside a sequential horizontal march (over sequence length N), inside grid-level parallelism across output row tiles. ¡ The naive three-kernel implementation serves as a clear baseline that exposes the memory bottleneck: each kernel reads from and writes to HBM, tripling the data movement.

-----

###### Summary 233

¡ Flash Attention's evolution from v1 to v4 shows a consistent pattern: each generation uses more hardware-specific features (TMA, WGMMA, warp specialization, FP8) for incremental speedups at increasing implementation complexity. ¡ Our WMMA-based implementation captures the core algorithmic ideas. Production kernels add asynchronous pipelines and warp specialization for the final percentage points of throughput.

-----

# 9

## Quantization

---

| This chapter covers |
|---|
| ¡ How FP32, INT8, and INT4 formats trade memory for accuracy |
| ¡ Implementing quantization/dequantization steps |
| and reasoning about error |
| ¡ Choosing granularity schemes that balance |
| hardware cost and fidelity |
| ¡ Calibrating scales and zero points with static or |
| dynamic approaches |
| ¡ Navigating symmetric versus asymmetric choices |
| alongside AWQ refinements |
| ¡ Stitching these ideas into an end-to-end model |
| quantization workflow |

Picture yourself trying to run a modern language model on a single GPU. A 70B-parameter model needs about 140 GB for its weights in FP16, and your GPU has 80 GB. At 4 bits per weight, the same model takes about 35 GB and fits on one 48 GB workstation card. The technique that gets you there is quantization.

-----

###### Quantization building blocks 235

When you compress a model from 16-bit to 4-bit precision, it needs 4× less memory to load and 4× less bandwidth to move data around, and inference runs twice as fast. Your GPU bill gets cut in half. It's the difference between "impossible on this hardware" and "runs smoothly in production."

Most models barely notice. A carefully weight-quantized model often maintains around 99% of its original accuracy on common benchmarks. Think about that tradeoff: 4× smaller, roughly 2× faster in memory-bound decoding, and you lose maybe 1% accuracy. The speed gain is workload dependent (it comes from moving fewer bytes, so bandwidth-bound inference benefits most), but for most real-world applications, this is close to a free lunch.

We're going to build your understanding from the ground up. We'll start by examining what happens when you squeeze a 32-bit number down to 8 bits-what gets lost, what stays, and why it matters less than you'd think. Then we'll discover how different parts of your model need different levels of precision, leading us to strategies for applying quantization selectively. Finally, we'll explore how to measure your data so the quantization process preserves what actually matters. The goal is to develop intuition for when and how to trade precision for speed.

## 9.1 Quantization building blocks

Quantization works because neural networks can tolerate small, structured errors when those errors buy large savings in memory and bandwidth. The building blocks in this section show where those savings come from, how quantization noise appears, and why the exact placement of the integer range matters before we ever write a CUDA kernel.

### 9.1.1 Data types: The foundation of quantization

Let's start with something concrete. Open up any neural network checkpoint file and look at the weights: you'll find arrays of 32-bit floating-point numbers. Each weight is an FP32 value, storing numbers with about 7 decimal digits of precision. This precision is wonderful for training, where tiny gradient updates need to accumulate over millions of iterations. But do we really need all that precision just to multiply activations by weights during inference?

Here's an experiment. Take a trained model, scale its weights into the integer range -127 to 127, and round them. You'd expect that to wreck the model. It doesn't. It still works, not perfectly, but far better than you'd expect.

If you can get away with integers instead of floats, you immediately cut your memory footprint to a quarter of the original size. A model that needed 4 GB now fits in 1 GB.

###### Memory savings that matter

Take a billion parameters, roughly the size of a small language model today. In FP32, you need 4 GB just to store the weights. Switch to INT8, and that drops to 1 GB. Push to INT4, and you're down to 500 MB. Figure 9.1 shows the comparison." 4x and 8x are not orders of magnitude.

-----

###### 236 Chapter 9 Quantization

But the savings cascade through your entire system. Your model loads 4× faster from disk because there's 4× less data to read. The PCIe bus between CPU and GPU becomes 4× more effective because # of

bits

you're transferring a quarter of the data. Most importantly, your GPU's cache hierarchy, from L2 cache down to shared fp32 fp16 int8 int4 memory, effectively becomes 4× larger

**Figure 9.1 Memory usage comparison across**

because each cache line now holds 4× as **data types, showing a 4x reduction from FP32 to** many parameters. **INT8 and an 8x reduction to INT4**

###### Precision loss and quantization noise

Of course, there's no such thing as a free lunch. When you force a continuous range of numbers into a discrete set of buckets, you lose information. The weight that was 0.347621 becomes 0.35, or maybe 0.34, depending on how you round. The difference between the original and quantized value is what we call quantization noise, and understanding this noise is the key to using quantization effectively. The fascinating part is that this noise behaves predictably. Imagine you're measuring people's heights, but you can report only in full inches, with no fractions. If someone is 68.7 inches tall, you round to 69. If they're 68.2 inches tall, you round to 68. The error you introduce is spread between -0.5 and +0.5 inches. Quantization works the same way, except that instead of inches, we're working with the step size between quantization levels. Bigger steps mean bigger errors. If you double your step size, your mean squared error grows four times (2 × 2 = 4). This means even small improvements in step size, which we control through scale factors, significantly reduce quantization error. Figure 9.2 visualizes this concept with a concrete example.

y

x

---

**4 bits = 2^4 = 16 combinations or "bins" we can place numbers into. 16 bins for ALL numbers is not very much, hence high quantization noise.**

**Figure 9.2 Quantization noise visualization showing how discrete bins create a staircase approximation of smooth data**

-----

###### Quantization building blocks 237

The diagram shows how few bins create a coarse staircase approximation of smooth data. With only 16 bins (4 bits), you can see significant gaps between the discrete levels and the actual values. This is quantization noise in action: the difference between where you want to be and where you're forced to land.

The art of quantization is finding the right balance. Too few bits, and the noise overwhelms your signal: your model's predictions become garbage. Too many bits, and you don't get enough memory savings to justify the added complexity. Most of the time, 8 bits hits the sweet spot: enough precision to maintain accuracy and enough compression to enable practical deployments.

Work through a concrete example to see how this plays out. Say we have four weights: [0.1, 0.3, 0.7, 0.9], and we want to squeeze them into just 2 bits each. With 2 bits, we get four possible values: 0, 1, 2, and 3. Our job is to map our continuous range to these four discrete levels.

First, we figure out the scale: our values span from 0.1 to 0.9, a range of0.8. We need to fit this range into four levels (0 through 3), so each level represents 0.8 / 3 ≈ 0.267 in the original scale.

Now for the quantization: 0.1 divided by 0.267 gives 0.37, which rounds to 0. The value 0.3 becomes 1.12, rounding to 1. The value 0.7 becomes 2.62, rounding to 3. And 0.9 becomes 3.37, but we clamp it to 3 because that's our maximum. When we convert back (dequantize), we get [0, 0.267, 0.8, 0.8].

Compare that to our originals [0.1, 0.3, 0.7, 0.9], and you can see the errors: we're off by 0.1 on the first value, pretty close on the second, and off by 0.1 on the last two. The mean squared error works out to about 0.0078. That's the price of at 2 bits. At 8 bits the step would be 0.8 / 255 ≈ 0.003, about 85 times finer.

###### Symmetric and asymmetric range placement

Now we need to talk about a subtle but important choice: how should you map your floating-point numbers to integers? The obvious answer might seem to be "Just use integers from -128 to 127," but the right range depends on the shape of your data.

A simple contrast makes the tradeoff clear. Suppose your weights range from -2.5 to +2.5. Symmetric INT8 is a natural fit: your scale is 2.5 / 127 ≈ 0.0197, zero maps to integer 0, -2.5 maps to -127, and +2.5 maps to +127. You use the signed range efficiently because the data is already centered around zero.

Now switch to activations after a ReLU. If those activations range from 0 to 5 and you still use symmetric INT8, the integers from -128 to -1 never get used. Half your precision is wasted on values that do not exist. Asymmetric quantization fixes that by shifting the usable range to where the data actually lies. In this case, 0 maps to 0, 5 maps to 255, and every integer in between carries useful signal. Figure 9.3 shows the difference.

This ability to move the range around is a major advantage of integer quantization. Figure 9.4 shows why.

-----

###### 238 Chapter 9 Quantization

INT8 ranges Symmetric:

zero-centered range for weights

---

| -128 |  | 0 |  |  | +127 |  |  |
|---|---|---|---|---|---|---|---|
| positive-only scale | activations | 0 can | Asymmetric: shifted positive use the | range for activations full range. |  |  | 255 |
| -- |  |  |  |  |  | ++ |  |
|  |  |  | 0 vs | zero | point |  |  |
| -- |  |  |  |  |  | ++ |  |

0 scale

**Figure 9.4 Integer quantization versus floating-point range placement. Integer quantization can shift the zero point and scale to match the tensor, while floating-point formats keep the same exponent and significand layout for every value.**

With integer quantization, you have complete control over where your zero point sits and how your scale factor maps to the data distribution. The two number lines show the same integer format aimed at two different regions of the real line: slide the zero point and stretch or shrink the scale, and your limited set of integer levels lands exactly where your data lives. Floating point offers no such knob. Its layout is fixed by the format itself: dense near zero and sparse far from it, whether or not that matches your data.

But floating-point formats such as FP8, FP16, and FP4 fix the exponent/significand split inside each value, and they always place zero at exactly zero: there is no additive zero point to slide the way affine integer quantization has. In practice, FP8 and FP4 recipes recover the missing range knob by attaching external scale factors

(per-tensor scaling in Transformer Engine, per-block scales in MXFP4/NVFP4);

what the format itself fixes is the shape of the value spacing, dense near zero and sparse far from it.

-----

###### Quantization building blocks 239

| Exponent vs. significand |
|---|
| Floating-point numbers split their bits among sign, exponent, and significand fields. |
| The exponent sets the rough order of magnitude on a logarithmic scale, while the significand, historically called the mantissa, fills in the fine detail within that range. Because those field widths are fixed by the format, FP8 and FP16 cannot slide their dynamic range around to match your data the way integer quantization can. |

### 9.1.2 Basic quant/dequant operations

With the foundations in place, it's time to **QUANT PHASE:** write some actual CUDA code. The beautifp32

ful thing about quantization is that the core operations are almost trivially simple. Fig- X ure 9.5 shows the pipeline step by step. quantized = round ( )

scale At its heart, quantization is just division,

int8

rounding, and clamping. Your FP32 value is divided by a scale factor, rounded to the nearest integer, and clamped to fit within **DEQUANT PHASE:** your target integer range. Dequantization dequantized = quantized \* scale reverses this process: multiply by the scale factor to get back to floating point. Every- **Figure 9.5 Quantization process showing**

**step-by-step conversion from FP32 to INT8**

thing comes down to choosing that scale factor wisely and applying it at the right level of detail.

###### FP32 → INT8 quantization

The forward direction comes first: taking floating-point values and compressing them into 8-bit integers. The formula fits on one line: quantized = round(input / scale). That's it. Say we have four weights: [1.2, -0.8, 2.5, -1.7]. Before we can quantize them, we need the scale factor. For symmetric quantization, we take the maximum absolute value (2.5 in this case) and divide it by 127: scale = 2.5 / 127 ≈ 0.0197. We divide by 127 because INT8 ranges from -128 to +127, and we want our maximum value to map exactly to +127, not overflow into -128. Now we quantize each value: 1.2 / 0.0197 ≈ 60.9, which rounds to 61. The value -0.8 becomes -40.6, rounding to -41. Our maximum value, 2.5, becomes exactly 127. And -1.7 becomes -86.3, rounding to -86. We add a clamp to ensure nothing escapes the [-127, 127] range, though if we computed our scale correctly, only the maximum value should ever hit the boundary.

The kernel in the following listing is remarkably simple.

-----

###### 240 Chapter 9 Quantization

**Listing 9.1 FP32-to-INT8 quantization kernel**

###### Computes

```
__global__ void quantize_fp32_to_int8(float* input, int8_t* output, the global
float scale, int size) { thread
int idx = blockIdx.x * blockDim.x + threadIdx.x; index
if (idx >= size) return;
```

###### Boundary check to prevent out-of-bounds access

```
float scaled = input[idx] / scale;
scaled = fmaxf(fminf(scaled, 127.0f), -127.0f); Clamps to [-127, 127]
output[idx] = (int8_t)roundf(scaled); to prevent overflow
}
```

###### Rounds and casts to INT8 Divides by the scale factor to map to an integer range

Each thread handles one element independently: classic embarrassingly parallel work. Divide by the scale, clamp to the valid range, round to the nearest integer, and write it out. The clamping step is critical: without it, a value that's slightly too large could overflow and wrap around to a large negative number, catastrophically corrupting your weights.

###### Dequantization: INT8 → FP32

Going back is even simpler. To dequantize, you just multiply by that same scale factor:

```
dequantized = quantized * scale. No rounding is needed because you're converting
```

from integer to float, and the float has enough precision to exactly represent any 8-bit integer. Take our quantized values [61, -41, 127, -86] and a scale of0.0197. Multiply each by the scale: 61 × 0.0197 ≈ 1.20. Then, -41 × 0.0197 ≈ -0.81. Our maximum is 127 × 0.0197 ≈ 2.50, and -86 × 0.0197 ≈ -1.69. Compare these values to our originals [1.2, -0.8, 2.5, -1.7], and you can see the quantization noise: the first value has zero error, the second is off by 0.01, the third has zero error, and the fourth is off by 0.01. These tiny errors are the price we pay for 4× compression. For most neural network computations, these errors are well within the model's noise tolerance.

The dequantization kernel in the following listing is even more stripped down.

**Listing 9.2 INT8 to FP32 dequantization kernel**

```
__global__ void dequantize_int8_to_fp32(int8_t* input, float* output,
float scale, int size) {
int idx = blockIdx.x * blockDim.x + threadIdx.x; Computes the
if (idx >= size) return; global thread index
```

###### Boundary check

```
output[idx] = (float)input[idx] * scale;
```

`}` **Casts to float and multiplies**

**by the scale factor**

-----

###### Quantization building blocks 241

Just a cast and a multiply. Dequantization is so cheap that in many quantized inference systems, you dequantize on the fly as you load weights from memory. In a memorybound kernel, that arithmetic usually hides behind the weight loads.

###### Scale computation

Everything hinges on that scale factor. Choose it too small, and your values overflow the integer range, clamping to the limits and losing information. Choose it too large, and you waste precision because you only ever use a narrow band of the available integer range, effectively quantizing to fewer than 8 bits. That scale of 0.0197 tells us something useful: each step in our quantized space represents about 0.02 in the original floating-point space. Any two values closer than 0.02 will round to the same quantized value. This gives you an intuitive sense of the resolution you're working with.

###### Accuracy validation

You need to validate the quantization quality, and there are two perspectives to consider: error metrics and task performance. Mean squared error (MSE) squares each error before averaging, which means it heavily penalizes outliers. If you quantize 1,000 values perfectly but completely butcher one value, MSE will be dominated by that single bad value. This is useful because those outliers often cause the worst problems in practice. Mean absolute error (MAE) averages the absolute values of the errors. It's more intuitive: it tells you the typical size of the error you should expect on any given value. For our running example with errors [0.0, -0.01, 0.0, -0.01], the MAE is 0.005; on average, we're off by half a hundredth. These error metrics don't tell you everything. What really matters is whether your model still works. You can have a high MSE, but if those errors are in parameters the model doesn't rely on heavily, your accuracy might barely budge. Conversely, even small errors in critical weights can tank performance. Always validate on your actual task: run your quantized model through your test set and measure the metric you actually care about. One useful sanity check is to compare MAE with the square root of MSE (RMSE) on the same batch. If RMSE jumps far above MAE, a small number of large errors are dominating the result, which usually points to clipping or a poorly chosen scale. Here's a trap people fall into constantly: if you compute your scale factor using your calibration data and then measure quantization error on that same calibration data, you're just measuring how well you fit that specific batch. Always validate on separate data. Otherwise, you're committing the same cardinal sin as evaluating a trained model on its training set.

###### INT4 quantization: Pushing further

Everything we've covered so far uses INT8 (8 bits per weight), giving us a 4× memory reduction. But what if we need even more compression? INT4 quantization uses only 4 bits per weight, achieving an 8× memory reduction. The tradeoff is more quantization noise, but for many large models, the memory savings make deployment possible on

-----

###### 242 Chapter 9 Quantization

hardware that would otherwise be unable to load the model. The challenge with INT4 is that it's not a native data type in C. We need to pack two INT4 values into a single byte. This means our quantization and dequantization kernels need bit-manipulation logic to pack and unpack values. The kernel in listing 9.3 packs two INT4 values per byte. Note the bounds check: with two elements per thread, a tensor with an odd number of elements needs (size + 1) / 2 threads to do useful work, so the last thread packs one real value alongside a zero.

| Listing 9.3 Packed INT4 quantization kernel |
|---|
| __global__ void quantize_fp32_to_int4_packed( float* input, uint8_t* output, float scale, int size ) { int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= (size + 1) / 2) return; int elem1_idx = idx * 2; Each thread processes two int elem2_idx = idx * 2 + 1; consecutive elements. float scaled1 = input[elem1_idx] / scale; scaled1 = fmaxf(fminf(scaled1, 7.0f), -7.0f); INT4 range: [-7, 7] int8_t quant1 = (int8_t)roundf(scaled1); (using 4 bits) float scaled2 = (elem2_idx < size) ? input[elem2_idx] / scale : 0.0f; scaled2 = fmaxf(fminf(scaled2, 7.0f), -7.0f); int8_t quant2 = (elem2_idx < size) ? (int8_t)roundf(scaled2) : 0; uint8_t packed = Packs two 4-bit ((quant1 & 0xF) << 4) \| (quant2 & 0xF); values into 1 byte output[idx] = packed; } Dequantization in the following listing unpacks and sign-extends the 4-bit values. |
| Listing 9.4 Packed INT4 dequantization kernel |

```
__global__ void dequantize_int4_to_fp32_packed(
uint8_t* input,
float* output,
float scale,
```

```
int size
) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= (size + 1) / 2) return;
int elem1_idx = idx * 2;
```

-----

###### Deployment strategies

```
int elem2_idx = idx * 2 + 1;
uint8_t packed = input[idx];
int8_t quant1 = (packed >> 4) & 0xF;
int8_t quant2 = packed & 0xF;
if (quant1 & 0x8) quant1 |= 0xF0;
if (quant2 & 0x8) quant2 |= 0xF0;
output[elem1_idx] = (float)quant1 * scale;
if (elem2_idx < size) {
output[elem2_idx] = (float)quant2
}
}
```

###### Extracts the two 4-bit values from the packed byte Sign-extends the 4-bit values to 8-bit signed integers

###### Multiplies by the by scale to recover approximate

`* scale;` **the original values**

INT4 quantization is particularly effective for large language models, where memory capacity is the primary bottleneck. A 70B-parameter model that needs two 80 GB data center GPUs at FP16 drops to about 35 GB of weights at INT4, fitting on a single 48 GB workstation card; a 30B model at INT4 (about 15 GB) fits on a single 24 GB consumer GPU. The accuracy loss is higher than with INT8, but careful calibration and techniques such as group-wise quantization help maintain acceptable performance.

## 9.2 Deployment strategies

Choosing a low-precision format is only the first half of the problem. You also have to decide how broadly each quantization parameter gets reused: one scale for an entire tensor, one per output feature, or one per local block. That reuse pattern determines both the quantization error and the amount of bookkeeping your kernels must carry.

### 9.2.1 Granularity schemes: How to apply quantization

Now we hit the real design question: do you use one scale factor for the entire tensor or different scale factors for different parts? This choice has profound implications for both accuracy and efficiency.

Think about a typical weight matrix in a transformer. Maybe it's 4096 × 4096-about 67 million parameters. You could compute one scale factor by finding the maximum absolute value across all 67 million weights, then quantize everything using that single scale. This is simple and efficient but potentially wasteful. What if one column of the matrix has weights ranging from -2.5 to +2.5, while another column ranges from just -0.1 to +0.1? With a single scale factor tuned for the larger range, you'd be squandering precision on that smaller-range column.

Or you could compute a different scale for each column: 4,096 different scales. Now each column gets quantized according to its own range, maximizing precision everywhere. But you've added 4,096 scale factors to your model, and your kernel needs to look up the correct scale for each column it processes. The code gets more complex, the memory footprint grows, and the kernel might slow down.

-----

###### 244 Chapter 9 Quantization

This is the granularity tradeoff, and the community has explored it thoroughly. In the figures that follow, each shaded region represents the subdivisions usually mean better precision, but they also mean more metadata and more scale lookups during inference. **Tensor-wise** time batch

**quantization**

Start with the simplest approach: one scale factor for everything. You scan through your entire ten- **One quantization parameter**

**for the WHOLE tensor**

sor, find the maximum absolute value, compute your scale, and quantize every element using that **Figure 9.6 Tensor-wise quantization**

**using one scale factor for the entire**

same scale. Figure 9.6 illustrates this approach.

**tensor**

This works beautifully when your data is reasonably uniform. Bias terms in neural networks are a great example: they're typically small values that don't vary much across dimensions. The same applies to small embedding tables or layer normalization parameters. The values might not be identical, but they're all in the same ballpark, so a single scale captures them well. The overhead is minimal: one float32 scale factor, typically 4 bytes. For a weight matrix with millions of parameters, that's completely negligible.And the quantization kernel stays simple-no conditional logic, no looking up which scale to use.Just divide every value by that one scale, quantize, and you're done. The implementation follows the same pattern as listings 9.1 and 9.2 we saw earlier, applying a single scale uniformly across all elements. But the weakness is obvious: if your tensor has widely varying ranges across different regions, you're either wasting precision on the small values or clamping the large values. You can't have it both ways with a single scale.

###### Channel-wise quantization

Now we get more targeted. In many neural networks, different output features have genuinely different scales. Think about the output of a multihead attention layer: each attention head learns to focus on different aspects of the input, and the magnitude of its output can vary significantly. Using one scale for all heads forces you to optimize for the average case, losing precision on both the smallest and largest heads. Channel-wise quantization solves this by using a different scale for each output channel or feature, as shown in figure 9.7. For a weight matrix with 4,096 output features, you compute 4,096 different scales, one per column. When you quantize element (i,j) of the matrix, you use the scale for column j. This adapts the quantization to the local statistics of each feature. Here, "feature" means the output dimension the layer produces. In a dense weight matrix with shape [input\_features, output\_features], each column corresponds to one output feature, so channel-wise quantization usually means one scale per column.

-----

###### Deployment strategies 245

**Batch-wise quantization Time-wise quantization Feature-wise quantization time batch time batch time batch**

**Figure 9.7 Feature-wise (channel-wise) quantization with a different scale for each output feature**

For activation tensors, you often see shapes like [batch, time, feature]: batch-wise means one scale per example in the batch, time-wise means one scale per timestep or token position, and feature-wise means one scale per channel or output dimension shared across the batch and sequence. The figure is meant to show those alternative slicing directions, not a mapping to CUDA thread blocks. The kernel in the following listing indexes into the scale array based on the channel dimension.

**Listing 9.5 Channel-wise quantization kernel**

```
__global__ void quantize_channelwise(
float* input,
int8_t* output,
float* channel_scales,
int num_channels,
int num_elements
) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= num_elements) return;
```

###### Determines which channel

```
int channel_idx = idx % num_channels; this element belongs to
float scale = channel_scales[channel_idx];
```

###### Loads the scale factor

```
float scaled = input[idx] / scale; specific to this channel
scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
output[idx] = (int8_t)roundf(scaled);
}
```

Dequantization in the following listing follows the same pattern.

**Listing 9.6 Channel-wise dequantization kernel**

```
__global__ void dequantize_channelwise(
int8_t* input,
float* output,
float* channel_scales,
```

-----

###### 246 Chapter 9 Quantization

```
int num_channels,
int num_elements
) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= num_elements) return;
int channel_idx = idx % num_channels;
float scale = channel_scales[channel_idx]; Multiplies by the channel-
```

**specific scale to recover**

```
output[idx] = (float)input[idx] * scale; the approximate original
}
```

The cost is clear: you've gone from storing 1 scale factor to storing 4,096 scale factors. For small models, that's negligible, but for very large language models, those scale factors start to add up. And your kernel becomes slightly more complex: instead of loading one scale at the start, you need to load the appropriate scale for each column you process. But the accuracy gains are often worth it. Convolutional networks benefit enormously because different channels learn completely different features, such as edge detectors, color detectors, and texture patterns, with wildly varying magnitudes. Transformer attention and feed-forward layers show similar patterns. If your features have heterogeneous ranges, channel-wise quantization substantially reduces quantization error compared to tensor-wise quantization, and the accuracy gap grows with how much the per-channel ranges differ.

###### Group-wise quantization

Channel-wise quantization raises an interesting question: what if you have so many channels that the scale factor overhead group tion is too coarse? Enter group-wise quantization, which sits right in the middle.

Figure 9.8 shows how grouping balances precision and overhead. Instead of time batch one scale per channel, you group adjacent channels together. Maybe every 64 or 128 **Each group sets a**

**separate quant parameter.**

channels share a scale. For that 4,096-output weight matrix, you'd compute 4,096/128 = **Figure 9.8 Group-wise quantization, with**

**groups of elements sharing scale factors**

32 scales instead of 4,096. You get most of

the benefits of channel-wise quantization

(adapting to local feature statistics) at a fraction of the memory cost.

The kernel in the following listing divides elements into groups and looks up the appropriate scale.

-----

###### Deployment strategies 247

| Listing 9.7 Group-wise quantization kernel |
|---|
| __global__ void quantize_groupwise( float* input, int8_t* output, float* group_scales, int group_size, int num_elements ) { int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= num_elements) return; Computes which group int group_idx = idx / group_size; this element belongs to float scale = group_scales[group_idx]; Loads the scale factor float scaled = input[idx] / scale; for this group scaled = fmaxf(fminf(scaled, 127.0f), -127.0f); output[idx] = (int8_t)roundf(scaled); } Dequantization in the following listing uses the same grouping logic. |
| Listing 9.8 Group-wise dequantization kernel |

```
__global__ void dequantize_groupwise(
int8_t* input,
float* output,
float* group_scales,
```

```
int group_size,
int num_elements
) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= num_elements) return;
int group_idx = idx / group_size;
float scale = group_scales[group_idx]; Multiplies by the
```

**group-specific scale**

```
output[idx] = (float)input[idx] * scale; to recover the value
}
```

This shows up constantly in large language model quantization. When you have weight matrices with tens of thousands of columns, storing a scale per column adds up fast. But grouping 128 columns together keeps the scale overhead to a few percent of the quantized model size (one FP16 scale per 128 INT4 weights is about 3%), while still capturing the variation across features. Group sizes of 64 or 128 have become standard in practice (AWQ and GPTQ both default to 128; QLoRA uses 64)-small enough to adapt to local statistics but large enough to keep overhead minimal.

-----

###### 248 Chapter 9 Quantization

###### Block-wise quantization

The schemes we've discussed so far think in terms of features or channels: they're designed for typical dense layers and attention operations. But what about spatial data like images or feature maps in a convolutional neural network?

| Block-wise quantization divides the spatial sions into blocks, as illustrated in | figure | dimen- 9.9. | Each | time | batch |  |
|---|---|---|---|---|---|---|
| block gets its own scale. Imagine a 32 × map-you might use 4 × 4 blocks, giving ferent scales across the spatial extent. This local spatial statistics: if part of an image is high contrast while another part is dark and trast, each region gets quantized The kernel in the following listing which 2D block each element belongs | 32 you bright low appropriately. to. | feature 64 captures and con- computes | dif- Figure | Each block own quant 9.9 divides spatial blocks. | sets its parameter. Block-wise dimensions | quantization into |
| Listing 9.9 Block-wise quantization | kernel |  |  |  |  |  |
| __global__ void quantize_blockwise( float* input, int8_t* output, float* block_scales, int block_height, int block_width, int tensor_height, int tensor_width, int num_elements ) { int idx = blockIdx.x if (idx >= num_elements) return; | * | blockDim.x index | + Converts the to 2D | threadIdx.x; linear coordinates |  |  |
| int row = idx / tensor_width; |  |  |  |  |  |  |
| int col = idx % tensor_width; |  |  | this | Determines which position falls into | block |  |
| int block_row = row / | block_height; |  |  |  |  |  |
| int block_col = col / (tensor_width + block_width | block_width; - 1) | / | block_width; |  | Computes from | the 2D block coordinates |
| int block_idx = block_row * | blocks_per_row |  | + | block_col; |  |  |
|  |  |  |  |  |  |  |
| float scale = block_scales[block_idx]; |  |  |  |  |  |  |
|  |  |  |  | Loads the scale | for |  |

**this spatial block**

```
float scaled = input[idx] / scale;
scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
output[idx] = (int8_t)roundf(scaled);
}
```

Dequantization in the following listing mirrors the block indexing logic.

-----

###### Deployment strategies 249

**Listing 9.10 Block-wise dequantization kernel**

```
__global__ void dequantize_blockwise(
int8_t* input,
float* output,
float* block_scales,
int block_height,
int block_width,
int tensor_height,
int tensor_width,
int num_elements
) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= num_elements) return;
int row = idx / tensor_width;
int col = idx % tensor_width;
int block_row = row / block_height;
int block_col = col / block_width;
int blocks_per_row =
(tensor_width + block_width - 1) / block_width;
int block_idx = block_row * blocks_per_row + block_col;
float scale = block_scales[block_idx]; Multiplies by the block-specific
output[idx] = (float)input[idx] * scale; scale to recover the value
}
```

The tradeoff is that you're multiplying the number of scales significantly. That 32 × 32 feature map with 256 channels, using 4 × 4 blocks, needs 8 × 8 × 256 = 16,384 scales. That's a lot of overhead, so you reach for block-wise quantization only when spatial variation is genuinely important, typically in computer vision tasks with high-resolution inputs where different regions of the image have dramatically different characteristics. With these tools in hand-tensor-wise, channel-wise, group-wise, and block-wiseyou can adapt your quantization strategy to the structure of your data. But all of them depend on computing good scale factors, which brings us to calibration.

### 9.2.2 Calibration: Computing quantization parameters

Everything we've built so far assumes you have good scale factors. But where do those scales come from? You can't compute them from your test data during inference-that would defeat the entire point of static quantization. You need to precompute them from representative data, a process called calibration. Get this wrong, and even the cleverest quantization scheme will fail. This is also where the earlier feature and channel language becomes concrete. Calibration computes one scale for each slice you decided to reuse during inference: one for the whole tensor in tensor-wise quantization, one per output feature in channel-wise quantization, one per group in group-wise quantization, and so on.

-----

###### 250 Chapter 9 Quantization

The challenge is that your scale needs to work for data the model hasn't seen yet. You're making a prediction: "I think activations in this layer will typically range from X to Y, so I'll set my scale based on that range." If you guess well, your quantization preserves the information that matters. If you guess poorly, you either waste precision or clip important values.

This is where calibration data comes in. You run a few hundred (or thousand) representative examples through your model, observe the actual ranges of activations and weights, and compute scales based on those observations. The key word is "representative"-if your calibration data doesn't match what you'll see in production, your carefully computed scales will be useless.

###### Min-max calibration

The simplest calibration strategy is min-max: find the minimum and maximum values in your calibration data and set your scale to span that entire range. If your activations range from 0.1 to 5.3, set your scale so that 0.1 maps to your minimum quantization level and 5.3 maps to your maximum.

This has an appealing logic to it.You're guaranteeing that every value in your calibration data fits within the quantization range: no clipping, no saturation. If your calibration data is representative and you don't have outliers, this works great.

But here's the problem: outliers. Imagine you have 1,000 activation values and 999 of them range from 0 to 5, but one bizarre outlier is 100. If you use min-max calibration, you set your scale to span 0 to 100. Now your 999 normal values are all squeezed into the bottom 5% of your quantization range: you're effectively throwing away precision because one weird outlier forced you to use a massive scale.

Let's make this concrete. Your calibration data is [0.1, 0.5, 0.8, 100.0]. With UINT8 asymmetric quantization, your scale is scale = (100.0 - 0.1) / 255 ≈ 0.392.

That means each quantization step represents almost 0.4 in the original space. The values 0.1, 0.5, and 0.8 all get quantized to nearly the same integer because they're so close relative to that huge scale. That one outlier has destroyed your precision.

Min-max works when your data is clean. If you've already filtered outliers, or if your model naturally produces well-bounded activations, min-max is fast and simple. But for messy real-world data, you need something more robust.

###### Percentile calibration

Percentile calibration offers a more robust alternative: instead of using the absolute minimum and maximum, it uses percentiles. Set your scale based on, say, the 99th percentile of your data. This means you intentionally clip the most extreme 1% of values but optimize your quantization range for the other 99%.

Those outlier values-the extreme activations that occur rarely-often don't matter much to your model's final predictions. Yes, you're clipping them, but if they're statistical flukes or numerical instabilities, your model has likely learned to be robust to them anyway. By ignoring them during calibration, you concentrate your limited quantization precision on the values that actually occur frequently.

-----

###### Deployment strategies 251

Let's revisit our problematic example, [0.1, 0.5, 0.8, 100.0], but now with a larger dataset. Say you have 1,000 values: 990 of them range from 0 to 5, and 10 are outliers around

100. The 99th percentile is somewhere around 5. Set your scale based on 0 to 5, and now those 990 normal values get the full quantization range. The 10 outliers get clamped to your maximum quantization level, but that's fine; they were probably anomalies anyway. For symmetric quantization, you often use the 95th or 99th percentile of absolute values. This means you're finding a value V such that 95% or 99% of your data falls within [-V, V], then setting your scale based on V rather than the true maximum. The choice of percentile is a knob you can tune: the 99th percentile is aggressive about excluding outliers; the 99.9th percentile is more conservative. The right choice depends on how clean your data is and how much clipping you can tolerate. Percentile calibration does introduce one complication: you need to compute percentiles, which is slower than just finding the max. For static quantization, where calibration happens once during model conversion, this is usually fine, but it's something to be aware of.

###### The effect of calibration quality

Bad calibration can damage your model more than aggressive quantization ever could. If you quantize to 4 bits with perfect calibration, you might lose 2% accuracy. If you quantize to 8 bits with terrible calibration, you might lose 20% accuracy. The scale factors matter that much.

This is why you should always validate calibration on separate test data. Don't measure quantization error on your calibration set; that's circular. Run your quantized model on data it hasn't seen, and measure both the quantization error (MSE/MAE) and the downstream task metric (accuracy, perplexity, or whatever you care about).

If calibration is failing, you have several levers to pull. Collect more calibration samples: maybe your current set isn't representative. Try different percentile thresholds: maybe the 99th percentile is too aggressive and you should use 99.9th percentile. Switch from tensor-wise to channel-wise calibration to capture per-feature statistics. Or manually filter your calibration data to remove known-bad examples.

Calibration is an empirical search for the scale factors that work best for your specific model and data distribution.

With reliable calibration methods in hand, the next question is when to compute those scales: predeployment or at runtime.

### 9.2.3 Static vs. dynamic quantization

With calibration understood, we need to address a critical timing question: when should you compute your scale factors?

Static quantization does all the calibration work up front during model conversion. You run your calibration dataset once, compute your scale factors, bake them into the model, and you're done. From that point forward, every inference uses those exact same scales. It's like meal prepping for the week: you spend time preparing once, then at serving time, everything is fast.

-----

###### 252 Chapter 9 Quantization

Dynamic quantization takes the opposite approach: it computes scale factors fresh for every batch during inference.Your activation ranges might vary from input to input, and dynamic quantization adapts to each one. It's like cooking to order: slower per serving, but each dish is optimized for the specific order.

Figure 9.10 illustrates the divergence in workflows.

###### Offline (once) inference (every batch)

**static** calibration data compute scale activations quantize scale stored, reused every batch

activations absmax -> scale quantize **dynamic** no offline step on the fly

**Figure 9.10 Static vs. dynamic quantization workflows showing precomputed vs. runtime scales**

In both paths, the weights are quantized ahead of time; the figure shows only where the activation scale comes from. Static quantization pays for calibration once, at conversion, and every inference afterward reuses the stored scale and zero point. Dynamic quantization skips calibration entirely and instead computes the absolute maximum over each incoming batch, so its scale adapts to the input at the cost of a runtime reduction on every forward pass.

###### Static quantization

Imagine you are deploying a language model to answer customer support queries. You run a few thousand representative queries through the model, observe the activation ranges in each layer, and compute scale factors. Those scales are serialized into your model checkpoint; they are now as much a part of your model as the weights themselves. When inference time comes, you load the scales along with the quantized weights, and you are ready to go. The metadata in the following listing is small compared with the weight tensor itself.

**Listing 9.11 Static quantization parameter struct**

```
struct StaticQuantParams {
float scales[num_channels];
int zero_points[num_channels];
};
```

-----

###### Deployment strategies 253

This struct might hold 4,096 floats for a large layer, but compared to the millions of quantized weights, that overhead is negligible. And the performance benefit is huge: you never spend inference time computing scales. Your forward pass is pure computation: no statistics gathering, no min-max reductions, no percentile calculations. But static quantization makes a bet: your calibration data was representative, and future inputs will have similar activation ranges. For many deployed applications, such as chatbots, image classifiers, and recommendation systems, this bet pays off. Your input distribution is stable enough that one set of calibration scales works for everything.

###### Dynamic quantization

But what if that bet doesn't pay off? Consider a content moderation system that processes user-generated content. One minute you're analyzing a meme with bright, saturated colors. The next, you're processing a dark, moody photograph. Then, a piece of text rendered as an image. The activation ranges vary wildly depending on the content type. With static quantization, you'd calibrate on average content and hope it works. But "average" content doesn't capture the extremes. Your bright, saturated image might have activations that exceed your calibrated range and get clipped. Your dark photograph might use only a tiny fraction of your quantization range, wasting precision. Dynamic quantization solves this by computing scale factors per batch at runtime. Before you quantize the activations in a layer, you scan through them, find the min and max (or compute percentiles), calculate the scale, and then quantize. Every input gets scales tuned to its actual range. The quantization adapts to the data instead of forcing the data to fit predetermined scales. There's one detail here that's easy to get wrong, and it's worth being careful about. Whatever scale you quantize with, you also need it later to dequantize. A scale you compute but don't store is a scale you can't recover, and without it, your INT8 values are just meaningless integers. That's why dynamic quantization is a two-pass operation: pass one reduces the whole tensor to a single stored absmax value, and pass two quantizes with the scale derived from it. The following listing shows pass one.

**Listing 9.12 Dynamic quantization pass one: Reduce to one stored absmax**

```
__global__ void reduce_absmax(
const float* input,
unsigned int* max_abs_bits,
int size
) {
extern __shared__ float sdata[];
int tid = threadIdx.x; Loads the absolute
int idx = blockIdx.x * blockDim.x + threadIdx.x; value into shared
```

**memory for**

```
sdata[tid] = (idx < size) ? fabsf(input[idx]) : 0.0f; reduction
__syncthreads();
```

-----

###### 254 Chapter 9 Quantization

```
for (int s = blockDim.x / 2; s > 0; s >>=
if (tid < s) {
sdata[tid] = fmaxf(sdata[tid], sdata[tid
}
__syncthreads();
}
if (tid == 0) {
atomicMax(max_abs_bits, __float_as_uint(sdata[0]));
}
}
```

1) { Parallel reduction

**to find this block's**

`+ s]);` **maximum absolute**

**value Folds the block maximum into one global maximum with an atomic max**

There are two things to notice. First is the callout 3 trick: CUDA has no atomicMax for floats, but for nonnegative floats, the raw bit patterns sort in the same order as the values, so an integer atomicMax on the bits gives the exact float maximum. Second, there is no pass-two listing because you already have it: the host reads back the stored maximum, divides by 127 to get the scale, and launches the exact same quantize\_

```
fp32_to_int8 kernel from the previous section with that scale. Static and dynamic
```

quantization share the quantize kernel; the only difference is where the scale came from. One came from calibration data last week; the other came from this batch a microsecond ago. The cost is obvious: you're doing extra computation during the forward pass. Before you can proceed with the actual layer computation, you need to gather statistics over all the activations. On a GPU, this requires a reduction kernel to find the maximum absolute value, which adds latency. The exact overhead depends on tensor size, reuse pattern, and hardware, but it is never free. Dynamic quantization is worth it when your input distribution is genuinely variable and you can't calibrate for the worst case without wasting precision on the common case. Research prototypes often use dynamic quantization because you're still exploring the input space. Applications with truly diverse inputs, such as multimodal models, content moderation, and medical imaging with different scan types benefit from the adaptability. And if your batch size is small (say, batch size 1 for interactive applications), the per-batch overhead matters less because you're not amortizing it over many examples. Match your quantization strategy to your deployment constraints. If you can get away with static, you should, because it's faster. If your accuracy suffers too much from static's rigid scales, dynamic gives you an escape hatch at the cost of performance.

### 9.2.4 Symmetric vs. asymmetric quantization

We touched on this earlier when discussing data types, but now that we understand calibration and granularity, we can see why the symmetric versus asymmetric choice matters so much. It decides how efficiently you use your limited quantization levels. The fundamental question is, where does zero live in your quantization range? Symmetric quantization puts zero right in the middle, using a signed integer type such

-----

###### Deployment strategies 255

as INT8, restricted to -127 to +127. Asymmetric quantization lets zero live anywhere in the range, typically using an unsigned integer type such as UINT8, which goes from 0 to 255. This seemingly small difference has major implications for precision.

###### Symmetric quantization (INT8)

Think about a weight matrix right after initialization. Modern initialization schemes, whether Xavier, Kaiming, or whatever you're using, deliberately center your weights around zero. You get roughly equal amounts of positive and negative weights, all with small magnitudes. This is perfect for symmetric quantization. With symmetric quantization, you compute the maximum absolute value in your data and use that to set the scale. If your weights range from -2.5 to +2.5, your scale is 2.5 / 127 ≈ 0.0197. The value 0.0 maps exactly to the integer 0. Positive values map to positive integers up to +127. Negative values map to negative integers down to -127. Everything is balanced:

```
Data range: [-2.5, +2.5]
Quantized range: [-127, +127]
Scale = 2.5 / 127 ~= 0.0197
Zero point = 0
```

Let's quantize 1.5: divide by 0.0197 to get 76.1, round to 76. Quantize -1.5: divide by 0.0197 to get -76.1, round to -76. Dequantize them: 76 × 0.0197 ≈ 1.50, and -76 × 0.0197 ≈ -1.50. Here is nearly perfect reconstruction. The math is clean, the implementation is simple (no zero-point offset to track), and for zero-centered data, you're using your quantization range efficiently. This is why symmetric quantization dominates for weight quantization: weights are almost always reasonably zero-centered.

###### Asymmetric quantization (UINT8)

But what about activations after a ReLU? ReLU clamps all negative values to zero, so your activation distribution is entirely nonnegative. If you use symmetric quantization with this data, you allocate half your quantization range (integers -127 to -1) to values that never occur. Asymmetric quantization fixes this by introducing a zero-point offset. Instead of forcing zero to map to integer 0, you let it map to any integer in your range. For purely positive data from 0 to 10, you'd use unsigned UINT8 integers from 0 to 255. The value 0.0 maps to integer 0, the value 10.0 maps to integer 255, and every integer in between is used:

```
Data range: [0, 10]
Quantized range: [0, 255]
Scale = 10 / 255 ~= 0.0392
Zero point = 0
```

But asymmetric quantization really shines with skewed data. Say your activations range from -2 to 18: mostly positive but with a bit of negative range. With symmetric

-----

###### 256 Chapter 9 Quantization

quantization, you'd need to set your scale for the maximum absolute value (18), wasting precision on that small negative range. With asymmetric quantization, the full range is 18 minus -2, which is 20, so scale = `20 / 255 ≈ 0.0784 and zero_point = 2 /`

```
0.0784 ≈ 25.5, which rounds to 26.
```

Now the quantization formula includes the offset: quantized = round(value / scale)

```
+ zero_point. To quantize 5: round(5 / 0.0784) + 26 = round(63.8) + 26 = 64 + 26 = 90.
```

The full UINT8 range from 0 to 255 is used proportionally to where your data actually lives. The asymmetric kernel in the following listing adds the integer zero-point offset after scaling.

| Listing 9.13 Asymmetric quantization kernel |  |
|---|---|
| __global__ void quantize_asymmetric( float* input, uint8_t* output, float scale, int zero_point, Adds the zero-point offset after int size scaling into the quantized domain ) { int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return; Clamps the float scaled = input[idx] / scale + (float)zero_point; [0, scaled = fmaxf(fminf(scaled, 255.0f), 0.0f); output[idx] = (uint8_t)roundf(scaled); Rounds and casts to } unsigned 8-bit Dequantization in the following listing subtracts the zero-point before scaling floating point. | to unsigned 255] range an integer back to |
| Listing 9.14 Asymmetric dequantization kernel |  |
| __global__ void dequantize_asymmetric( uint8_t* input, float* output, float scale, int zero_point, int size Subtracts the ) { and then int idx = blockIdx.x * blockDim.x + threadIdx.x; the scale to if (idx >= size) return; approximate | zero-point multiplies by recover the original |
| output[idx] = ((float)input[idx] - (float)zero_point) * scale; |  |

```
}
```

The cost is complexity. Your quantization and dequantization kernels need an extra addition/subtraction operation, and you need to store and load the `zero_point`

-----

###### Advanced techniques 257

parameter. For many applications, though, the precision gain is worth it, especially for activation quantization, where ReLU creates those asymmetric distributions. In practice, frameworks differ here. PyTorch's eager-mode recipe pairs symmetric per-channel INT8 weights, which are naturally zero-centered, with affine per-tensor activations, which are often skewed positive, while TensorRT requires symmetric quantization for both weights and activations. The common thread: symmetric where it works well, asymmetric where the extra zero point buys real precision.

## 9.3 Advanced techniques

The core deployment choices are now on the table: data type, granularity, calibration, and runtime strategy. The techniques in this section build on those foundations rather than replacing them. They are what you reach for when the standard pipeline still leaves too much accuracy on the floor.

### 9.3.1 Activation-aware weight quantization

All the techniques we've covered so far treat quantization as a compression problem: you have weights, you want to make them smaller, and you try to minimize the error. Activation-aware weight quantization (AWQ) takes a completely different angle. Instead of asking, "How do I quantize these weights better?" it asks, "How can I transform my weights to make them easier to quantize?" In a neural network, not all weights matter equally. A weight that gets multiplied by huge activations has a massive effect on the output. A weight that gets multiplied by tiny activations barely matters. Traditional quantization ignores this; it quantizes both weights with the same precision. AWQ protects the important weights by transforming the network, as shown in figure 9.11. The trick: If a weight w gets multiplied by activation a, you can pick a number s, replace the weight with w \* s, and divide the activation by s. The product is untouched:

```
(w * s) * (a / s) = w * a. The layer computes exactly what it computed before. All you've
```

done is move scale from one side of the multiplication to the other. The question is which direction to move it, and AWQ's answer is to scale the important weights up. That sounds backward at first: shouldn't you shrink the thing you want to protect? But remember how quantization error works. Every weight in a group gets rounded with the same step size, set by the largest weight in the group. Rounding costs each weight up to half a step, no matter how big or small the weight is. So a weight's rounding error is fixed; what varies is how much that error gets amplified when it hits the activation. Scale the weight up by s and the activation down by s, and the rounding error now gets multiplied by an activation that's s times smaller. The error that reaches the output shrinks by s. That's the whole mechanism.

### 9.3.2 Simple example: Big vs. small inputs

Make this concrete. Say a quantization group holds three weights: [1.1, 0.8, 2.0]. The largest is 2.0, so the shared INT8 step size is 2.0 / 127 ≈ 0.0157, and every weight in

-----

###### 258 Chapter 9 Quantization

Activation magnitude per channel (calibration data)

/ s

###### Largest activations mark salient channels (~1%).

---

x s

###### Salient column scaled up before quantization

Weight matrix (columns = input channels)

quantize (INT4)

1/s folds into the previous layer: no runtime cost

**Figure 9.11 AWQ rescales weights and activations so the most important products become easier to quantize accurately.**

the group gets rounded to a multiple of that. The weight 1.1 is the important one: its activation channel runs around 25.0, so its rounding error (up to half a step, about 0.008) reaches the output multiplied by 25, which is roughly 0.2 of output error. The 0.8 weight sees activations around 0.1, so its identical rounding error barely registers.

Now apply AWQ with s = 1.8 on the important channel. The weight becomes 1.1 × 1.8 = 1.98, and its activations become 25.0 / 1.8 ≈ 13.9. Look at what happened to the group: the largest weight is still 2.0, so the step size didn't move, and the rounding error is still about 0.008, but it now gets multiplied by 13.9 instead of 25.0. The error reaching the output just dropped by 1.8×, and nothing else in the group got worse.

This example also shows why s stays moderate. Push it to 3.0 and the scaled weight becomes 3.3, overtaking the old maximum of 2.0. Now the step size grows to 3.3/127, and every weight in the group rounds more coarsely: you've protected one weight by taxing all its neighbors. The AWQ paper (Lin et al., "AWQ: Activation-aware

-----

###### Putting it all together 259

Weight Quantization for LLM Compression and Acceleration," [https://arxiv.org/](https://arxiv.org/abs/2306.00978) [abs/2306.00978) searches for the sweet spot per layer, and it typically lands near s =](https://arxiv.org/abs/2306.00978)

```
(average activation magnitude)^0.5: big enough to help, small enough not to blow the
```

range. One practical constraint shapes the implementation: s is chosen per input channel, not per individual weight. Every weight in a column shares one scale, because that's the only granularity at which the compensating division can be folded away: the 1/s gets absorbed into the previous layer's output (or the layer norm parameters), so nothing extra happens at inference time. The following listing shows the conversion-time transformation.

**Listing 9.15 AWQ per-channel rescaling at conversion time**

```
float s = powf(avg_abs_activation[ch], 0.5f); One scale per input channel,
for (int row = 0; row < num_rows; row++) { from calibration-data
```

`weight[row][ch] *= s;` **activation statistics**

```
}
```

`prev_layer_scale[ch] /= s;` **Salient channels (large activations)**

**get scaled up before quantization. Folds the compensating division into the previous layer, so inference sees no extra work**

AWQ performs this transformation once during model conversion by analyzing activation statistics from calibration data. You don't need to gather statistics at runtime, and the quantized model runs exactly like any other weight-quantized model: the protection is baked into the weights before the rounding ever happens.

## 9.4 Putting it all together

A practical quantization workflow combines choices that balance accuracy, performance, and complexity.

### 9.4.1 Real model quantization workflow

Start with granularity. For transformer attention layers, different heads often have different output scales, so channel-wise quantization makes sense: compute one scale per head. For massive feed-forward networks, you might have 4096 × 4096 weight matrices, where channel-wise quantization means 4,096 scales, adding noticeable overhead. Here, group-wise quantization with groups of 128 gives you 32 scales per row: enough granularity to capture variation, with overhead small enough to be practical. Table 9.1 summarizes the granularity that fits each kind of tensor. Next, calibration. You need representative data that spans the distribution your model will see in production. Gather about 1,000 diverse examples, run them through your model, and observe the activation ranges. Use percentile-based calibration, perhaps the 99.9th percentile, for robustness against outliers. Those few weird examples with enormous activations shouldn't dictate your scales for everyone else.

-----

| 260 Table 9.1 | Chapter 9 Which quantization granularity fits each kind of tensor | Quantization Which quantization granularity fits each kind of tensor |
|---|---|---|
| Structure Typical scheme Why it fits |  |  |
| Transformer attention | Channel-wise or | Different heads often settle at different magni- |
| outputs | head-wise | tudes, so one scale per output slice preserves detail. |
| Transformer feed-forward | Group-wise | Large matrices benefit from local adaptation, but |
| weights |  | one scale per column creates too much metadata. |
| CNN feature maps | Block-wise | Spatial regions can vary in contrast and texture, so local blocks capture that variation better. |
| Layer norms, biases, and | Tensor-wise | The tensors are small, stable, and cheap enough |
| other small tensors |  | that one global scale is often sufficient. |

---

Now choose symmetric versus asymmetric. Your weights came from training with proper initialization, so they're centered around zero. Use symmetric INT8 quantization. But your activations flow through ReLU layers, creating positive-only distributions. Use asymmetric UINT8 for those. PyTorch's eager-mode recipe uses this pairing, while TensorRT quantizes both symmetrically.

For deployment mode, static quantization is your default. Compute your scales during the model conversion process and bake them into the checkpoint. Reach for dynamic quantization only if you discover that your input distribution varies so wildly that static scales hurt accuracy. If you're doing weight-only quantization (keeping activations in FP32 or FP16 but quantizing weights to save memory), consider AWQ. The weight transformation happens once during model conversion, and it can significantly improve quantization quality without adding runtime cost. Finally, implement your pipeline. Start with the basic quantize and dequantize kernels we developed earlier. Add support for your chosen granularity, loading the right scale for each channel or group. Integrate your calibration code to compute scales from your calibration dataset. Wire it all together into a model conversion tool that takes an FP32 checkpoint and produces a quantized version. For that typical transformer, you'd end up with something like this: attention layers using channel-wise symmetric static quantization, feed-forward networks using group-wise symmetric static quantization, layer norms using tensor-wise quantization (they're small and stable), and embeddings using channel-wise asymmetric quantization if they're skewed. Calibrate on 1,000 diverse samples with percentile-based ranges. Against the FP32 checkpoint, INT8 weights take 4× less memory. The speedup and accuracy loss depend on the model, so measure both as the next section describes.

### 9.4.2 Accuracy validation strategy

You've made all these choices-granularity, calibration method, symmetric versus asymmetric. Now you need to know whether they actually work. This requires comprehensive evaluation across multiple dimensions.

-----

###### Putting it all together 261

First, measure on data your model hasn't seen. Set aside a proper validation set that represents real production inputs. This is how you catch overfitting to your calibration samples. If your quantized model performs well on calibration data but poorly on validation data, your calibration wasn't representative enough. Track both error metrics and task performance. MSE tells you the average quantization error across all your parameters, which is useful for debugging, but what you really care about is downstream performance. For a language model, measure perplexity or next-token prediction accuracy. For an image classifier, measure top-1 accuracy. For object detection, measure mean average precision (mAP). These task-specific metrics tell you whether your users will notice the quantization. Always compare against your FP32 baseline.You need to know the exact cost of quantization. If your FP32 model gets 85.2% accuracy and your quantized model gets 83.1%, you've lost 2.1 percentage points. Is that acceptable for your use case? Only you can answer that, but you need the number. Validate across different data distributions. Test on typical in-domain data where your model should excel. Test on out-of-domain data to see how robust the quantization is to distribution shifts. Test on edge cases: the weird, rare inputs that might expose quantization problems. A model might handle typical inputs fine but completely fail on outliers. Finally, profile end-to-end performance. The whole point of quantization is to make inference faster and cheaper. Measure memory usage: did you actually get that 4× reduction? Measure latency: is each inference really 2× faster? Measure throughput: can you handle 2× more requests per second? Sometimes the memory savings don't translate to speed improvements if you're compute-bound, or the quantization overhead eats into your gains. A good validation report looks like the following:

```
FP32 baseline: 85.2% accuracy, 1000ms latency, 8GB memory
INT8 quantized: 83.1% accuracy (-2.1%), 450ms latency (2.2x faster), 2GB
memory (4x smaller)
```

Now you can make an informed decision: are 2.2× speed and 4× memory savings worth a 2.1% accuracy loss? But you need these numbers to make the call.

### 9.4.3 Common pitfalls

The most common failure mode is poor calibration. You can implement perfect kernels, choose the right granularity, and use all the advanced techniques, but if your calibration data doesn't represent real usage patterns, everything falls apart. Spend time gathering representative calibration samples; it's more important than almost anything else. Another trap is assuming quantization is free. Sometimes, the overhead of dynamic quantization or complex granularity schemes eats up your performance gains. You might save 4× on memory but see only a 1.2× speedup because you're spending so much time computing scales or looking up per-channel parameters. Profile everything to make sure you're actually getting the benefits you expect.

-----

###### 262 Chapter 9 Quantization

And finally, don't overengineer from the start. Begin with simple tensor-wise, minmax, static quantization. Measure the results. If accuracy is good enough, you're done. If not, identify the specific problem (outliers? varying feature scales? distribution shift?) and apply the appropriate advanced technique. AWQ and group-wise quantization are powerful but add complexity, so reach for them only when simpler approaches fail.

### 9.4.4 The tradeoffs in one place

Before moving on, it's worth collecting every choice this chapter introduced into one picture, because in practice you make all of them at once and they interact. The first choice is bit width, and it's really a memory budget question. INT8 buys you 4× compression and, with good calibration, near-zero accuracy loss: it's the safe default. INT4 buys 8× compression, but the noise is no longer negligible, so it almost never travels alone: it brings group-wise scales and usually a protection scheme like AWQ or GPTQ along with it. If INT8 fits your hardware and your latency target, stop there. The second choice is where zero lives. Symmetric quantization is simpler: no zeropoint to store, no offset arithmetic in the kernel, and it's the right call for weights, which training keeps centered around zero. Asymmetric quantization costs you that extra offset but stops you from wasting half the integer range on values that never occur, which is exactly the situation ReLU activations put you in. The tradeoff is a little kernel complexity for a lot of recovered precision, it's why PyTorch's eager-mode recipe quantizes weights symmetrically and activations asymmetrically. The third choice is how many scales to keep. One scale per tensor is nearly free but lets a single outlier ruin precision for millions of well-behaved values. One scale per channel, or per group of128, contains the damage locally, at the cost of storing and loading more metadata. The finer you slice, the better each slice fits its data, and the more overhead you carry: granularity is the dial that trades accuracy against bookkeeping. The fourth choice is when scales get computed. Static quantization computes them once from calibration data and pays nothing at inference time, but it's a bet that tomorrow's inputs will look like the calibration set. Dynamic quantization recomputes the scale per batch-a reduction pass before every quantized layer-and in exchange, it never gets caught by a distribution it hasn't seen. You're trading latency for robustness, and the honest way to decide is to measure static quantization's accuracy on your worst-case inputs first. The last choice is whether to spend effort protecting the weights that matter most. Plain rounding treats every weight as equally important; AWQ uses calibration activations to find the weights whose errors get amplified the most and shifts scale onto them before rounding, all at conversion time, with zero inference cost. The cost is pipeline complexity and the need for decent calibration data. None of these choices is exotic on its own. The skill is noticing which failure you're actually seeing: clipped outliers point to granularity or calibration, wasted range points to asymmetric quantization, distribution shift points to dynamic quantization, and a few

-----

###### Where we can go further 263

damaged salient weights point to AWQ. Reach for the one dial that fixes the problem instead of turning all of them.

## 9.5 Where we can go further

The techniques we've covered, from basic quant/dequant kernels to calibration strategies and granularity schemes, give you a solid foundation for practical quantization. But the field continues to evolve, and more advanced approaches build on these fundamentals. Here's what lies beyond, even if we won't implement these approaches here.

### 9.5.1 GPT quantization

GPTQ (Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," arXiv:2210.17323) is a one-shot posttraining quantization method designed specifically for large generative models like Qwen3-235B. We will not implement GPTQ in this book, but it is worth recognizing because it extends the same scale, granularity, and error-management ideas to create a stronger posttraining method. The core idea is error compensation. GPTQ processes each row of a weight matrix independently. When you quantize one weight and introduce error, GPTQ adjusts the remaining unquantized weights in that row to compensate, so the row's output stays close to the original. What makes it fast is giving up the careful pick-the-next-weight ordering of earlier methods (the fixed shared order turns out to cost little accuracy at scale) and batchprocessing 128 columns at once to keep the GPU busy. The weights are stored as INT4 (4 bits), but the clever part is that they're converted back to FP16 on the fly during inference in the same kernel that uses them. You get 4× memory savings because INT4 weights stay compressed until the exact moment you multiply them.

### 9.5.2 NormalFloat4

While GPTQ focuses on compensating for quantization errors, NormalFloat4 (NF4) takes a completely different approach: designing a quantization scheme specifically optimized for how neural network weights are actually distributed. We include it here because it is one of the most common 4-bit formats you will encounter in practice, especially through the bitsandbytes and Quantized Low-Rank Adaptation (QLoRA) ecosystems. Neural network weights aren't uniformly distributed; they follow roughly a normal (Gaussian) distribution centered around zero. Most weights cluster near zero, with fewer weights at the extremes. Traditional uniform quantization wastes precious quantization levels on ranges where few weights actually live. NF4 fixes this by deriving its quantization levels from the quantiles of a standard normal distribution. The QLoRA authors describe the result as information-theoretically optimal for normally distributed weights; later analysis showed it is not exactly optimal, but it is close, and it works well in practice.

-----

###### 264 Chapter 9 Quantization

Here's how it works: NF4 uses 16 fixed quantization levels (4 bits), but unlike uniform quantization, where levels are evenly spaced, NF4's levels cluster densely around zero, where most weights live, with sparser coverage at the extremes.

The quantization process operates block-wise (typically 64 weights per block). For each block, find the maximum absolute value (absmax), normalize all weights by this value to the range [-1, 1], and then map each normalized weight to its closest NF4 level. Store the 4-bit index and the absmax scale factor. Dequantization reverses this process: look up the NF4 level from the 4-bit index, multiply it by the stored absmax, and you recover the approximate original weight.

What makes NF4 particularly powerful is double quantization: the absmax scale factors themselves are quantized to save even more memory. Because you have one scale per block, and typical models have millions of blocks, those scales add up. Quantizing them (typically to 8-bit) provides additional compression with minimal accuracy loss.

NF4 shines in QLoRA, where the entire pre-trained base model is quantized to NF4 and frozen, while small trainable LoRA adapters stay in bfloat16. This lets you finetune 65B-parameter models on a single 48 GB GPU, or 70B-parameter models on dual consumer GPUs, something impossible with full-precision weights. The base model's 4-bit weights are dequantized to bfloat16 just in time during computation, but they stay compressed in memory.

The results are impressive: QLoRA reports that 4-bit NF4 fine-tuning matches full 16-bit fine-tuning task performance, while using 4× less memory than 16-bit and 8× less than 32-bit.

### 9.5.3 Quantization-aware training

All the techniques we've covered are posttraining quantization-you train your model normally, then quantize afterwards. Quantization-aware training (QAT) takes the opposite approach: include quantization effects during training itself.

The idea is to simulate quantization during forward passes. Your weights are stored in full precision, but during the forward pass, you quantize them, use the quantized values for computation, and then backpropagate gradients as if the rounding didn't happen. The model learns which weight values survive quantization well and adjusts during training so that it is robust to quantization noise from the start.

QAT can achieve remarkable results: models that maintain accuracy even at 4-bit precision or lower. But there's a cost: you need to retrain (or at least fine-tune) your entire model, which is expensive. You also need to carefully tune hyperparameters to balance quantization effects with normal training. For production models where you have the compute budget and accuracy is paramount, QAT is worth considering. But for most use cases, post-training quantization with good calibration gets you 90% of the way there at a fraction of the cost.

### 9.5.4 Quantization kernel profiling

The profiling techniques from earlier chapters apply directly here, but there are quantization-specific patterns to watch for.

-----

###### Exercises 265

Start with memory bandwidth. Quantization kernels are often memory-bound: you're loading quantized weights (1 byte per parameter), loading scales (4 bytes per channel or group), dequantizing, and computing. Measure your HBM utilization with Nsight Compute. If you're not saturating memory bandwidth, you might be spending too much time on dequantization arithmetic or scale lookups. Check compute efficiency. Are your streaming multiprocessors (SMs) sitting idle? Is your occupancy low? Quantization kernels are typically simple enough that you should be able to keep the SMs busy. Low occupancy might indicate you're launching too few threads or your memory access patterns are causing stalls. Measure end-to-end latency, not just individual kernels. It's easy to optimize one kernel in isolation only to discover that it's not your bottleneck. Profile the complete forward pass: quantized matrix multiplications, dequantization steps, and calibration overhead (if you're using dynamic quantization). Use CUDA events for precise timing of each component. The techniques from chapters 3 to 7-shared memory tiling, warp-level primitives, tensor core utilization-all apply to quantization kernels. Start with Nsight Systems to identify bottlenecks at a high level, then drill down with Nsight Compute to understand the performance characteristics of individual kernels.

### 9.5.5 The road ahead

Quantization research moves fast. Techniques like GPTQ and QAT push what's possible, mixed-precision strategies balance accuracy and performance, and hardware continues to evolve with better support for low-precision arithmetic. But everything builds on the fundamentals we've covered: understanding data types, implementing quant/ dequant operations, choosing granularity wisely, and calibrating carefully.

## 9.6 Exercises

**1** Implement a complete FP32-to-INT8 quantization and dequantization pipeline. Use the kernels we developed to quantize a tensor of 1,000 random values in the range [-5, 5], then dequantize it and compute the MSE. Compare max-based and 95th-percentile scale computation. **2** Compare tensor-wise, channel-wise, and group-wise quantization on a synthetic 1024 × 1024 weight matrix. Measure MSE for each approach and discuss the accuracy-versus-overhead tradeoff. Use group sizes of 64 and 128. **3** Test min-max versus percentile calibration on data with outliers. Create a dataset of 1,000 values with a few extreme outliers 100× larger than the typical values. Compare how each calibration method handles the outliers and affects quantization accuracy. **4** Implement both static and dynamic quantization for a batch of activations. Compare their accuracy and runtime cost across different batch sizes. At what point does dynamic quantization become worthwhile for your workload?

-----

###### 266 Chapter 9 Quantization

**5** Quantize the same activation data using symmetric INT8 and asymmetric UINT8 approaches. Compare the resulting MSE and interpret when asymmetric quantization provides the largest benefit. **6** Implement the basic AWQ transformation on a small weight matrix. Apply different activation magnitudes and observe how the transformation affects quantization difficulty. Compare MSE before and after AWQ. **7** Build a complete quantization pipeline that combines granularity selection, calibration, and static quantization. Apply it to a small transformer-like model with synthetic weights and measure the end-to-end tradeoff between accuracy and performance.

###### Summary

¡ Data type fundamentals establish the foundation of quantization, with INT8

providing 4× memory reduction and INT4 enabling 8× compression through reduced precision representation of neural network parameters. ¡ Basic quantization operations convert between floating-point and integer representations using simple scaling formulas: FP32-to-INT8 quantization applies division by a scale factor, and INT8-to-FP32 dequantization uses multiplication by the same scale. ¡ Granularity schemes determine quantization parameter reuse across tensor

elements, from tensor-wise schemes that use single scales for entire tensors to channel-wise schemes that apply different scales per feature dimension to optimize accuracy versus overhead tradeoffs. ¡ Calibration techniques compute quantization parameters from representative

data samples: min-max calibration uses full data ranges, while percentile calibration provides robustness against outliers through statistical thresholding. ¡ Static quantization precomputes scale and zero-point parameters during model

conversion for maximum inference performance, while dynamic quantization calculates these parameters at runtime for each batch to adapt to varying data distributions. ¡ Symmetric quantization assumes zero-centered data with zero\_point = 0 for

simpler implementation using INT8 ranges, whereas asymmetric quantization handles skewed distributions with nonzero zero\_points using UINT8 ranges for better use of precision. ¡ AWQ optimizes weight matrices for quantization by scaling up the weights of

input channels with large activations, measured on calibration data, and folding the inverse scale into the previous layer. ¡ Real-world quantization workflows combine multiple techniques strategically,

selecting granularity based on model architecture, calibration methods for

-----

###### Summary 267

robustness, and symmetric/asymmetric approaches that match data-distribution characteristics. ¡ Accuracy validation requires comprehensive evaluation using test sets separate

from the calibration data, measuring both mathematical error metrics like MSE/ MAE and downstream task performance to ensure quantization preserves model capabilities. ¡ Quantization enables 2× inference throughput improvements and memory

reductions that make larger models practical on constrained hardware while maintaining acceptable accuracy levels. ¡ Advanced quantization research continues pushing boundaries through novel

calibration techniques, mixed-precision strategies, and hardware-specific optimizations that further reduce computational requirements for AI inference.

-----

~~*1 0*~~

## Distributed computing

---

| This chapter covers |
|---|
| ¡ Mapping multi-GPU interconnects and understanding NCCL collectives |
| ¡ Implementing tensor parallelism across |
| 8 to 16 GPUs |
| ¡ Building pipeline parallel schedules that keep |
| every stage busy |
| ¡ Coordinating data, tensor, and pipeline |
| parallelism within one node |
| ¡ Evaluating scaling behavior and communication |
| tradeoffs |
| ¡ Choosing the right parallelism mix for real |
| workloads |

GPUs in one machine are not all connected the same way, and the links between them set how fast your distributed code can run. The two main ways GPUs communicate inside a server are over the PCIe bus and NVLink:

-----

###### Understanding the hardware 269

¡ *PCIe (Peripheral Component Interconnect Express)-This is* the standard bus that connects everything on a motherboard (your GPU, network cards, and storage drives) to the CPU. It's a general-purpose interconnect. ¡ *NVLink-This proprietary, point-to-point interconnect was developed by NVIDIA* specifically for connecting GPUs to each other and to the CPU. It offers significantly higher bandwidth and lower latency than PCIe. NVLink 4 gives each H100 900 GB/s of GPU-to-GPU bandwidth against the 128 GB/s of its PCIe Gen 5 interface, 7× the raw bandwidth, and measured collectives stretch the gap further: NCCL AllReduce manages roughly 8 to 12 GB/s on a PCIe-only node versus about 475 GB/s of bus bandwidth on an 8-GPU NVLink ode. The first thing to do on a new multi-GPU machine is map out these links. Figure 10.1 gives a hardware-level view of the PCIe and NVLink paths that dominate single-node GPU communication.

Server chassis

CPU

PCIe bus (shared, lower bandwidth)

GPU 0 GPU 1

GPU 2 GPU 3

NVLink (high bandwidth, point-to-point)

###### Data can travel directly between GPUs over the NVLink mesh or take a slower, indirect route through the CPU over the shared PCIe bus.

**Figure 10.1 Physical interconnects inside a multi-GPU server**

## 10.1 Understanding the hardware

Every distributed strategy depends on the physical links between GPUs. This section walks through the tools and concepts you need to map your hardware, understand the

-----

###### 270 Chapter 10 Distributed computing

communication library that sits on top of it, and choose the right algorithm for each message size.

### 10.1.1 Lab 1: Mapping your hardware

Before we write distributed code, we should inspect the topology of the machine we are running on. NVIDIA's nvidia-smi command can generate that map directly. Open your terminal and run nvidia-smi topo -m, which queries the physical connections between each GPU and the CPU:

```
nvidia-smi topo -m
```

The output shows a matrix in which each cell indicates the connection type between a pair of GPUs. The key codes are as follows: NV# means a bonded set of # NVLinks (NV18 on an H100 SXM node means 18 links and 900 GB/s bidirectional; NV12 on A100, 600 GB/s); PIX means a path through a single PCIe switch; PHB means a path through a PCIe host bridge (typically the CPU); and SYS means a path that also crosses the interconnect between CPU sockets. Each step down that list costs bandwidth and latency. For multi-GPU work, you want NVLink connections between every pair. On a typical 8-GPU H100 SXM server, GPUs within the same node are fully connected via NVLink 4.0 through NVSwitch. When you run collective operations like AllReduce, NCCL (NVIDIA Collective Communications Library, pronounced "nickel") automatically routes data over these fast paths.

###### 10.1.2 Understanding NCCL's primitives

Before we write distributed code, we need to understand the communication library that makes it all possible. NCCL is to multi-GPU programming what cuBLAS is to matrix multiplication: a highly optimized library that handles the hard parts for you. NCCL provides collective operations that work across multiple GPUs; they are called "collective" because all GPUs participate simultaneously. NCCL is topology aware. When you call a collective operation, NCCL looks at your hardware topology (which GPUs have NVLink and which don't) and chooses the fastest route. On a system with NVLink, NCCL will use ring or tree algorithms that maximize NVLink bandwidth. On systems without NVLink, it falls back to PCIe. You can benchmark NCCL performance using the nccl-tests suite. On an 8-GPU H100 system with NVLink, you should see aggregate AllReduce bus bandwidth (busbw) of roughly 450 to 480 GB/s for large messages, corresponding to about 270 GB/s of algorithm bandwidth. That is the ceiling for large messages on that node. Small messages land far below it because per-step latency dominates.

NCCL provides collective operations, including AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, AllToAll, Gather, and Scatter; we focus on the ones distributed training leans on most heavily. Each serves a specific communication pattern that appears repeatedly in distributed deep learning. We'll cover the theory of each operation here; the actual implementations are handled by NCCL's highly optimized kernels.

-----

###### Understanding the hardware 271

###### AllReduce

AllReduce is the workhorse of distributed training. The name tells you exactly what it does: it reduces (combines) data from all GPUs and then gives the result to all GPUs.

Here's the concrete scenario: you have four GPUs, and each GPU has computed a partial result. Maybe GPU 0 computed gradients from its batch of training data and got the value 1. GPU 1 got 2, GPU 2 got 3, and GPU 3 got 4. You want to combine these partial results by summing them (1 + 2 + 3 + 4 = 10) and then give that final result (10) to every GPU. That's exactly what AllReduce does.

The "reduce" part means we apply an operation to combine the values. The most common operation is sum, but you can also use max, min, or product. The "all" part means every GPU gets the final result. This is different from a regular reduce operation, where only one GPU gets the answer.

In data-parallel training, each GPU processes a different batch of data and computes gradients. But you want all GPUs to update their weights using the average gradient across all batches. AllReduce with sum gives you the total gradient, which you then divide by the number of GPUs to get the average. Each GPU performs this AllReduce after its backward pass, and then each GPU has the same averaged gradients to update its local copy of the model.

In tensor parallelism, we saw this in action: each GPU computed a partial matrix multiplication result, and AllReduce summed the results to give every GPU the complete output matrix. In distributed batch normalization, you'd use AllReduce to compute the mean and variance across all GPUs' data, not just one GPU's batch.

Two costs matter, and it helps to keep them separate. The first is bytes: with NCCL's ring algorithm, each GPU sends and receives about 2N bytes in total (2(P-1)/P × N, to be exact), where N is the size of the data being reduced. That bandwidth term is O(N) and barely changes as you add GPUs. The second is steps: the ring needs 2(P-1) sequential steps for P GPUs, so the latency term grows with P. For a model with 1 billion parameters, that's 4 GB of data at FP32 precision, and on an 8-GPU NVLink system the whole operation takes about 16 ms (measured with nccl-tests on an 8× H100 SXM node: 15.8 ms, a bus bandwidth of476 GB/s).

###### Broadcast

Broadcast is simpler than AllReduce. One GPU has some data, and you want to copy that exact data to every other GPU. There's no combining and no reduction-just copying.

Say GPU 0 has the original data, and GPUs 1, 2, and 3 are receivers who each get an identical copy. After the broadcast, all four GPUs have exactly the same data. Figure 10.2 shows this one-to-many pattern.

When do you need this? At the start of training, you typically initialize the model weights on one GPU (maybe GPU 0 loads them from a checkpoint file). You need to broadcast those weights to all other GPUs so everyone starts with the same model. Without broadcast, each GPU would have random, uninitialized weights, and your training would be nonsense.

-----

###### 272 Chapter 10 Distributed computing Before

| GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|
| [DATA] | [ ] | [ ] | [ ] |

###### After

| GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|
| [DATA] | [DATA] | [DATA] | [DATA] |

**Figure 10.2 Broadcast**

###### The source GPU sends identical data to all others. copies one GPU's data

**Broadcast is used for distributing model weights or input batches. to every other GPU.**

In tensor parallelism, you broadcast the input activations from GPU 0 to every GPU so each can apply its own slice of the layer to the same input. Or you might broadcast hyperparameters or configuration settings that all GPUs need. The same two costs apply. Bytes: every GPU still has to receive the full N bytes, so the bandwidth term is O(N), the same order as AllReduce (NCCL pipelines the data in chunks so no GPU has to forward much more than N). Steps: there is no combining pass, and a pipelined tree broadcast completes in about log2(P) steps for P GPUs. On 8 GPUs, that's just 3 steps instead of the 14 the ring AllReduce needs. So broadcast wins on latency, not on bytes: for small messages, where the step count dominates, broadcast is much faster than AllReduce.

###### Gather

Gather is the opposite of broadcast. Instead of one GPU sending data to everyone, each GPU sends data to one destination GPU. The destination collects all the pieces and concatenates them. Say GPU 0 has "A," GPU 1 has "B," GPU 2 has "C," and GPU 3 has "D," and GPU 0 is the destination. At the end, GPU 0 has "A, B, C, D" all together, while the other GPUs still have only their original pieces. Figure 10.3 shows this many-to-one collection step. At the end of inference, each GPU might have processed a different batch of images and produced predictions. You might want to gather all those predictions onto GPU 0 so you can write them to a file or compute overall accuracy. Or, during training, you might gather statistics from all GPUs onto one GPU for logging to TensorBoard. Gather is less common than AllReduce in training loops because you usually want all GPUs to have the combined result (AllReduce), not just one GPU (Gather). But for collecting final outputs or debugging information, Gather is exactly what you need. The communication cost is O(N) to the destination GPU, which receives data from all P-1 other GPUs.

-----

###### Understanding the hardware 273

GPU 0 GPU 1 GPU 2 GPU 3

| [A] |  | [B] |  | [C] |  | [D] |
|---|---|---|---|---|---|---|
| [A,B,C,D] |  | [B] |  | [C] |  | [D] |

**Figure 10.3 Gather**

###### Each GPU sends its data to a single destination. collects data from all GPUs

**Gather is used for collecting results or debugging. onto one destination GPU.**

###### Reduce

Reduce is like Gather, but instead of just concatenating the pieces, it combines them with an operation.

Say GPU 0 holds 1, GPU 1 holds 2, GPU 2 holds 3, and GPU 3 holds 4. Instead of collecting all four values (Gather), you want their sum. Reduce with sum gives GPU 0 the answer: 10. The other GPUs don't get this result; only the destination GPU (GPU 0) knows the total. Figure 10.4 shows this many-to-one reduction pattern.

| GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|
| [1] | [2] | [3] | [4] |

SUM

###### After:

| GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|
| [10] | [2] | [3] | [4] |

**Figure 10.4 Reduce**

**Reduce is like Gather but combines data with an operation. combines values onto a It's used when only one GPU needs the final result. single destination GPU.**

The key difference from AllReduce is that only one GPU gets the result. AllReduce would give all four GPUs the value 10. Reduce gives it only to GPU 0. This saves communication bandwidth when you don't need every GPU to have the answer. Use Reduce instead of AllReduce mainly when you're dealing with logging and metrics. If you're computing the total training loss across all GPUs and only GPU 0 needs to

-----

###### 274 Chapter 10 Distributed computing

print it to the console or write it to a log file, use Reduce. All other GPUs can continue working; they don't need to wait for or store the global loss value. This is less common in modern training because most frameworks use AllReduce everywhere for simplicity, but Reduce is more efficient when only one GPU needs the result.

###### ReduceScatter

ReduceScatter is more complex, but it's crucial for understanding modern memoryefficient training methods like Fully Sharded Data Parallel (FSDP). The name describes exactly what happens: reduce the data, then scatter the result.

Let's break it down with a concrete example. Each of your four GPUs has four values. GPU 0 has [A0, B0, C0, D0], GPU 1 has [A1, B1, C1, D1], GPU 2 has [A2, B2, C2, D2], and GPU 3 has [A3, B3, C3, D3]. Now you want to sum all the A values, all the B values, all the C values, and all the D values. That's the "reduce" part, and figure 10.5 shows how the reduced chunks are then distributed back out.

| GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|
| [A0, B0, C0, D0] | [A1, B1, C1, D1] | [A2, B2, C2, D2] | [A3, B3, C3, D3] |

ReduceScatter (sum)

| GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|
| [A0+A1+A2+A3] | [B0+B1+B2+B3] | [C0+C1+C2+C3] | [D0+D1+D2+D3] |

**Each GPU gets a unique chunk of the reduced result. ReduceScatter is used in FSDP and ZeRO optimizers.**

**Figure 10.5 ReduceScatter reduces values and returns one shard to each GPU.**

But instead of giving every GPU all four sums (which would be AllReduce), we scatter the results: GPU 0 gets the A sum, GPU 1 gets the B sum, GPU 2 gets the C sum, and GPU 3 gets the D sum. Each GPU gets exactly one chunk of the final result. Why is this useful? In FSDP, each GPU is responsible for updating a different shard of the model parameters. After computing gradients, you need to average them across GPUs, but each GPU only needs the averaged gradients for its own shard. ReduceScatter does exactly this: it computes the averaged gradients and gives each GPU only the piece it needs. This saves memory because no GPU has to store the full gradient array.

-----

###### Understanding the hardware 275

If you did ReduceScatter followed by AllGather (which we'll cover next), you'd get the same result as AllReduce. But by stopping halfway, you save memory and communication bandwidth when each GPU needs only part of the result.

###### AllGather

AllGather is the opposite of ReduceScatter and the second half of AllReduce. Each GPU starts with a unique piece of data, and AllGather collects all the pieces and gives the complete set to every GPU. There's no reduction and no math-just collecting and distributing. Think of this scenario: GPU 0 has chunk A, GPU 1 has chunk B, GPU 2 has chunk C, and GPU 3 has chunk D. After AllGather, every GPU has the complete data: A, B, C, and D concatenated together. It's like Gather (collecting from everyone), but the "All" prefix means every GPU gets the complete result, not just one destination GPU. Figure 10.6 shows this reconstruction step.

###### Before

| GPU 0 [A] | GPU 1 [B] | GPU 2 [C] AllGather | GPU 3 [D] |
|---|---|---|---|
| GPU 0 [A,B,C,D] | GPU 1 [A,B,C,D] | GPU 2 [A,B,C,D] | GPU 3 [A,B,C,D] |

###### After

**Figure 10.6 AllGather**

**Each GPU contributes its chunk: all receive the complete data. reconstructs the full AllGather is used in FSDP to reconstruct full model weights. tensor on every GPU.**

This is the key operation in FSDP. Remember how FSDP shards model parameters across GPUs to save memory? When GPU 0 needs to run a forward pass through a layer, it needs the complete layer weights, not just its shard. So it calls AllGather to temporarily reconstruct the full weights from all the shards. GPU 0 contributes its shard A, GPU 1 contributes shard B, and so on. After AllGather, all GPUs have the complete layer weights and can compute the forward pass. Then they discard the gathered weights to free memory. If you combine ReduceScatter and AllGather, you get AllReduce. ReduceScatter computes partial sums and distributes them, then AllGather collects those partial sums and gives everyone the complete result. NCCL actually implements AllReduce this way internally for large messages, using the ring algorithm to make both stages efficient.

-----

###### 276 Chapter 10 Distributed computing

### 10.1.3 Ring vs. tree algorithms

NCCL implements collective operations using two main algorithmic patterns: ring and tree. The choice between them depends on the operation type, message size, and hardware topology.

###### Ring algorithm

The ring algorithm arranges GPUs in a logical ring. Data flows in one direction around the ring, with each GPU receiving data from its predecessor and sending it to its successor. This pattern achieves optimal bandwidth utilization because all links are active simultaneously. Figure 10.7 shows the resulting communication pattern.

GPU 0

A

GPU 7 GPU 1

D

**Each GPU sends and receives simultaneously.**

GPU 6 GPU 2 **All network links are active. The ring**

**pattern is optimal for large messages and bandwidth-bound operations.**

B

GPU 5 GPU 3

C

GPU 4

**Figure 10.7 Ring collectives keep every link active to maximize bandwidth.**

Characteristics are as follows:

¡ *Bandwidth optimal-All links are active simultaneously, achieving an aggregate*

bandwidth of P × link\_bandwidth. ¡ *Latency-Requires 2(P - 1) steps for AllReduce, where P is the number of GPUs.* ¡ *Best for-Large messages (>1 MB), bandwidth-bound operations.* ¡ *Used by-AllReduce, ReduceScatter, AllGather.*

-----

###### Understanding the hardware 277

Here is an example for AllReduce on eight GPUs with 1 GB data:

¡ *Steps 1 to 7-ReduceScatter phase (each GPU gets 1/8 of the reduced data).* ¡ *Steps 8 to 14-AllGather phase (each GPU collects all chunks).* ¡ *Total-14 steps, 1.75 GB transferred per GPU.*

###### Tree algorithm

The tree algorithm arranges GPUs in a logical binary tree. During reduction, data flows up from the leaves to the root: each leaf sends its value to a parent node, which sums the inputs and passes the result one level higher. Figure 10.8 shows this process with four GPUs. Values [1], [2], [3], [4] combine into partial sums at the intermediate nodes, and the root ends up with the total.

Root 1+2+3+4

Step 2: Parents send to root

1+2 3+4

Step 1: Leaves send to parents

| GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|
| [1] | [2] | [3] | [4] |

###### Tree reduce: each level halves the number of active

**nodes. Only log(P) steps to reach the root.**

**Figure 10.8 Tree reduce: partial sums flow upward from leaves to root in log(P) steps.**

Broadcasting is the reverse. The root holds the final result and sends it down to its children, which forward it to the leaves. Figure 10.9 shows the same tree, this time distributing the result [10] to every GPU. The key advantage is step count. Both reduce and broadcast complete in log(P) steps, so a tree AllReduce on eight GPUs needs only 2 × log(8) = 6 steps, compared to the ring's 14. For small messages where latency dominates, those extra steps matter more than bandwidth.

Characteristics are as follows: ¡ *Latency optimal-Completes AllReduce in 2 × log(P) steps* ¡ *Bandwidth limited-Not all links active simultaneously* ¡ *Best for-Small messages (<1 MB), latency-bound operations* ¡ *Used by-Broadcast, Reduce, small AllReduce*

-----

###### 278 Chapter 10 Distributed computing

Root [10]

Step 2: Children send to leaves

[10] [10]

Step 1: Root sends to children

| GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|
| [10] | [10] | [10] | [10] |

###### Tree broadcast: the root's result fans out to all leaves

**in log(P) steps. It's the reverse of tree reduce.**

**Figure 10.9 Tree broadcast: the root's result fans out to all leaves in log(P) steps.**

Here is an example for AllReduce on eight GPUs with 1 KB of data:

¡ *Steps 1 to 3-Reduce up the tree (leaves to root).* ¡ *Steps 4 to 6-Broadcast down the tree (root to leaves).* ¡ *Total- 6 steps, which is much faster than the ring's 14 steps for small data.*

###### NCCL's adaptive selection

NCCL automatically chooses between ring and tree based on message size and topology. For AllReduce:

¡ *Small messages, where latency dominates-Tree wins.* ¡ *Large messages, where bandwidth dominates-Ring wins.* ¡ *The crossover point-Not a fixed byte threshold; NCCL runs an internal cost model*

over message size, GPU count, topology, and measured link speeds, so it moves from system to system. This is why you should always use NCCL instead of implementing your own collectives. NCCL's decision logic accounts for dozens of factors: NVLink topology, PCIe switches, message size, data type, and even GPU architecture. NVIDIA engineers have tuned the library over the years to extract maximum performance from every hardware configuration.

## 10.2 Tensor parallelism

We'll start with tensor parallelism because it demonstrates the core pattern of distributed computing: split work across GPUs, compute locally, combine results. The example is a matrix multiplication split across eight GPUs.

-----

###### Tensor parallelism 279

The problem is this: multiply a 4096 × 4096 input matrix by a 4096 × 4096 weight matrix. At FP16 precision, that's 32 MB for the input, 32 MB for the weights, and 32 MB for the output. That fits easily on one GPU. But imagine scaling to 16384 × 16384 matrices. Now we're at 512 MB each, and we haven't even loaded the actual model yet. Or consider a real transformer: today's large dense models use hidden dimensions of 8,192 to 16,384, so a single projection matrix can be 12288 × 12288 or bigger. One such matrix is about 302 MB at FP16, but a model built from hundreds of them, like GPT-3 at 350 GB, needs multiple GPUs. The solution is tensor parallelism. We split the weight matrix by rows across eight GPUs, so each GPU gets 1/8 of the rows (a 512 × 4096 shard), and we split the input matrix by columns to match (a 4096 × 512 shard). Each GPU multiplies its input slice by its weight shard. Because the split runs along the inner (K) dimension of the matrix multiplication, each GPU's product is a full-size 4096 × 4096 matrix that contains 1/8 of every output element's inner product: a partial sum. Then we use AllReduce to sum the partial outputs across GPUs, giving each GPU the complete result. Figure 10.10 shows the sharding pattern. The full code is in book.cu/8\_distributed/tensor\_parallel/tensor\_parallel.cu. We'll break it into logical cells, showing only the critical parts. The same binary runs on any GPU count that divides K, which is how all the measurements in this section were produced.

### 10.2.1 Initialization and GPU assignment

The first step in any distributed program is identifying which process you are and which GPU you should use. We use message passing interface (MPI) for this. We use processes rather than threads because the one-process-per-GPU model isolates device state and gives each GPU a clear rank. MPI starts and numbers those processes and lets them exchange small setup messages. The heavy tensor traffic between GPUs is NCCL's job.

**Listing 10.1 Initializing MPI and assigning each rank to a GPU**

`MPI_Init(&argc, &argv);` **Gets this process's unique**

###### ID (0 through 7 for 8 GPUs)

```
int rank, world_size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank); Gets the total number of processes
MPI_Comm_size(MPI_COMM_WORLD, &world_size); (eight for single-node)
```

`cudaSetDevice(rank);` **Assigns each rank to one GPU. On a single**

**node, the MPI rank doubles as the GPU index.**

```
ncclUniqueId nccl_id;
if (rank == 0) ncclGetUniqueId(&nccl_id);
MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
ncclComm_t nccl_comm;
ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank); Every rank joins the
```

**NCCL communicator. Rank 0 creates the NCCL communicator id; MPI broadcasts it.**

-----

###### 280 Chapter 10 Distributed computing

B0

B1 **GPU r owns one column slice of A** A0 A1 A2 A3 x **and the matching row slice of B**

**(1/4 of each matrix in memory).**

B2

I

B3

input A (M x K) weight B (K x N)

split by columns split by rows

each GPU: C\_partial = A\_r x B\_r

GPU 0 + GPU 1 + GPU 2 + GPU 3 A0 x B0 A1 x B1 A2 x B2 A3 x B3 four full-size (M x N) partial products, one per GPU

ncclAllReduce (sum)

C = A x B

complete result on every GPU

**Figure 10.10 Tensor parallelism splits the input by columns and the weight by rows; each GPU computes a fullsize partial product, and AllReduce sums them.**

MPI launches eight separate processes (one per GPU). Each process gets a unique

```
rank from 0 to 7 and pins itself to GPU rank. (A multi-node launcher would instead
```

pass each process its `rank` within the node, on one node the two are the same number.) The last three lines wire up NCCL: rank 0 creates a unique communicator ID, MPI broadcasts that ID to every process, and each process joins the NCCL communicator.

From this point forward, each process operates independently on its assigned GPU until the final collective combines their results.

-----

###### Tensor parallelism 281

### 10.2.2 Memory allocation

Each GPU allocates only its input slice, its weight shard, and a full-size output buffer for the combined result.

| Listing 10.2 Allocating memory for the input slice, weight shard, and output |
|---|
| const int M = 4096; Each rank owns a 512-wide slice of const int N = 4096; the K dimension (4,096 / 8). const int K = 4096; const int Kp = K / world_size; Input slice A_r: 4096 × 512 (4 MB at FP16 instead of 32 MB) half *d_A, *d_B, *d_C; cudaMalloc(&d_A, M * Kp * sizeof(half)); Weight shard B_r: 512 × 4096 (4 MB at FP16 instead of 32 MB) cudaMalloc(&d_B, Kp * N * sizeof(half)); cudaMalloc(&d_C, M * N * sizeof(half)); Output C: full 4096 × 4096 (32 MB). AllReduce fills it with the sum. The input memory per GPU shrinks by a factor of 8. That is the point of tensor parallelism: for matrices too large for one GPU, each GPU stores and touches only its own shard. The output buffer stays full size because AllReduce leaves every GPU holding the complete result, ready to feed the next layer without any further data movement. Each GPU performs a standard matrix multiplication on its shards using cuBLAS. The result is that GPU's partial product: full size, but containing only its slice of the inner-product sums. |
| Listing 10.3 Computing the local partial product using cuBLAS |

```
cublasHandle_t cublas_handle;
cublasCreate(&cublas_handle);
```

```
half alpha = __float2half(1.0f);
half beta = __float2half(0.0f);
```

`cublasHgemm(` **FP16 GEMM uses Tensor Cores**

`cublas_handle,` **for maximum performance.**

```
CUBLAS_OP_N,
CUBLAS_OP_N,
```

| M, N, Kp, | Shard dimensions: C_partial(4096 × 4096) = |
|---|---|
| &alpha, | A_r(4096 × 512) × B_r(512 × 4096) |

```
d_A, M,
d_B, Kp,
&beta,
d_C, M
);
```

-----

###### 282 Chapter 10 Distributed computing

```
cublasHgemm is the FP16 (half-precision) GEMM routine, and on H100 GPUs, it runs
```

on tensor cores. Each GPU's GEMM performs 1/8 of the arithmetic of the full problem (the K dimension shrank from 4,096 to 512), so with eight GPUs computing concurrently, the compute time should drop by roughly 8×. Each GPU computes independently. There's no communication during the GEMM. Communication only happens once, at the end, which is what gives tensor parallelism a chance to scale.

### 10.2.4 Combining the results with AllReduce

After local computation, one collective combines the partial products.

| Listing 10.4 Summing partial products across all GPUs with | NCCL AllReduce |
|---|---|
| ncclAllReduce(d_C, d_C, (size_t)M * N, |  |
| ncclHalf, ncclSum, nccl_comm, stream); | In-place sum of every |

```
cudaStreamSynchronize(stream); GPU's C_partial; all
```

**Waits for the collective GPUs receive the total. to finish before using the result**

After this single call, every GPU holds the complete C, as if it had computed the full GEMM locally. Figure 10.11 shows the single-node reduction path.

Single node

| GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|
| data A | data B | data C | data D |

| GPU 4 | GPU 5 | GPU 6 | GPU 7 |
|---|---|---|---|
| data E | data F | data G | data H |

---

**ncclAllReduce ring over NVLink. Each GPU sends to its neighbor and receives from the other: all links busy at once. When the ring completes, every GPU holds SUM = A+B+C+D+E+F+G+H.**

**Figure 10.11 Single-node AllReduce uses NVLink to combine partial results across eight GPUs.**

The AllReduce operation uses a ring algorithm optimized for the NVLink topology. Data flows in a circular pattern through all eight GPUs, with each GPU sending and

-----

***Tensor parallelism*** **283**

receiving simultaneously. On H100 SXM systems with NVLink 4.0, each GPU has 900 GB/s of bidirectional NVLink bandwidth, and NCCL's ring sustains a few hundred GB/s of effective AllReduce bandwidth for large messages. The ring pattern ensures that no single link becomes a bottleneck, though it still requires a number of steps that scales linearly with the GPU count. The following listing shows the measured output on an 8× H100 SXM (NVLink/ NVSwitch) node, run exactly as described earlier.

**Listing 10.5 Tensor parallel 8-GPU results (measured, 8× H100 SXM)**

```
=== 8-GPU single-node tensor parallel (K-split + AllReduce) ===
Matrix dimensions (M, N, K): 4096 x 4096 x 4096
Shard per GPU: A 4096 x 512, B 512 x 4096
Verification vs FP32 reference:
max |C| = 135.2, max abs error = 0.149265 (0.110403% of max |C|)
PASSED
```

```
Single-GPU baseline (full 4096x4096x4096 FP16 GEMM): 0.177 ms, 776 TFLOPS
Per-GPU local GEMM throughput (shard 4096x4096x512):
```

| GPU 0: 508 TFLOPS | GPU 4: 502 TFLOPS |
|---|---|
| GPU 1: 512 TFLOPS | GPU 5: 510 TFLOPS |
| GPU 2: 511 TFLOPS | GPU 6: 511 TFLOPS |
| GPU 3: 509 TFLOPS | GPU 7: 506 TFLOPS |

```
=== Summary ===
Local partial GEMM (slowest rank): 0.034 ms
Full TP step, GEMM + AllReduce (slowest rank): 0.220 ms
AllReduce share of step: 84%
Speedup vs single GPU: 0.80x on 8 GPUs
```

Read that last line again: on eight GPUs, this tensor-parallel multiply is slower than running the whole thing on one GPU. The result is correct (the verification against an FP32 reference passes, with FP16-level error), each GPU sustains around 510 TFLOPS on its shard, and the compute really did drop by 8×, from 0.177 ms to 0.034 ms. But the AllReduce takes about 0.19 ms regardless, because it has to move the full 4096 × 4096 output no matter how thin the K slices are, and at this problem size, communication accounts for 84% of the step. This is the single most important measurement in the chapter. Tensor parallelism is first a memory strategy: it lets you run layers whose weights do not fit on one GPU. Whether it also makes you faster depends on the ratio of compute to communication. The local GEMM does 2MNK / P FLOPs, while the AllReduce moves M × N output elements no matter what K is. Grow the inner dimension, and the balance shifts. Table 10.1 shows the same binary at larger sizes on the same node.

-----

###### 284 Chapter 10 Distributed computing

**Table 10.1 Measured 8-GPU tensor-parallel scaling vs. problem size (8× H100 SXM)**

| Problem (M × N × K) | 1-GPU baseline | 8-GPU TP step | AllReduce share | Speedup |
|---|---|---|---|---|
| 4,096 × 4,096 × 4,096 8,192 × 8,192 × 8,192 12,288 × 12,288 × 12,288 4,096 × 4,096 × 32,768 | 0.18 ms 1.38 ms 4.95 ms 1.34 ms | 0.22 ms 0.83 ms 1.85 ms 0.41 ms | 84% 74% 66% 52% | 0.80× 1.67× 2.67× 3.30× |

The trend is exactly what the arithmetic predicts. The 12288 × 12288 case is the projection size we used to motivate this section: 2.7× faster on 8 GPUs, with two-thirds of the step still spent in AllReduce. The last row grows only K, which multiplies compute by 8 while leaving the communication volume untouched, and the speedup improves to 3.3×. Real training and inference systems live even further up this curve: they shard much larger layers, batch many tokens per step, and overlap communication with computation for the next operation, which is how production tensor parallelism earns its keep. The mechanics, though, are exactly what you just ran: shard, compute locally, AllReduce. The honest summary for a single node: tensor parallelism always buys memory capacity. It buys speed only when each GPU has enough work to amortize the collective. Pipeline parallelism attacks the same capacity limit along model depth instead of layer width.

## 10.3 Pipeline parallelism

Tensor parallelism splits a single layer across multiple GPUs. Pipeline parallelism assigns different layers to different GPUs, so GPU 0 can work on layer 1 of one batch while GPU 1 works on layer 2 of another. The advantage of pipeline parallelism is higher throughput through concurrent execution. The challenge is keeping all GPUs busy. The naive approach (process one batch at a time through the pipeline) leaves most GPUs idle. The optimized approach (use CUDA streams to process multiple batches simultaneously) achieves near-perfect utilization. We'll implement a four-layer MLP distributed across four GPUs, with each GPU owning one complete layer. The first three GPUs each run a linear(4096 → 4096) transformation followed by a ReLU activation, while the final GPU runs linear(4096 → 1024) followed by softmax. This setup extends naturally to eight GPUs by adding more hidden layers. The difference between naive and optimized approaches is visualized in figure 10.12. The staircase pattern in figure 10.13 is used in production systems like TensorRT-LLM and vLLM for LLM inference. At steady state, all four GPUs work simultaneously on different batches.

-----

***Pipeline parallelism*** **285**

**Naive sequential: one batch at a time**

Time 0 1 2 3 4

GPU 0 ~~Batch 0~~

GPU 1 ~~Batch 0~~

GPU 2 ~~Batch 0~~

GPU 3 ~~Batch 0~~

**cudaDeviceSynchronize() after every layer: only one GPU active per timestep. Batch 1 cannot start until batch 0 leaves GPU 3.**

**Concurrent with CUDA streams: four batches in flight**

Time 0 1 2 3 4 5 6 7

GPU 0 ~~B0 B1 B2 B3~~

GPU 1 ~~B0 B1 B2 B3~~

GPU 2 ~~B0 B1 B2 B3~~

GPU 3 ~~B0 B1 B2 B3~~ steady state begins

**Each GPU starts the next batch as soon as it hands off the previous one. From timestep 3 onward, all four GPUs work on different batches simultaneously.**

**Figure 10.12 Pipeline-parallel execution timeline for sequential and concurrent batch processing across four GPUs**

***10.3.1 Naive pipeline parallelism implementation: Sequential processing***

The simplest way to implement pipeline parallelism is sequential: copy data to GPU 0, compute, copy to GPU 1, compute, copy to GPU 2, compute, copy to GPU 3, compute, copy back to host. Repeat for the next batch.

**Listing 10.6 Sequential batch processing with blocking synchronization**

```
for (int batch = 0; batch < num_batches; batch++) {
```

`cudaMemcpy(` **Blocking host-to-device (H2D)**

`d_input[0],` **copy for batch data**

-----

###### 286 Chapter 10 Distributed computing

```
h_batches[batch],
size,
cudaMemcpyHostToDevice
);
```

```
for (int gpu = 0; gpu < num_gpus; gpu++) {
cudaSetDevice(gpu);
mlp_layer<<<grid, block>>>(
d_input[gpu],
d_output[gpu]
);
cudaDeviceSynchronize();
if (gpu < num_gpus - 1) {
cudaMemcpy(
d_input[gpu+1],
d_output[gpu],
size,
cudaMemcpyDeviceToDevice
);
}
```

###### Computes a forward pass on this GPU's layer

###### Explicit sync forces sequential execution. Blocking peer-to-peer (P2P) copy to the next GPU in pipeline

```
}
```

`cudaMemcpy(` **Blocking device-to-host (D2H)**

`h_output[batch],` **copy to retrieve the final result**

```
d_output[num_gpus-1],
size,
cudaMemcpyDeviceToHost
);
}
```

The complete implementation is available at book.cu/8\_distributed/pipeline\_ parallel/pipeline.cu. This file contains both the naive and optimized implementations in a single program for direct comparison. Running ./pipeline 4 executes both versions sequentially and reports the speedup. The code structure is covered in detail in book.cu/8\_distributed/pipeline\_parallel/ARCHITECTURE.md. The problem is cudaDeviceSynchronize(). This blocks the CPU until the GPU finishes. Since we call it after every layer, we force a global ordering: GPU 0 must finish before GPU 1 starts, GPU 1 must finish before GPU 2 starts, and so on. Only one GPU is active at a time. Figure 10.13 shows the intended steady-state pipeline shape. The following listing captures the metrics from the naive sequential pipeline run so we can quantify the bottleneck.

**Listing 10.7 Sequential pipeline performance results**

```
Naive Pipeline (4 GPUs)
Throughput: 167,434 samples/sec
Speedup: 1.1x (compared to 1 GPU)
Efficiency: 27%
```

-----

###### Pipeline parallelism 287

| Time | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| GPU 0 (layer 1) | batch 0 | batch 1 | batch 2 | batch 3 |  | bubble (idle) |  |
| GPU 1 (layer 2) |  | batch 0 | batch 1 | batch 2 | batch 3 |  |  |
| GPU 2 (layer 3) GPU 3 |  | bubble (idle) | batch 0 | batch 1 batch 0 | batch 2 batch 1 | batch 3 batch 2 | batch 3 |

(layer 4)

###### Synchronization barrier: the host waits here once, after all batches are submitted.

###### Each microbatch advances one stage per timestep, so in steady state

###### (timesteps 3-4 here) every GPU is busy with a different batch. Gray regions are idle bubbles while the pipeline fills and drains.

**Figure 10.13 Pipeline parallelism splits layers across GPUs and overlaps multiple batches in flight.**

Adding GPUs makes things worse. We have four GPUs but get only a 1.1× speedup over a single GPU. The efficiency (speedup divided by GPU count) is 27%, meaning we're wasting 73% of our hardware. The extra GPUs add latency (data transfer overhead) without adding throughput.

### 10.3.2 Optimized implementation: CUDA streams

CUDA streams solve the concurrency problem. A stream is an independent queue of GPU operations. Operations in different streams can execute simultaneously. By assigning each batch to its own stream, we allow multiple batches to be in flight at once.

Here is the core idea: ¡ *Stream 0 processes batch 0 (GPU 0 → GPU 1 → GPU 2 → GPU 3).* ¡ *Stream 1 processes batch 1 (GPU 0 → GPU 1 → GPU 2 → GPU 3).* ¡ *Stream 2 processes batch 2 (GPU 0 → GPU 1 → GPU 2 → GPU 3).* ¡ *Stream 3 processes batch 3 (GPU 0 → GPU 1 → GPU 2 → GPU 3).* While batch 0 is on GPU 1, batch 1 can start on GPU 0. While batch 0 is on GPU 2, batch 1 is on GPU 1, and batch 2 is on GPU 0. This creates a "staircase" pattern where all GPUs are working simultaneously.

**Listing 10.8 Async batch processing with streams and events**

```
cudaStream_t streams[num_batches];
cudaEvent_t events[num_batches][num_gpus];
for (int b = 0; b < num_batches; b++) {
cudaStreamCreate(&streams[b]); Creates one stream per batch
for (int g = 0; g < num_gpus; g++) { for independent execution
cudaEventCreate(&events[b][g]);
```

-----

###### 288 Chapter 10 Distributed computing

```
}
```

```
}
for (int b = 0; b < num_batches; b++) {
```

`cudaMemcpyAsync(` **Async H2D copy returns**

`d_input[0],` **immediately without blocking.**

```
h_batches[b],
size,
cudaMemcpyHostToDevice,
streams[b]
);
for (int g = 0; g < num_gpus; g++) {
cudaSetDevice(g);
if (g > 0) {
```

`cudaStreamWaitEvent(` **Waits for the previous GPU's**

`streams[b],` **event before copying data**

```
events[b][g-1],
0
);
cudaMemcpyAsync(
d_input[g],
d_output[g-1],
size,
cudaMemcpyDeviceToDevice,
streams[b]
);
}
mlp_layer<<<grid, block, 0, streams[b]>>>( Launches the kernel
```

`d_input[g],` **on the stream for** `d_output[g]` **concurrent execution**

```
);
cudaEventRecord(events[b][g], streams[b]); Records the event to
```

`}` **signal completion to**

**the next GPU**

```
cudaMemcpyAsync(
h_output[b],
d_output[num_gpus-1],
size,
cudaMemcpyDeviceToHost,
streams[b]
);
}
for (int b = 0; b < num_batches; b++) { Final sync only at the
cudaStreamSynchronize(streams[b]); end, not per batch
}
```

There are three critical changes from the naive version:

¡ *All operations are async. cudaMemcpyAsync returns immediately. The kernel launch*

`<<<grid, block, 0, streams[b]>>>` also returns immediately. The host CPU doesn't wait for the GPU.

-----

###### Evaluating scaling strategies 289

¡ *We use events instead of cudaDeviceSynchronize(). GPU 1 calls cudaStreamWaitEvent*

to wait for GPU 0's event, but this doesn't block the CPU. GPU 1 tells the CUDA runtime "don't start my work until GPU 0's event fires," then the CPU continues to submit work for GPU 2, GPU 3, and other batches. ¡ *We synchronize only at the very end after all batches have been submitted. This allows the*

CUDA runtime to schedule work optimally across all streams and GPUs. The following listing reports the improved throughput once we overlap stages with CUDA streams.

**Listing 10.9 Streamed pipeline performance results**

```
Optimized Pipeline (4 GPUs)
Throughput: 697,926 samples/sec
Speedup: 4.17x
Efficiency: 104%
Time saved: 76%
```

We get a 4.17× speedup from four GPUs. The apparent 104% efficiency comes from overlapping host-device transfers with compute, not from the GPUs somehow exceeding their limits. While GPU 3 finishes batch 0, GPU 0 can already be loading batch 1.

The practical result is a 76% reduction in wall-clock time for the same work.

### 10.3.3 Why streams work

Operations in different streams can execute simultaneously, even on different GPUs. The CUDA runtime looks at all pending work across all streams and schedules it to maximize hardware utilization.

Let's compare the two approaches: ¡ *Naive-The CPU submits work, waits for the GPU, submits more work, waits, and*

so on. ¡ *Streams-The CPU submits work to multiple streams without blocking after each*

step. The CUDA runtime can then overlap transfers and kernels across GPUs wherever dependencies allow. The CPU is the bottleneck in the naive version. Without the per-layer cudaDevice-

```
Synchronize(), the CPU queues all the work up front and the runtime overlaps it
```

across GPUs.

## 10.4 Evaluating scaling strategies

There are two ways to think about scaling: strong scaling and weak scaling:

¡ *Strong scaling-Fix the problem size, add more GPUs, and measure speedup. For*

example: a 4096 × 4096 GEMM on one GPU takes 1 ms. On eight GPUs, how long does it take? If it takes 0.125 ms (1/8), that's perfect strong scaling. If it takes 0.2 ms, that's a 5× speedup (62.5% efficiency).

-----

###### 290 Chapter 10 Distributed computing

¡ *Weak scaling-Scale the problem size with the GPU count, then measure effi-* ciency. For example: a 4096 × 4096 GEMM per GPU takes 1 ms. If you have eight GPUs and do 8× the work (4096 × 32768), does it still take 1 ms per GPU? If yes, that's perfect weak scaling. If it takes 1.2 ms, that's 83% weak scaling efficiency. Large training runs care more about weak scaling, because extra GPUs usually go to a larger model rather than a faster run of the same one. The questions that matter day to day are, does the model fit, and does each GPU stay busy?

Our benchmarks demonstrate both types: ¡ *Strong scaling example-In the pipeline parallelism test, we processed 256 batches* of size 64 (a fixed problem). With one GPU, throughput was ~600,000 samples/s. With four GPUs using streams, we achieved 698,000 samples/s (a 4.17× speedup). The problem size stayed fixed, but we added GPUs to go faster. This is strong scaling. ¡ *Weak scaling example-The* tensor parallelism sweep in table 10.1 shows the weak-scaling effect along the problem-size axis: growing the problem from 4096^3 to 12288^3 multiplies the compute by 27×, but the measured 8-GPU step time by only 8.4× (0.22 ms to 1.85 ms), because the fixed communication cost is amortized over more arithmetic. Per-GPU shard throughput held at roughly 500-750 TFLOPS across the whole sweep. That is the regime real systems aim for: scale the

work per GPU up until compute, not communication, dominates the step.

Pipeline parallelism (which we implemented) typically shows good strong scaling up to 8 to 16 GPUs; then efficiency drops as communication dominates. Tensor parallelism shows excellent weak scaling: if you double the model size and double the GPUs, efficiency stays high.

A single node's GPUs can hold only so much. When a node reaches that limit, combine the three strategies: data parallelism gives each GPU a different batch, tensor parallelism splits wide layers across the GPUs, and pipeline parallelism splits model depth across stages. Each strategy targets a different dimension: batch size, layer width, or model depth.

### 10.4.1 Parallelism strategies

Before diving into when to use each strategy, we need to understand the three main approaches to distributing deep learning workloads: data parallelism, tensor parallelism, and pipeline parallelism. Each splits the work differently and targets different bottlenecks.

###### Data parallelism

Data parallelism is the simplest and most common form of distributed training. It keeps a complete copy of the model on every GPU and gives each GPU a different batch of training data. Each GPU computes gradients from its batch independently, and then all GPUs average their gradients before updating the model. Because every GPU starts with the same model and updates with the same averaged gradients, all GPUs stay perfectly synchronized. Figure 10.14 shows this replication pattern.

-----

###### Evaluating scaling strategies 291

###### 8 GPUs in

Data w/ batch size 1,024 **this node** (indices from 0-1,023)

B[0:127] B[128:255] B[256:383] B[384:511]

B[512:639] B[640:767] B[768:895] B[896:1023]

Neural

net **Figure 10.14 Data**

**parallelism replicates the**

###### Copy the entire network of weights to model and assigns a different

**each GPU to process 1/8th of a batch. batch to each GPU.**

Let's walk through a concrete example. Say you're training an image classifier with a batch size of 256 images. On a single GPU, you'd load all 256 images, run a forward pass, compute the loss, run a backward pass to get gradients, and then update the weights. This takes, say, 100 ms per iteration. Now you add seven more GPUs for a total of eight. With data parallelism, each GPU gets 32 images (256 / 8 = 32). GPU 0 processes images 0 to 31, GPU 1 processes images 32 to 63, and so on. Each GPU runs the forward and backward passes independently on its 32 images. This takes about 100 / 8 = 12.5 ms because each GPU has 8× less work. After the backward pass, each GPU has computed gradients from its 32 images. But you want gradients averaged across all 256 images, not just one GPU's batch. So all eight GPUs perform an AllReduce operation to average their gradients. This AllReduce takes about 7 ms on NVLink. Then every GPU updates its local model copy with the same averaged gradients. The total time is 12.5 ms (compute) + 7 ms (communication) = 19.5 ms per iteration. That's a 5× speedup over the single-GPU 100 ms, even with communication overhead. If you had perfect scaling, it would be 8×, but 5× is still excellent. The main advantage of data parallelism is that it does not require changing the model architecture. PyTorch's DistributedDataParallel wrapper handles the AllReduce calls automatically. For models that fit on one GPU, it is usually the first distributed strategy to try. The limitations appear only when your model grows too large to fit on a single GPU. A seven-billion-parameter (7B) model with optimizer states needs about 112 GB of memory, which exceeds even an 80 GB H100. At that point, you need FSDP or tensor parallelism to shard the model across GPUs.

-----

###### 292 Chapter 10 Distributed computing

###### FSDP

FSDP is an advanced form of data parallelism that addresses the memory limits of standard replication. The reason for using it becomes clear once the model no longer fits comfortably on a single GPU.

Standard data parallelism requires the full model to fit on each GPU. For a 7B-parameter model at FP16 (2 bytes per parameter), you need 14 GB just for the weights. But that's not all. You also need 14 GB for gradients (the same size as the weights), plus the Adam optimizer state kept at FP32: a 28 GB master copy of the weights, 28 GB of momentum, and 28 GB of variance. The standard accounting for mixed-precision training is 16 bytes per parameter (see the ZeRO paper at https:// [arxiv.org/abs/1910.02054), which for 7B parameters is 112 GB per GPU.](https://arxiv.org/abs/1910.02054)

An H100 has 80 GB of memory, so the training state alone already does not fit, and that's before the activations (the intermediate values during the forward pass) and the batch data, which can add another 10 to 20 GB for large models. Standard data parallelism fails.

FSDP solves this by sharding (splitting) everything across GPUs. With eight GPUs, instead of each GPU storing 112 GB, each GPU stores 112 / 8 = 14 GB. Now the model trains comfortably, with room for activations and batch data.

If each GPU only has 1/8 of the model weights, it still has to reconstruct the active layer to run the forward pass. FSDP uses AllGather to temporarily reconstruct the full weights for each layer, computes that layer, then immediately discards the gathered weights to free memory.

Let's walk through a forward pass on GPU 0. It needs to compute the first layer. GPU 0 has shard 0 of the layer weights, GPU 1 has shard 1, and so on. GPU 0 calls AllGather to collect all eight shards and reconstruct the complete layer weights. Now GPU 0 has the full weights in memory temporarily. It computes the layer's output, then discards the gathered weights. Only the 14 GB shard stays in memory. GPU 0 repeats this process for every layer.

The backward pass works the same way: AllGather reconstructs the weights and computes gradients, then ReduceScatter averages gradients across GPUs and distributes the averaged gradients back to their respective shards. Each GPU updates only its own shard of parameters.

The tradeoff is communication. Standard data parallelism does one AllReduce per training step. FSDP does AllGather + ReduceScatter for every layer, which moves about 1.5× the data of standard data parallelism. But this enables training models that otherwise wouldn't fit. On eight H100s, you can train a seven-billion-parameter model with FSDP that would be impossible with standard data parallelism.

This is the same basic idea used by PyTorch's FSDP and DeepSpeed's ZeRO family of optimizers. When the model no longer fits with standard data parallelism, sharding weights, gradients, and optimizer state is often the next step.

-----

###### Evaluating scaling strategies 293

###### Hardware: SXM vs . PCIe GPUs

We've discussed NVLink and PCIe, but there's a third connection type that matters for high-end servers: SXM, NVIDIA's proprietary high-bandwidth GPU socket (SXM4 on Ampere, SXM5 on Hopper).

PCIe GPUs are what most people are familiar with. You buy an RTX 4090 or an A100 PCIe card, plug it into a PCIe slot on your motherboard, connect the power cables, and you're done. The GPU communicates with the CPU and other GPUs through the PCIe bus at 32 to 64 GB/s per direction, depending on the PCIe generation. You can have four to eight GPUs in a server, and if you need to replace a failed GPU, you just unplug it and swap in a new one. This flexibility comes at a cost: bandwidth. PCIe 4.0 x16 gives you 64 GB/s total (32 GB/s in each direction), and PCIe 5.0 x16 doubles that, which is fine for single-GPU workloads but becomes a bottleneck for distributed training.

SXM GPUs are modules mounted on a custom GPU baseboard. The entire server is designed around the GPUs, not the other way around. Power is delivered through the motherboard at up to 700 W per GPU (compared to 300 to 450 W for PCIe). Most importantly, all eight GPUs are connected via an NVLink fabric built into the motherboard, giving each GPU 900 GB/s of total NVLink bandwidth (18 links, reachable toward any peer through NVSwitch).

This bandwidth difference is why we see 99% to 100% scaling efficiency in the 8-GPU tensor parallelism benchmarks. Those benchmarks were run on SXM H100s with NVLink. If you ran the same code on PCIe GPUs, you'd see 70% to 80% efficiency because the AllReduce communication would take 20× longer. The compute time stays the same, but the communication time dominates.

The tradeoff is cost. SXM parts carry a premium of roughly 15% to 40% per GPU over their PCIe siblings, so they make sense for production training at scale where NVLink bandwidth pays for itself; for development or small-scale work, PCIe is fine.

### 10.4.2 When to use each strategy

Choosing the right parallelism strategy depends on your model architecture and hardware constraints. The decision tree starts with the simplest approach and adds complexity only when necessary.

The starting point for most distributed training is data parallelism. When your model fits comfortably on a single GPU but training is too slow, data parallelism offers the easiest path to speedup. This approach is built into PyTorch's DistributedDataParallel and TensorFlow's distribution strategies, making it nearly transparent to users. Data parallelism scales well across a single node's GPUs before communication overhead becomes significant. The main tradeoff is that batch size must scale with GPU count, which can sometimes hurt convergence if the batch becomes too large.

When data parallelism alone isn't enough, the next question is whether your bottleneck is width or depth. Models with very wide layers (hidden dimensions of 8,192 to 16,384 in today's large dense models) benefit from tensor parallelism, which we

-----

###### 294 Chapter 10 Distributed computing

demonstrated with the 8-GPU example. Each GPU handles a portion of the matrix multiplication, computing partial results that are combined with AllReduce. This approach supports training models that physically won't fit in a single GPU's memory. The communication cost is proportional to the number of layers because every layer requires an AllReduce. Figure 10.15 contrasts this with pipeline parallelism. For models that are extremely deep rather than wide, pipeline parallelism offers a different tradeoff. In our stream-based example, once the pipe- machine 1 Layer 1 machine 2 line filled, all four GPUs stayed busy on different batches at the same time. Each GPU owns complete layers rather than shards of layers, which means communication happens only at layer boundaries instead of within each machine 3 Layer 2 machine 4 layer. This minimizes the total volume of data transferred but introduces a

new challenge: balancing computation **Figure 10.15 Tensor parallelism splits layers** across pipeline stages. If one GPU's **horizontally, while pipeline parallelism splits them**

**vertically.**

layers are slower than the others, it becomes a bottleneck that idles the rest of the pipeline. The other tradeoff is latency: although throughput is high (many batches processed per second), individual batches take longer because they must traverse the entire pipeline sequentially.

Real production systems rarely use a single strategy. Llama 3 405B, for example, combined tensor parallelism, pipeline parallelism, and data parallelism. The general principle is to start with data parallelism for its simplicity, add tensor parallelism when individual layers exceed single-GPU memory, and introduce pipeline parallelism when the full model depth won't fit even with tensor parallelism.

###### Summary

¡ GPU interconnect topology determines every distributed performance decision. PCIe Gen4 × 16 provides 32 GB/s per direction (Gen5 doubles it), while NVLink 4.0 delivers 450 GB/s per direction (900 GB/s bidirectional) between GPUs on the same node. That gap is why a collective over NVLink can be an order of magnitude faster than one over PCIe. ¡ NCCL's six collective operations (AllReduce, Broadcast, Gather, Reduce, Reduce- Scatter, AllGather) handle all distributed communication. The library automatically routes data over the fastest available links. ¡ NCCL selects ring or tree algorithms with an internal cost model: tree for small messages where latency dominates, ring for large messages where bandwidth

-----

###### Summary 295

dominates. The crossover depends on topology and GPU count, not a fixed byte threshold. ¡ Tensor parallelism splits a single layer's weight matrix across GPUs. Our 8-GPU

and 16-GPU implementations achieved 99% to 100% single-node efficiency using MPI for coordination and NCCL for AllReduce. ¡ Pipeline parallelism assigns different layers to different GPUs. The naive sequential version achieved only 27% efficiency, while the CUDA stream-based version reached a 4.17× speedup by overlapping computation across stages. ¡ Strong scaling fixes the problem size and adds GPUs; weak scaling grows the

problem with GPU count. Both metrics matter when evaluating a distributed strategy. ¡ Practical single-node scaling efficiency can exceed 90% across 8 GPUs and stay

near 100% under weak scaling. The exact result depends on the workload's compute-to-communication ratio. ¡ Tensor parallelism is primarily a memory strategy. In our 8× H100 measurements,

it only becomes a speedup once the per-GPU compute is large enough to amortize the AllReduce, which took 84% of the step at 4096^3 and 52% at 4096 x 4096 x

```
32768. This chapter therefore keeps tensor parallelism inside one node.
```

¡ SXM GPUs with NVLink mesh are the standard for multi-GPU training nodes.

PCIe GPUs work for inference and single-GPU training but lack the interconnect bandwidth for tight coupling. ¡ Production systems can combine data parallelism across the GPUs, tensor parallelism across layer width, and pipeline parallelism across model depth. Start with data parallelism and add complexity only when memory or model size demands it.

-----

~~*1 1*~~

## CUTLASS

---

| This chapter covers |
|---|
| ¡ Why CUTLASS lets us fuse GEMM workflows beyond cuBLAS |
| ¡ Hierarchical tiling that maps grids → |
| threadblocks → warps |
| ¡ Extending kernels with CuTe layout and iterator |
| abstractions |
| ¡ Tuning tile shapes, pipeline stages, and |
| epilogues for throughput |
| ¡ Comparing architectures and precision choices |
| from Ampere to Blackwell |
| ¡ Profiling and scaling CUTLASS kernels, including |
| multi-GPU execution |

CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) matters when the default GEMM path is almost right but not quite enough. Once a workload needs fused epilogues, custom tile shapes, architecture-specific scheduling, or multi-GPU-aware kernels, the convenience of a black-box library starts to involve a

-----

###### Why CUTLASS matters 297

tradeoff with the control needed for performance work. This chapter puts that tradeoff in concrete terms by showing where CUTLASS sits between using cuBLAS as is and writing every kernel detail by hand. Along the way, we'll cover how a GEMM is decomposed across the GPU hierarchy, how CUTLASS 3.x assembles kernels from reusable pieces, how precision and architecture choices affect the implementation, and where profiling fits into the workflow when you need to justify using a custom kernel at all.

## 11.1 Why CUTLASS matters

So far, cuBLAS has been the default answer for GEMM-heavy workloads. We hand it our matrices, and it returns fast, reliable results. For a large class of problems, that is exactly the right choice. The reason to look beyond it is that deep learning layers are rarely just a GEMM. They are usually a sequence of operations: a GEMM, a bias add, an activation, and sometimes more. The following listing shows the kind of sequence we would prefer to fuse.

**Listing 11.1 PyTorch layer sequence to fuse**

```
# A typical sequence in a neural network layer
D = torch.matmul(A, B)
E = D + C # Add bias
F = torch.relu(E) # Apply activation
```

Each of these operations-the matrix multiply, the bias add, and the ReLUtraditionally requires a separate kernel launch. Each launch has overhead, and more importantly, each one involves writing intermediate results back to global GPU memory (D and E in our example) only to immediately read them back in the next kernel. This round trip to DRAM is one of the most expensive things we can do on a GPU. This is where CUTLASS comes in. CUTLASS is not a library in the same sense as cuBLAS. It is a collection of C++ templates that gives us the building blocks to construct our own high-performance kernels. It matters when the library path is close but not quite right: when we need control over tiling, data movement, or the epilogue, not just a faster torch.matmul. Imagine rewriting our PyTorch example into a single CUDA kernel. The output of the matrix multiplication would never leave the high-speed registers or shared memory of the streaming multiprocessors (SMs). The bias addition and ReLU application would happen on the fly, right before the final result is written to global memory just once. The performance gains from eliminating those extra memory round trips can be substantial, though the tradeoff still has to be measured against strong framework or cuBLAS baselines. In this chapter, we're going to open the black box. We'll use CUTLASS to build a GEMM kernel from first principles, understanding how CUTLASS decomposes the problem into tiles that can be processed in parallel. Then we'll see how to add our own

-----

###### 298 Chapter 11 CUTLASS

custom epilogue-the set of operations that happen after the main multiplication-to build a fused kernel that can avoid extra launches and extra DRAM traffic when the default library path is not enough.

### 11.1.1 The core idea: Decomposing a GEMM

To understand how CUTLASS achieves high performance, we need to revisit how matrix multiplication is mapped to the GPU's execution model. The core principle is *hierarchical decomposition. We take a large problem and break it down into smaller pieces* that fit neatly into the different levels of the GPU's hierarchy: grids, threadblocks, warps, and threads.

Imagine our goal is to compute C = A x B, where A, B, and C are large matrices. CUTLASS views this not as a single massive operation but as a grid of smaller, independent matrix multiplications. Figure 11.1 shows the hierarchy we will keep referring back to throughout the chapter.

| Block tile | Warp tile Il Thread tile |
|---|---|
|  | Uses hierarchical "CuTe" layouts to define how a kernel is run |

###### Whole matrix

**Figure 11.1 CUTLASS decomposes a large GEMM into grids of threadblocks, warps, and threads.**

The decomposition works as follows. First, the entire output matrix C becomes the full problem, which CUTLASS partitions into a 2D grid of threadblocks. Each thread block then computes a single rectangular tile of the output matrix C, typically around 128 × 256 elements in size. To compute its output tile, the thread block iterates through tiles from the A and B matrices, loading them into shared memory and performing a sequence of smaller matrix multiplications. Within each thread block, the work is further divided among groups of 32 threads called warps, with each warp computing

-----

###### Building our first CUTLASS kernel 299

a smaller tile of the result, usually around 64 × 64 elements. Finally, each individual thread within a warp handles a tiny portion of the computation, typically accumulating just a few values in its own registers. This hierarchical approach maximizes data reuse by keeping tiles in fast shared memory for multiple computations before moving on to the next set of data, minimizing the number of slow trips to global DRAM. As we build out the CUTLASS kernel, we will see how this hierarchical model is expressed in code, from defining the tile sizes at each level to writing the main loop that iterates over the K dimension of the input matrices.

## 11.2 Building our first CUTLASS kernel

Now let's translate these concepts into code. We'll walk through a complete, working example: book.cu/9\_cutlass/unofficial/single\_gpu\_gemm.cu. This example uses the CUTLASS 3.x API, which is designed to be highly composable and target modern NVIDIA architectures like Hopper. The code is a stack of C++ templates that declares the GEMM we want to perform. The code consists of three main parts: defining the kernel's characteristics, building the kernel from those definitions, and finally, launching it from host code.

###### 11.2.1 Defining the kernel's shape

The first part of the file is a series of using statements. This is where we define the configuration of our GEMM, including the data types, memory layouts, and, most importantly, the tile shapes. The following listing highlights that setup.

**Listing 11.2 CUTLASS type definitions for the GEMM kernel**

```
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;
```

This configuration tells CUTLASS that we're performing a mixed-precision multiply in which the input matrices A and B use a 16-bit floating point, but intermediate accumulations happen in a 32-bit floating point. With FP16 inputs, a long inner-product sum accumulated in FP16 loses precision as small terms are swamped by the growing sum: once the running total is large enough, each new term is smaller than the sum's least significant bit and is silently dropped. It can also overflow past FP16's maximum of 65,504. Promoting the accumulator to FP32 is what keeps the calculation numerically stable. The matrices are stored in row-major order, which is standard for C-style arrays. The TileShape directly implements our tile hierarchy, telling CUTLASS that each thread block will compute a 128 × 256 tile of the output matrix C, with the thread block iterating by loading 128 × 64 tiles from A and 64 × 256 tiles from B into shared

-----

###### 300 Chapter 11 CUTLASS

memory. The ClusterShape enables threadblocks to be organized into clusters, which is particularly important on Hopper GPUs for asynchronous data movement capabilities.

### 11.2.2 Composing the kernel with CollectiveBuilder

Once we have our definitions, we use CollectiveBuilder to compose the two main parts of our kernel: the Mainloop and the Epilogue. The following listing shows the relevant snippet.

**Listing 11.3 Building a CUTLASS kernel with CollectiveBuilder**

```
using CollectiveMainloop = typename
cutlass::gemm::collective::CollectiveBuilder<
ArchTag, cutlass::arch::OpClassTensorOp,
ElementInput, LayoutA, AlignmentA,
ElementInput, LayoutB, AlignmentB,
ElementAccumulator,
TileShape, ClusterShape,
cutlass::gemm::collective::StageCountAutoCarveout<
static_cast<int>(sizeof(typename
CollectiveEpilogue::SharedStorage))>,
KernelSchedule
>::CollectiveOp;
using CollectiveEpilogue = typename
cutlass::epilogue::collective::CollectiveBuilder<
ArchTag, cutlass::arch::OpClassTensorOp,
TileShape, ClusterShape,
cutlass::epilogue::collective::EpilogueTileAuto,
ElementAccumulator, ElementCompute,
ElementC, LayoutC, AlignmentC,
ElementD, LayoutD, AlignmentD,
cutlass::epilogue::collective::EpilogueScheduleAuto,
EpilogueOp
>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
Shape<int, int, int>,
CollectiveMainloop,
CollectiveEpilogue
>;
```

```
CollectiveBuilder automatically selects and the most efficient implementation details
```

for scheduling instructions and moving data based on the high-level configuration we provided. CollectiveMainloop handles the core A × B computation: loading tiles from global to shared memory, managing the computation pipeline, and performing the matrix-multiply-accumulate operations within the thread block. CollectiveEpilogue takes the accumulated result from the main loop, performs any final operations, and stores it to the output matrix D in global memory. Finally, GemmUniversal stitches these two parts together into a complete, runnable kernel.

-----

###### Building our first CUTLASS kernel 301

One detail in the CollectiveBuilder signature is worth pulling out: the AlignmentA,

```
AlignmentB, AlignmentC, and AlignmentD parameters. These parameters tell CUTLASS,
```

in elements, how the rows (for RowMajor operands) of each matrix are aligned in memory, and they are not independent of the TileShape. CUTLASS uses the alignment to pick the widest legal vectorized load/store instructions for moving tiles between global, shared, and register memory, so the alignment values you choose must be compatible with the K- and N-dimensions of the tile shape (here Shape<\_128, \_256, \_64>). For 16-bit operands on Hopper, an alignment of eight elements (a 128-bit access) is the common choice, and it divides cleanly into the inner-tile dimensions chosen earlier. If you change the tile shape, you may need to revisit the alignment values to keep the chosen vector width legal.

### 11.2.3 Launching the kernel from the host

The last piece of the puzzle is the host-side code that launches this kernel. CUTLASS provides a convenient wrapper, GemmUniversalAdapter, to handle the details of setting up and running the kernel. The following listing shows the launch code.

**Listing 11.4 Host-side launch code using GemmUniversalAdapter**

```cpp
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
int main() {
int M = 8192, N = 8192, K = 8192;
StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {K, N, 1});
StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
typename Gemm::Arguments args {
cutlass::gemm::GemmUniversalMode::kGemm,
{M, N, K},
{d_A, stride_a, d_B, stride_b},
{{1.0f, 0.0f}, d_C, stride_c, d_D, stride_d}
};
Gemm gemm_op;
size_t workspace_size = Gemm::get_workspace_size(args);
void* workspace_ptr = nullptr;
if (workspace_size > 0) {
cudaMalloc(&workspace_ptr, workspace_size);
}
cutlass::Status status = gemm_op.initialize(args, workspace_ptr);
if (status != cutlass::Status::kSuccess) {
std::cerr << "CUTLASS GEMM initialization failed" << std::endl;
exit(1);
}
```

-----

###### 302 Chapter 11 CUTLASS

```
gemm_op.run();
cudaDeviceSynchronize();
if (workspace_ptr) cudaFree(workspace_ptr);
}
```

The host code is responsible for the standard CUDA logistics: allocating memory on the device, copying input data from the host, and eventually copying the result back. The key CUTLASS-specific part is the creation of the Arguments struct. This struct packages up all the information the kernel needs to run: the problem dimensions

```
(M, N, K), pointers to the device memory for the matrices, and their strides (how to
```

move from one row or column to the next in memory). With the arguments prepared, we simply initialize() the gemm\_op object and then call run() to launch the kernel on the GPU. We defined a high-performance GEMM kernel without writing a single line of deviceside global or device code. We simply described the shape and structure of our problem, and the CollectiveBuilder did the heavy lifting of generating an optimized kernel to match.

## 11.3 CuTe: The language of layouts

A powerful new library called CuTe underpins the composable CollectiveBuilder API of CUTLASS 3.x. CuTe is a core component of modern CUTLASS that provides a "language" for describing and manipulating tensor layouts in memory. At its heart, CuTe is all about decoupling the logical view of a tensor (its shape) from its physical representation in memory (its strides). A Layout in CuTe is a compile-time function that maps a logical coordinate (row, col, …) to an offset in memory. Consider a simple 128 × 64 matrix stored in row-major order. The following listing defines its layout in CuTe.

**Listing 11.5 Defining a CuTe row-major layout**

```cpp
#include <cute/tensor.hpp>
auto shape = cute::make_shape(128, 64);
auto stride = cute::make_stride(64, 1);
auto layout = cute::make_layout(shape, stride);
```

This layout object now "knows" how to find any element. If we ask for the element at coordinate (r, c), it will compute the memory offset as r \* 64 + c \* 1. CuTe allows us to perform "layout algebra." We can programmatically slice, tile, and partition these layouts to create new views of the data without ever moving the data itself. This is exactly what CUTLASS does internally to manage the complex mapping of tiles from global memory through shared memory and into the registers of each thread. It's the engine that powers the hierarchical decomposition we saw earlier.

-----

###### Tuning for performance 303

While our single\_gpu\_gemm.cu example hides most of the CuTe complexity behind the CollectiveBuilder, we can see it peeking through in the host-side stride definitions. The following listing shows the relevant fragment.

| Listing 11.6 CuTe stride definitions for host-side setup |
|---|
| using StrideA = typename Gemm::GemmKernel::StrideA; StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}); Here, we're using a CuTe helper function to create a stride object that the CUTLASS kernel understands. We're telling it that our matrix A, of size M x K, is stored contigu- ously in memory (a "packed" layout). Although the high-level builders are often sufficient, writing truly custom and novel kernels often requires dropping down to the CuTe level to define bespoke data layouts and transformations. Once the basic kernel shape is clear, the next question is how to make it practical. The high-level, declarative nature of the CUTLASS 3.x API does not mean we give up control. It exposes several tuning knobs for fitting the kernel to a specific problem and GPU architecture. The choice of TileShape is the single most important tuning parameter. The follow- ing listing illustrates how we set it. |
| Listing 11.7 Configuring TileShape for the GEMM |

```
using TileShape = Shape<_128, _256, _64>;
```

This shape determines the tradeoff between arithmetic intensity and resource usage. Larger M and N dimensions increase the amount of computation each thread block performs for the data it loads, improving arithmetic intensity and potentially improving performance, but they also require more shared memory to store the tiles of A and B and more registers to hold the accumulators. Larger K dimensions can improve performance by doing more work per data load, though this has a more complex relationship with the kernel's main loop.

Finding the optimal tile shape is a process of experimentation. The best shape depends on the GPU architecture (e.g., the amount of shared memory per SM), the data types used, and the specific dimensions of the matrices in your workload. The CUTLASS profiler is a practical tool for sweeping through different tile shapes to find the best one empirically. A tile shape tuned for one GPU and one matrix size may not perform as well on another GPU, another shape, or with fewer free SMs, so tune across the shapes and GPUs the kernel will actually run on, not just the case you care about.

-----

###### 304 Chapter 11 CUTLASS

High-performance kernels almost always use software pipelining to hide the latency of loading data from global memory. While the warps are busy computing with one set of tiles in shared memory, the kernel is already prefetching the next set of tiles from global memory in the background. This is also known as double-buffering. The number of stages in this pipeline is a tunable parameter. In our single\_gpu\_ gemm.cu, this was set automatically. The following listing shows where the pipeline stages are configured.

| Listing 11.8 Configuring pipeline stages in CUTLASS |
|---|
| using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< ArchTag, cutlass::arch::OpClassTensorOp, ElementInput, LayoutA, AlignmentA, ElementInput, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout< static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, KernelSchedule >::CollectiveOp; StageCountAutoCarveout tells CUTLASS to choose the number of pipeline stages based on the amount of shared memory available after reserving space for the epilogue's needs. More stages can improve the kernel's ability to hide latency, but each additional stage requires more shared memory. For recent architectures like Hopper, which have hardware support for asynchronous memory transfers (Tensor Memory Accelerator), deeper pipelines are often beneficial. The epilogue is where CUTLASS becomes especially useful. It is the set of operations applied to the accumulated GEMM result before that result is written to global memory. By fusing work into the epilogue, we can remove entire kernel launches. Our example already uses a simple epilogue for the standard D = alpha * A * B + beta * C operation. The following listing shows that definition. |
| Listing 11.9 Basic CUTLASS epilogue configuration |

```
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
ElementOutput, ElementCompute, ElementOutput, ElementCompute,
cutlass::FloatRoundStyle::round_to_nearest>;
```

But we can easily replace this with a more complex sequence. For example, to fuse a bias + ReLU operation, we could use CUTLASS's prebuilt fusion operators, as shown in the following listing.

-----

###### Architecture and precision 305

| Listing 11.10 Using a fused bias + ReLU epilogue |
|---|
| #include "cutlass/epilogue/fusion/operations.hpp" #include "cutlass/epilogue/thread/activation.h" using FusionOp = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct< cutlass::epilogue::thread::ReLu, ElementOutput, ElementCompute>; LinCombPerRowBiasEltAct reads as "linear combination, per-row bias, elementwise acti- vation": it computes alpha * acc + beta * C, adds a bias vector broadcast across each row, and then applies the activation functor we pass as the first template argument, here ReLu. Swapping this FusionOp into the CollectiveEpilogue builder takes one more change. The EpilogueScheduleAuto policy from our baseline kernel supports only the plain linear-combination epilogue; CUTLASS rejects any other fusion operator with a com- pile-time error that tells us to pick a TMA warp-specialized schedule. The fix is to name the schedules explicitly: cutlass::epilogue::TmaWarpSpecializedCooperative in the epilogue builder, paired with cutlass::gemm::KernelTmaWarpSpecializedCooperative as the main loop kernel schedule. With that pair in place, the builder generates a ker- nel that performs the entire matmul-bias-ReLU sequence in a single launch, keeping intermediate data in the GPU's fastest memories. This is one of the main reasons to choose CUTLASS over a precompiled library like cuBLAS when a standard library call sequence is no longer a good fit. The same high-level GEMM structure must adapt to different hardware generations and precision formats. To understand these nuances, we need to compare the pro- gramming model across architectures and then narrow in on how Blackwell-era low-precision support changes the design space. CUTLASS targets several generations of GPU hardware from one library. To see this in action, let's compare our modern CUTLASS 3.x-style kernel with an official CUTLASS example written in the older 2.x style for the previous generation Ampere architecture. Evolution from CUTLASS 2.x to 3.x The following listing shows the CUTLASS 2.x style used for Ampere/A100 GPUs. |
| Listing 11.11 Ampere-style CUTLASS 2.x kernel definition |

```
using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using Gemm = cutlass::gemm::device::Gemm<
```

-----

###### 306 Chapter 11 CUTLASS

```
ElementA, LayoutA,
ElementB, LayoutB,
ElementC, LayoutC,
ElementAccumulator,
cutlass::arch::OpClassTensorOp,
cutlass::arch::Sm80,
ThreadblockShape,
WarpShape,
InstructionShape,
EpilogueOp,
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
3
>;
```

By contrast, the modern CUTLASS 3.x style (used for Hopper/H100 and Blackwell/ B200 GPUs) is more declarative and composable, as the following listing demonstrates.

**Listing 11.12 CUTLASS 3.x declarative kernel setup**

```
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;
using CollectiveMainloop = typename
cutlass::gemm::collective::CollectiveBuilder<
cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
ElementA, LayoutA, AlignmentA,
ElementB, LayoutB, AlignmentB,
ElementAccumulator,
TileShape, ClusterShape,
cutlass::gemm::collective::StageCountAutoCarveout<
static_cast<int>(sizeof(typename
CollectiveEpilogue::SharedStorage))>,
cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
using CollectiveEpilogue = typename
cutlass::epilogue::collective::CollectiveBuilder<
cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
TileShape, ClusterShape,
cutlass::epilogue::collective::EpilogueTileAuto,
ElementAccumulator, ElementAccumulator,
```

| ElementC, | LayoutC, | AlignmentC, |
|---|---|---|
| ElementD, | LayoutD, | AlignmentD, |

```
cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue
>;
```

The evolution from CUTLASS 2.x to 3.x represents a shift from explicit, monolithic templates to declarative, composable building blocks. The older style required spelling

-----

###### Architecture and precision 307

out every detail (ThreadblockShape, WarpShape, InstructionShape), while the newer style describes the what (tile shapes, data types) and lets the CollectiveBuilder handle the how. This abstraction enables better performance portability, allowing the same high-level kernel description to automatically use architecture-specific optimizations like Hopper's Tensor Memory Accelerator or Blackwell's FP4 support.

### 11.5.2 Narrow precision operations: FP4 on Blackwell

As model size and complexity increase, the demand for computational efficiency grows quickly. One of the most promising approaches is to reduce the precision of computations while maintaining accuracy through careful scaling techniques. NVIDIA's Blackwell architecture introduces native support for FP4 operations.

###### FP4 technology and Blackwell architecture

Modern deep learning workloads use a "precision ladder" that trades off accuracy for performance. FP32 offers full precision and excellent accuracy but is memory and compute intensive. FP16/BF16 provides half precision and is widely used for training and inference. FP8 uses 8-bit floating point with E4M3/E5M2 formats and is emerging as the standard for inference. FP4 uses just 4 bits per value with 1 sign bit, 2 exponent bits, and 1 mantissa bit, enabling 2× the throughput of FP8 and 4× that of FP16 (each precision halving roughly doubles tensor core throughput, as the B200 rows of table 7.1 shows), though it requires sophisticated scaling techniques to maintain numerical stability. FP4 relies on block-scaled computation. Instead of using a single global scaling factor, which would crush the dynamic range, matrices are divided into small blocks, each with its own scaling factor stored in a dedicated 8-bit format: MXFP4 pairs 32-element blocks with a UE8M0 power-of-two scale, and NVFP4 pairs 16-element blocks with a UE4M3 scale. The Blackwell SM100 architecture introduces hardware specifically designed for narrow-precision operations. The TCGen05 tensor cores provide native FP4 support with block-scaled operations. Tensor Memory (TMEM) offers per-SM memory that improves data locality for tensor operations. Warp-specialized execution allows different warps to handle different aspects of the computation. Dynamic cluster scheduling enables efficient coordination across multiple SMs.

###### Implementing FP4 GEMMs with CUTLASS

The FP4 example adapts our unofficial GEMM kernel for Blackwell's native FP4 support. The following listing highlights the key code changes from unofficial/single\_ gpu\_gemm.cu.

**Listing 11.13 Blackwell FP4 kernel modifications**

```
using ElementInput = cutlass::float_e2m1_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = cutlass::bfloat16_t;
using LayoutA = cutlass::detail::BlockScaledLayout<
```

-----

###### 308 Chapter 11 CUTLASS

```
cutlass::layout::RowMajor, cutlass::bfloat16_t>;
using LayoutB = cutlass::detail::BlockScaledLayout<
cutlass::layout::ColumnMajor, cutlass::bfloat16_t>;
using ArchTag = cutlass::arch::Sm100;
BlockScaledLayout automatically handles per-block scaling factors, enabling FP4 preci-
```

sion while maintaining numerical stability. This replaces manual scaling with hardwareaccelerated block-wise operations. The following listing shows the commands required to build and run the FP4 samples.

| Listing 11.14 Commands for building and running the FP4 examples |
|---|
| cd book.cu/9_cutlass/fp4/ # build.sh: Clones CUTLASS and builds kernel for SM100 ./build.sh # verify.sh: Tests correctness on 1024³ matrices ./verify.sh # benchmark.sh: Measures performance on 8192³ matrices ./benchmark.sh FP4 performance and use cases The following listing shows sample benchmark output highlighting FP4 throughput on Blackwell. |
| Listing 11.15 Sample FP4 benchmark output |

```
==== nvFP4 GEMM Benchmark
Problem size: 8192 x 8192 x 8192 (batch=8)
Architecture: SM100 (Blackwell)
Precision: FP4 (A, B) x BF16 (C, D)
```

```
==== Results
Average kernel time: 1.93 ms
Performance: 4.55 PFLOPS
Performance: 4553 TFLOPS
```

FP4 is not a universal replacement for higher-precision formats. This level of performance makes it viable for memory-bound workloads where data movement dominates execution time, large models that are challenging to fit in memory, inference optimization where accuracy can be maintained through quantization techniques, and specialized training scenarios where reduced-precision training is acceptable. The Blackwell architecture's native FP4 support makes these use cases increasingly practical for production deployments.

-----

###### Profiling and verification 309

## 11.6 Profiling and verification

To measure a kernel's performance and check its correctness, CUTLASS ships a command-line tool, the CUTLASS Profiler. The profiler is an executable (cutlass\_profiler) that lives in the tools/profiler directory of the CUTLASS build. It can launch and benchmark any kernel that has been instantiated in the CUTLASS library. Its basic usage is straightforward, as shown in the following listing.

| Listing 11.16 Using cutlass_profiler from the command line |
|---|
| ./tools/profiler/cutlass_profiler --kernel=<kernel_name> --m=<M> --n=<N> --k=<K> This command runs the specified kernel for the given problem size (M, N, K) and reports its performance, typically in TFLOP/s. The profiler's real power comes from its ability to sweep through many options to help you find the best configuration for your needs.You can give the profiler a problem size and ask it to run all available GEMM kernels to find the fastest one.You can provide a list of thread block tile shapes to test, and the profiler will benchmark each one and report the winner. The profiler automatically runs a CPU-based reference implementa- tion to verify that the GPU results are correct within a given tolerance. Our unofficial/single_gpu_gemm.cu example contains its own simple benchmark- ing and verification logic within its main() function, which is great for a self-contained example. The following listing shows that helper. |
| Listing 11.17 Benchmark and verification helper inside the sample GEMM |

```cpp
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < bench_iters; i++) {
gemm_op.run();
}
CUDA_CHECK(cudaDeviceSynchronize());
auto end = std::chrono::high_resolution_clock::now();
double time_ms = std::chrono::duration<double, std::milli>(end - start).
count() / bench_iters;
double tflops = (2.0 * M * N * K) / (time_ms * 1e9);
std::cout << "Average time: " << time_ms << " ms" << std::endl;
std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
```

The official examples take this a step further. They are compiled into PyTorch extensions and benchmarked using a Python script (benchmark.py). This is a common workflow when you need a custom CUDA/CUTLASS kernel inside a larger Python-based deep learning framework. For standard operators, the framework path should remain the default. The CUTLASS extension path is for cases where kernel-level control

-----

###### 310 Chapter 11 CUTLASS

justifies the extra maintenance. The script creates random tensors, calls the CUTLASS kernel, and compares its performance against PyTorch's built-in torch.matmul, which typically calls cuBLAS under the hood.

## 11.7 Beyond a single GPU: Distributed GEMM

So far, we have focused on maximizing the performance of a GEMM on a single GPU. Chapter 10 covered the communication side of multi-GPU execution with NCCL and parallelism strategies; now we connect those ideas back to CUTLASS. CUTLASS provides the building blocks for this world as well, enabling the construction of distributed *GEMMs that can use the combined power of multiple GPUs within a node.*

The core idea behind a distributed GEMM is to shard the matrices across GPUs and have each GPU compute a portion of the result. This approach requires careful coordination and communication, typically using a library like NVIDIA's NCCL to exchange data between GPUs over high-speed interconnects like NVLink.

One point of terminology first: the ClusterShape we saw earlier has nothing to do with multiple GPUs. It configures the thread block cluster, a group of thread blocks co-scheduled on neighboring SMs of a single GPU so they can share distributed shared memory and multicast TMA loads. Multi-GPU GEMM is a separate, experimental CUTLASS feature: Distributed GEMM (examples/65\_distributed\_gemm for Hopper,

```
examples/82_blackwell_distributed_gemm for Blackwell), which parameterizes a ker-
```

nel by its tensor-parallel rank and overlaps communication with computation. The book.cu/9\_cutlass/unofficial/multi\_gpu\_gemm.cu and book.cu/9\_cutlass/official/ official\_multi\_gpu\_gemm/ examples provide a simpler starting point, showing how to initialize NCCL communicators and launch a kernel on multiple devices.

There are many strategies for distributing a GEMM, such as splitting the M dimension, the N dimension, or both, each with different communication patterns and performance characteristics. CUTLASS provides the flexibility to implement these strategies, such as AllGather + GEMM, where each GPU has a shard of the input matrix

```
A, and an AllGather collective is used to collect the full matrix on every GPU before a
```

local GEMM is performed, or GEMM + ReduceScatter, where each GPU computes a partial result and a ReduceScatter collective is used to sum the partial results and distribute the final output shards.

Building efficient distributed kernels is a complex topic that goes beyond the scope of this chapter, but it's important to know that the same principles of tiled decomposition and layout-centric design that we've learned apply here as well, extended across the new dimension of multiple devices.

-----

###### Summary 311

###### Summary

¡ CUTLASS is a C++ template library, not a binary like cuBLAS. It provides building blocks for GEMM kernels with custom tiling, fused epilogues, and explicit control over data movement. ¡ CUTLASS decomposes a GEMM hierarchically, partitioning it into tiles processed at the thread block, warp, and thread levels, keeping data resident in shared memory and registers rather than round-tripping through DRAM.

```
¡ CollectiveBuilder (CUTLASS 3.x) composes a kernel from a Mainloop and an
Epilogue; the host launches it through GemmUniversalAdapter and an Arguments
```

struct that packages problem dimensions, device pointers, and strides. ¡ CuTe is the layout system underlying modern CUTLASS. It separates a tensor's

logical shape from its physical strides and supports compile-time layout algebra such as slicing, tiling, and partitioning.

```
¡ TileShape is the most important tuning knob and trades arithmetic intensity
```

against shared-memory and register pressure. Pipeline staging (e.g., StageCount-

```
AutoCarveout) hides global-memory latency by prefetching the next tile while
```

the current tile is being computed. ¡ Epilogue fusion (e.g., LinCombPerRowBiasEltAct) folds bias additions and activations into the GEMM kernel, eliminating the extra launches and DRAM round trips that a naive matmul → bias → ReLU sequence would incur. ¡ CUTLASS 2.x targets Ampere with explicit ThreadblockShape, WarpShape, and

```
InstructionShape declarations. CUTLASS 3.x targets Hopper and newer with
```

declarative tile and cluster shapes that let CollectiveBuilder pick schedules specific to the architecture. ¡ Blackwell FP4 kernels use block-scaled layouts with per-block 8-bit scale factors

(UE8M0 for MXFP4, UE4M3 for NVFP4) to maintain numerical stability while running at FP4 throughput on TCGen05 tensor cores. ¡ `cutlass_profiler benchmarks` any instantiated kernel and sweeps tile-shape

configurations against a CPU reference for correctness; self-contained harnesses inside the sample GEMMs measure TFLOPS over many iterations using

```
chrono::high_resolution_clock.
¡ ClusterShape configures the on-GPU thread block cluster, not multi-GPU exe-
```

cution. Multi-GPU GEMMs come from CUTLASS's experimental distributed GEMM examples or from sharding by hand, exchanging shards via NCCL collectives such as AllGather + GEMM or GEMM + ReduceScatter over NVLink.

-----

# 12

## Now what?

---

| This chapter covers |
|---|
| ¡ Topics we deliberately skipped and when they matter |
| ¡ When to stay above CUDA and when to drop |
| below frameworks |
| ¡ Serving systems and why end-to-end |
| performance matters |
| ¡ Compiler and generated-kernel workflows |
| ¡ Hardware targets beyond data center NVIDIA GPUs |
| ¡ How to keep learning without chasing every new tool |
| ¡ Resources for keeping pace with the ecosystem |
|  |

By this point, you've gone from writing your first cudaMalloc to implementing tensor core operations, fused attention kernels, quantization strategies, and multi- GPU distributed systems. You can now read CUDA code in the wild, understand the kernel-level decisions, and write your own kernels when needed.

I kept this book deliberately narrow in scope. We focused on CUDA, the fundamentals, and the building blocks that do not change every few months. I did not try

-----

###### What this book intentionally left out 313

to cover every framework, library, serving stack, compiler, or hardware target in the AI ecosystem because that would have obscured the core ideas.

This final chapter is about what comes next. Some of that is technical: which tools to learn, which abstractions to trust, and where custom CUDA still matters. Some of it is professional: how to stay useful as compilers, generated kernels, and production inference systems absorb more of the work that used to require handwritten kernels. Think of this as a road map for continued study, not a checklist to finish immediately.

## 12.1 What this book intentionally left out

This book stayed close to the CUDA programming model: kernels, memory hierarchy, synchronization, tensor cores, fusion, quantization, and multi-GPU communication. That means we left out several topics that matter in real systems but sit one layer above, beside, or beyond the core lesson.

Once you know the CUDA foundation, you can decide whether a new tool is hiding the same constraints behind a cleaner interface, solving a genuinely different systems problem, or targeting hardware with a different programming model.

###### 12.1.1 Data center-level considerations

Chapter 10 stayed within one multi-GPU server. Scaling that work across servers requires infrastructure that sits outside CUDA itself. If you are operating a GPU cluster with hundreds or thousands of GPUs, you need cluster orchestration tools, such as SLURM for job scheduling, Kubernetes with Kubeflow for containerized machine learning workloads, or Ray for distributed computing with autoscaling and fault tolerance. These tools manage resource allocation, queues, and job coordination across shared clusters.

The network topology becomes critical at this scale. You are dealing with fat-tree networks that remove the oversubscription conventional hierarchies introduce above the rack, dragonfly networks at the tens-of-thousands-of-nodes scale, and rail-optimized designs (a layer on top of a fat-tree) where GPUs are grouped by InfiniBand "rail" to minimize cross-rail traffic. A poor match between topology and workload can turn the interconnect into the bottleneck and erase much of the benefit of scaling out.

When things go wrong, you need monitoring and debugging tools. DCGM (Data Center GPU Manager) handles GPU health monitoring, while Prometheus and Grafana provide cluster-wide metrics visualization. NCCL debugging with environment variables like NCCL\_DEBUG=INFO helps diagnose communication problems.

Fault tolerance becomes essential. You need checkpointing every N iterations so you can resume from failures, elastic training frameworks such as PyTorch Elastic and Horovod Elastic that let you dynamically add or remove nodes without restarting, and redundant communication paths for critical collectives. Cost optimization also matters: spot instances with preemptible cloud GPUs can reduce costs substantially if you checkpoint aggressively. Mixed-precision training (FP8/FP16) reduces memory and bandwidth requirements, while gradient compression reduces AllReduce volume, though it can hurt accuracy.

-----

###### 314 Chapter 12 Now what?

These topics are beyond the scope of this book, but they're essential for production deployments. If you want to experiment with distributed training yourself, I recommend checking out llm.c (https://github.com/karpathy/llm.c), Andrej Karpathy's educational implementation of distributed large language model (LLM) training. It demonstrates multi-GPU training with NCCL, mixed precision, gradient accumulation, and checkpointing, all in clean, readable C/CUDA code without framework abstractions. It's exactly the kind of"just-in-time" learning resource that follows the philosophy of this book.

### 12.1.2 Convolution-heavy architectures

We focused heavily on transformers, MLPs, and matrix multiplication. That matches a lot of modern language model work, but it is not the whole field of deep learning. Convolutional neural networks still matter for image classification, detection, segmentation, robotics, medical imaging, and many edge workloads. Diffusion models and multimodal models often mix attention with convolution-like image operations, resampling, normalization, and layout conversions.

Those architectures rely on different kernels, such as conv2d, maxpool2d, batchnorm2d, resize operators, and image/tensor layout transforms, but the intuition you built in this book transfers directly. You implemented conv2d and maxpool2d in chapter 3 as practice kernels. BatchNorm is similar to LayerNorm, but it reduces and normalizes across different dimensions. Modern architectures mostly reorganize these kernel families with different shapes, schedules, and data flows.

If you understand GEMM optimization, as you do after chapter 6, you can also understand modern convolution implementations. Frameworks often lower convolutions into matrix-multiplication-like problems using techniques such as `im2col,` implicit GEMM, or Winograd transforms. The reduction concept you learned in softmax applies to maxpool but across a 2D window instead of a 1D sequence. The CUDA skill is recognizing the same memory and parallelism problems when the tensor shapes change.

### 12.1.3 The Blackwell TCGen05 instructions

In chapter 7, we covered tensor cores across multiple generations: MMA and WMMA, introduced with Volta (and extended with new precisions on Turing and Ampere), and then WGMMA for Hopper. But we stopped before Blackwell's TCGen05 instructions, the next-generation tensor core programming model that supports FP4 and replaces WGMMA with improved efficiency.

I chose to omit these topics because raw TCGen05 programming adds complexity without teaching new concepts. The programming patterns you learned with MMA and WGMMA transfer directly to TCGen05. The core concepts, including fragment layouts, tile hierarchies, and asynchronous operations, remain the same. Only the specific instructions change. For readers who want to target Blackwell (B200 GPUs are available on AWS, GCP, and Azure), CUTLASS 4.x provides the cleanest path.

-----

###### Choosing the right abstraction layer 315

## 12.2 Choosing the right abstraction layer

Everything in this book has been about writing CUDA kernels from scratch. In practice, many workloads should start higher. If PyTorch, torch.compile, Triton, CUTLASS, or an inference server already gives you the performance and control you need, handwritten CUDA is not automatically the nobler choice.

The question is, "Where is the bottleneck, and which layer gives me enough control to move it?" Stay high-level when the profiler shows that most time is spent in well-optimized library calls, when development speed matters more than squeezing out the last few percent, or when the workload changes too often for a custom kernel to stay useful. Drop lower when the profiler points to a specific operator, memory layout, synchronization pattern, or fusion opportunity that the higher-level stack cannot express.

### 12.2.1 Triton: Python-level GPU programming

Triton (https://github.com/triton-lang/triton) is a Python-based domain-specific language (DSL) for writing GPU kernels. It raises the abstraction level so you can work at the tile or block level instead of spelling out every low-level CUDA detail yourself.

You write kernels in Python at the block level (think: one block of threads), not at the thread level. Triton handles much of the scheduling and memory-management boilerplate for you, which lets you focus more on the algorithm and less on the mechanics of CUDA.

Triton is also hardware agnostic. Your Triton code officially targets NVIDIA GPUs (compute capability 8.0+) and AMD GPUs (ROCm 6.2+), with a CPU backend under development and third-party out-of-tree backends extending it to other accelerators such as neural processing units (NPUs). The promise is, "Write once, run anywhere," though the real portability story still depends on backend maturity. With compiler-initiated optimizations and developer-initiated fusion, Triton kernels can approach the performance of tuned CUDA kernels without forcing you to manage the memory hierarchy directly.

The integration with PyTorch is particularly compelling. With torch.compile and TorchInductor, PyTorch can automatically generate optimized kernels for your operations, often using Triton on GPU backends. You often do not have to write the Triton code yourself.

This is still a CUDA book, not a Triton book. Triton matters because it is a major tool in the ecosystem, and learning raw CUDA first makes you better at using Triton. You understand what the compiler is doing under the hood, when it helps, and when you may need to intervene.

The practical test is simple: start from a PyTorch or NumPy-like reference; write the Triton version if the framework output is not good enough; then compare correctness and latency across the shapes you actually care about. If the performance gain appears for only one tiny benchmark shape, keep digging before treating it as a production win.

-----

###### 316 Chapter 12 Now what?

### 12.2.2 torch.compile

PyTorch 2.0 introduced torch.compile, which uses TorchDynamo to capture Pythonlevel PyTorch programs and TorchInductor to generate optimized code. The following listing shows the basic usage pattern.

```python
Listing 12.1 torch.compile example
@torch.compile
def my_model(x):
return torch.nn.functional.gelu(x @ weights)
```

For many practitioners, this is becoming the default first optimization step: write highlevel PyTorch and then let the compiler generate faster kernels and fuse operations where it can. But knowing how CUDA kernels work makes you a better user of torch

```
.compile. You understand why shape changes can cause recompilation, why graph
```

breaks matter, why fusion helps, and why a generated kernel can still be limited by memory bandwidth. The job-relevant skill is knowing how to profile before and after, inspect whether the compiler actually captured the hot path, and decide whether the next step should be a model rewrite, a Triton kernel, a CUTLASS kernel, or no custom kernel at all.

## 12.3 Serving systems and end-to-end performance

Once you've trained or downloaded a model, you need to serve it. In many deployed systems, inference accounts for most of the GPU time spent after the model ships. Serving is not just "running the model forward." It includes request scheduling, batching, token streaming, key-value (KV) cache management, quantization, model loading, parallelism, and API integration. This is where isolated CUDA thinking can mislead you. A single kernel might be faster in a microbenchmark, but the production system might still be bottlenecked on batching policy, memory fragmentation, model loading, network overhead, or the shape distribution of real requests. The right measurement is the end-to-end system: time to first token, intertoken latency, throughput at a target latency, GPU utilization, memory footprint, and reliability under real traffic.

### 12.3.1 vLLM

vLLM (https://github.com/vllm-project/vllm) introduced PagedAttention, which applies virtual memory concepts to KV cache management. Instead of allocating contiguous memory for the entire KV cache upfront, which leads to fragmentation and waste, vLLM uses paging to allocate memory dynamically as needed.

vLLM delivers high-throughput inference with efficient memory management through its continuous batching approach: it processes requests as they arrive rather than waiting for full batches. It supports tensor parallelism for large models and

-----

###### Serving systems and end-to-end performance 317

provides an OpenAI-compatible API that makes it straightforward to integrate into existing systems. The following listing shows the basic programming model.

| Listing 12.2 vLLM inference example |
|---|
| from vllm import LLM, SamplingParams llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct") outputs = llm.generate(prompts, sampling_params) vLLM is often the default choice for general-purpose deployments because it is rela- tively easy to get running and usually delivers strong throughput. SGLang (https://github.com/sgl-project/sglang) focuses on structured generation and efficient constrained decoding. If your application needs JSON outputs, function calling, or grammar-constrained generation, SGLang handles this efficiently. SGLang's RadixAttention automatically reuses KV cache across requests with similar prefixes. It includes an xGrammar backend for fast structured outputs, cache-aware load balancing, and a batch scheduler designed to keep GPU utilization high. The following listing shows the structured-generation style it targets. |
| Listing 12.3 SGLang structured generation example |

```python
import sglang as sgl
@sgl.function
def tool_use(s, question):
s += "To answer: " + question + ", "
s += "I need: " + sgl.gen("tool", choices=["calculator", "browser"])
```

SGLang shines when your inference workload involves structured outputs or repeated patterns that can benefit from KV cache reuse.

### 12.3.3 TensorRT-LLM

TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM) is NVIDIA's productiongrade inference engine. It is the most complex of the three, but it is also one of the highest-performance options when properly configured. TensorRT-LLM provides FP8 quantization and INT8/INT4 support with calibration, multi-GPU tensor and pipeline parallelism, and in-flight batching similar to vLLM's continuous batching. It includes custom CUDA kernels optimized by NVIDIA engineers. Think of it as cuDNN for LLMs. cuDNN is NVIDIA's long-running library of optimized primitives for deep learning operations such as convolutions, normalization, and recurrent networks. TensorRT-LLM plays a similar role for LLM inference: it packages many low-level

-----

###### 318 Chapter 12 Now what?

optimizations behind a production-oriented interface while still exposing enough configuration to matter. TensorRT-LLM can deliver excellent performance, especially on Hopper and Blackwell GPUs with FP8 support. The tradeoff is complexity: it is a lower-level tool that requires more configuration and operational understanding. Start with vLLM for general-purpose serving, because it is usually the quickest path to a strong baseline. Use SGLang if you need structured outputs or high cache reuse, particularly for applications with repeated patterns or JSON generation requirements. Use TensorRT-LLM when you need tighter low-level control and NVIDIA-specific optimization and are willing to invest more in setup and configuration. And when the model is not LLM-shaped at all, or has to ship to hardware that is not the same H100 you trained on, Open Neural Network Exchange (ONNX) is usually the cleaner export path; we cover it next.

### 12.3.4 ONNX

vLLM, SGLang, and TensorRT-LLM are LLM-shaped servers. They assume autoregressive decoding, KV caches, and token streaming. A lot of production inference is not shaped that way. Recommender models, vision encoders, speech models, classical classifiers, and the smaller "everywhere" models that run inside larger products still need to ship across framework boundaries and hardware vendors. ONNX (https://onnx.ai) is the interchange format that makes them portable. ONNX is a serialization format for computation graphs. You train in PyTorch (or JAX or TensorFlow), export the graph to .onnx, and then run it through a runtime such as ONNX Runtime, TensorRT, or a hardware-vendor backend. Because the format is shared, the same exported model can run on NVIDIA GPUs through TensorRT, on CPUs through ONNX Runtime, on mobile accelerators, and on edge silicon that does not speak PyTorch natively. The following listing shows the typical export step.

**Listing 12.4 Exporting a PyTorch model to ONNX**

`import torch` **The older dynamic\_axes argument is**

**deprecated; it works only with the legacy**

`torch.onnx.export(` **TorchScript exporter (dynamo=False).**

```
model, example_input, "model.onnx",
opset_version=17, dynamic_shapes={"input": {0: "batch"}},
)
```

ONNX is most useful when the deployment target is not an H100 sitting next to your training cluster. For an LLM serving on NVIDIA hardware, a purpose-built engine like vLLM or TensorRT-LLM will usually win on throughput. For "ship this model to many places," ONNX is often the cleanest handoff between the team that trains the model and the team that runs it.

-----

###### High-level frameworks and languages 319

## 12.4 Compiler and generated-kernel workflows

Compilers already generate kernels from high-level programs, and LLM-assisted tools can generate CUDA or Triton candidates from reference implementations. Some generated kernels will be wrong. Some will be slow. Some will be surprisingly good. This area has less friction than many other kinds of AI-assisted programming because progress is unusually measurable. A generated kernel either compiles or it does not. It either matches the reference output within tolerance or it does not. It either runs faster for a defined shape on a defined GPU or it does not. You can put the loop in a script: generate, compile, test, benchmark, profile, revise. A kernel that tops a leaderboard on one shape may lose once you include data movement, layout conversion, framework overhead, dynamic shapes, batching, memory pressure, and maintainability. Production systems care about end-to-end behavior. They also care about whether another engineer can understand and safely modify the generated code six months later. People who know only how to hand-write kernels may find more isolated benchmark work replaced. People who can verify kernels, build regression tests, profile the real bottleneck, integrate generated code into a framework, and explain the tradeoffs will be harder to replace.

## 12.5 High-level frameworks and languages

Some ecosystems change the programming model above the kernel level, deciding when computations are traced, compiled, transformed, vectorized, or lowered to accelerator code. CUDA knowledge still helps, but the right mental model is compiler aware: what graph did the framework see, what operations can it fuse, and what code did it ultimately emit? The common thread is that these tools raise the unit of optimization. Instead of optimizing one CUDA block, you may optimize a whole function, a whole graph, or a whole tensor program. The danger is that you lose visibility. The advantage is that the compiler can see patterns that are hard to express by hand at the CUDA level.

### 12.5.1 JAX

JAX (https://github.com/google/jax) is Google's functional machine learning framework. It feels like NumPy, but with automatic differentiation, just-in-time (JIT) compilation via the XLA compiler, and built-in support for GPU/TPU workflows. The following listing shows the basic style.

**Listing 12.5 JAX functional programming example**

```python
import jax.numpy as jnp
from jax import grad, jit
@jit
def loss_fn(params, x, y):
pred = jnp.dot(x, params)
```

-----

###### 320 Chapter 12 Now what?

```
return jnp.mean((pred - y) ** 2)
grad_fn = jit(grad(loss_fn))
```

JAX's approach is different from PyTorch: it's functional (no in-place operations), composable (transformations like vmap, pmap, grad), and explicitly designed for research. If you're doing anything that involves heavy use of vmap (vectorizing across batch dimensions) or pmap (parallelizing across devices),JAX is worth learning. The functional style matters because it gives the compiler a cleaner view of the computation. If a function has no hidden mutation,JAX can trace it, transform it, differentiate it, batch it, and compile it through XLA. From a CUDA perspective, this means the programmer often describes array programs while the compiler decides how to fuse operations, allocate intermediates, and lower the graph to GPU code. That does not remove the constraints you learned about in this book. If a JAX program materializes a huge intermediate tensor, moves data unnecessarily, or uses shapes that prevent efficient fusion, performance will still suffer. CUDA knowledge helps you ask better questions when a high-level program is slow.

### 12.5.2 Mojo

Mojo (https://www.modular.com/mojo) is a new programming language from Chris Lattner (creator of LLVM and Swift). It's designed to be a "Python for systems programming," combining Python's syntax and usability with C/C++ performance. Mojo is still in the early stages, so I would not treat it as required reading in the same way as CUDA, PyTorch, Triton, or an inference server. But it is worth watching because it represents a recurring goal in this space: let people write high-level code without giving up systems-level performance. If it succeeds, the same CUDA instincts still matter: memory layout, vectorization, parallelism, and data movement.

## 12.6 CUTLASS and template libraries

In chapter 11, we covered CUTLASS and the CuTe DSL, but there's more to explore here.

### 12.6.1 Python CuTe DSL

NVIDIA recently introduced a Python interface for CuTe that lets you write tensor algebra code in Python and have it compiled to optimized CUDA kernels. It is similar to Triton but more closely integrated with CUTLASS's tile abstractions. The following listing is simplified pseudocode. The real API spells these steps with calls such as cute

```
.local_tile and cute.gemm.
```

**Listing 12.6 Python CuTe DSL example**

```python
@cute.kernel
def gemm_kernel(A: Tensor, B: Tensor, C: Tensor):
# Tile-level programming with CuTe abstractions
```

-----

###### Expanding your toolkit 321

```
tile_A = cute.partition(A, tile_shape=(128, 64))
tile_B = cute.partition(B, tile_shape=(64, 128))
# Automatic scheduling and memory management
C = cute.mma(tile_A, tile_B, C)
```

This is still experimental, but it represents a useful middle ground: higher-level than raw CUDA but with direct control over tiling and tensor core usage. The FlashAttention v4 implementation is one example of this style being used for a complex kernel without dropping all the way down to handwritten CUDA.

### 12.6.2 CUTLASS profiler and auto-tuning

CUTLASS comes with a profiler that can benchmark all instantiated kernel variants and auto-tune for your specific problem size, data type, and hardware. Instead of manually choosing tile sizes and pipeline stages, you can let CUTLASS search the space for you. The following listing shows a representative command.

**Listing 12.7 CUTLASS profiler command**

```
$ ./cutlass_profiler --operation=gemm --m=2048 --n=2048 --k=2048 \
--providers=cutlass --kernels=cutlass_tensorop_*
```

The profiler will test dozens, or hundreds, of kernel configurations and report the fastest. This is invaluable when you're optimizing for a specific production workload.

## 12.7 Expanding your toolkit

These are good places to keep learning.

### 12.7.1 GPU MODE

GPU MODE (https://github.com/gpu-mode) is a community-driven initiative focused on GPU programming education. It runs weekly lectures on GPU optimization topics, maintains a Discord server with active discussions, and develops open source projects and tutorials. It is a strong starting point if you want regular discussion, examples, and practical optimization advice.

### 12.7.2 Ultra-Scale Playbook

The Ultra-Scale Playbook (https://huggingface.co/spaces/nanotron/ultrascale-playbook) is a comprehensive guide to training LLMs at scale. It covers data, tensor, pipeline, and context parallelism; memory and communication tradeoffs; and profiling methodology on real clusters. Although it focuses on training rather than inference, it is one of the best single resources for building intuition about multi-GPU scaling, and it pairs naturally with the distributed communication material from chapter 10.

-----

###### 322 Chapter 12 Now what?

### 12.7.3 LeetGPU

LeetGPU (https://leetgpu.com) is similar to LeetCode but for GPU programming. It has practice problems, solutions, and leaderboards for optimizing GPU kernels.

## 12.8 High-level inference tricks

We covered kernel-level optimization extensively, but high-level tricks also matter for inference. Speculative decoding generates multiple tokens in parallel by having a small "draft" model propose tokens and a large "verifier" model accept or reject them. Continuous batching processes requests as they arrive rather than waiting for full batches. vLLM and SGLang both build this kind of scheduling into their serving models.

Prefix caching stores the KV cache for common prefixes, such as system prompts, so every request with the same prefix reuses the same cached KV values. We covered weight quantization in chapter 9, but you can also quantize the KV cache, which saves memory and enables longer contexts, and activations, which saves bandwidth during inference. Model compilation tools like TensorRT and torch.compile can fuse operations, remove unnecessary copies, and generate specialized code for your model architecture and hardware.

These techniques can produce substantial additional speedups on top of optimized kernels, though the gains are highly workload-dependent. The vLLM documentation [(https://docs.vllm.ai) is the most practical resource for this serving-level optimization,](https://docs.vllm.ai) and Simon Boehm's blog (https://siboehm.com) is the classic walk-through of the underlying kernel-level work, taking a CUDA GEMM step by step to near-cuBLAS performance.

## 12.9 Running on edge hardware

Everything in this book has assumed data center GPUs (A100, H100, etc.). But what about running LLMs on edge devices, such as laptops, drones, and robots?

This is an active area of research. Techniques include extreme quantization FP4 and INT4 model distillation, where you train a small model to mimic a large model; structured pruning, which removes entire attention heads or layers; and hardware-specific optimization for NVIDIA Jetson, Apple M-series with Metal, and Qualcomm Hexagon digital signal processors (DSPs).

Running a small vision model on a drone requires different optimizations than running an LLM in a data center. The programming interface might be CUDA, Metal, a DSP SDK, or a vendor-specific compiler. But the fundamental questions remain the same: where is the data, how much data moves, how much parallelism exists, and which resource is saturated first?

TensorRT is a useful high-level optimization tool on NVIDIA edge.

## 12.10 Charting your next moves

The next step is not to learn every GPU framework at once. Start by identifying which layer of the stack you work in: model code, custom kernels, inference serving, compiler

-----

###### Charting your next moves 323

output, or hardware-specific deployment. Then pick tools that expose the right amount of control for that layer, and use profiling to decide when dropping lower is worth it.

### 12.10.1 The meta-point: The fundamentals generalize

When you encounter a new tensor core instruction (TCGen05), a new framework (JAX, Mojo), a new optimization technique (speculative decoding), or a new piece of production infrastructure (TensorRT-LLM), you are not starting from zero. You can read the documentation, understand the paper, and follow the code.

### 12.10.2 Where to go from here

One reasonable next step is to pick a small number of practical follow-on projects. Over the next month, join GPU Mode's Discord, ask questions, and read discussions. Implement one kernel in Triton by taking a CUDA kernel you wrote in this book, perhaps softmax or GEMM, and rewriting it in Triton. Deploy a small model with vLLM and benchmark it so you can see how inference serving works in practice.

In the next quarter, contribute to an open source project. Find a GPU library you use and fix a bug, add a feature, or improve documentation. Read FlashAttention 3's code now that you have the background from chapter 8. Optimize something you care about by profiling a real workload, identifying the bottleneck, and then deciding whether a custom kernel is actually justified.

In the next year, stay current with hardware. As new architectures emerge beyond Blackwell, as interconnect speeds increase, and as new precision formats appear, read the relevant documentation and examples. Teach someone else by writing a blog post, giving a talk, or mentoring a colleague. Build something nontrivial, whether that means training a model with your own distributed code, building a task-specific inference path, or implementing a paper that requires new kernel work.

### 12.10.3 Final thoughts

I set out to write the CUDA book I wish I had when I was learning: a book that didn't assume too much, didn't skip the hard parts, and didn't hide behind abstractions. It takes you from cudaMalloc to distributed multi-GPU inference, step by step, with code you can actually run.

The field of GPU computing and deep learning moves quickly. New hardware, frameworks, and optimization techniques will keep appearing. What changes more slowly are the underlying constraints: memory hierarchy, parallelism, data movement, and bottleneck analysis.

You have the fundamentals now.

### 12.10.4 Resources

Here are all the links from this chapter in one place:

Distributed training: ¡ [*llm.c-https://github.com/karpathy/llm.c*](https://github.com/karpathy/llm.c)

-----

###### 324 Chapter 12 Now what?

Inference serving:

¡ [*SGLang-https://github.com/sgl-project/sglang*](https://github.com/sgl-project/sglang) ¡ *TensorRT-LLM-https://github.com/NVIDIA/TensorRT-LLM* GPU programming abstractions:

¡ [*Triton-https://github.com/triton-lang/triton*](https://github.com/triton-lang/triton) ¡ *Triton documentation-https://triton-lang.org* ¡ [*JAX-https://github.com/google/jax*](https://github.com/google/jax) ¡ [*Mojo-https://www.modular.com/mojo*](https://www.modular.com/mojo) Compiler and generated-kernel workflows:

¡ [*StandardKernel-https://standardkernel.com/*](https://standardkernel.com/) Community and learning:

¡ [*GPU Mode-https://github.com/gpu-mode*](https://github.com/gpu-mode) ¡ *Ultra-Scale Playbook-https://huggingface.co/spaces/nanotron/ultrascale*

[-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) ¡ [*LeetGPU-https://leetgpu.com*](https://leetgpu.com) ¡ *How to Scale Your Model-http://jax-ml.github.io/scaling-book/* ¡ *Modal GPU Glossary-https://modal.com/gpu-glossary*

###### Summary

¡ Data center-scale GPU work adds cluster orchestration, network topology, monitoring, checkpointing, and cost management on top of CUDA and NCCL. ¡ Convolution-heavy and multimodal architectures use different operator families,

but the same memory, tiling, reduction, and parallelism concepts still apply. ¡ Triton and torch.compile raise the abstraction level, but CUDA knowledge helps

you understand generated kernels, graph breaks, fusion, recompilation, and memory bottlenecks. ¡ vLLM, SGLang, and TensorRT-LLM optimize inference serving as a system, not

just as a set of kernels. Measure end-to-end latency, throughput, memory use, and reliability before trusting isolated benchmark wins. ¡ Generated-kernel workflows are easier to verify than many other AI-assisted coding tasks because kernels can be compiled, tested for correctness, benchmarked, and profiled. Production integration is still the hard part. ¡ JAX, Mojo, CUTLASS, Python CuTe, and related tools represent different

tradeoffs between control, portability, compiler visibility, and development speed. ¡ The meta-point is that this book gave you the foundation to approach the

rest of the GPU computing space. Hardware will change and frameworks

-----

###### Summary 325

will evolve, but memory hierarchy, parallelism, and data movement remain fundamental. If you want a single concrete next step, take a model you already run, profile it end to end with Nsight Systems, find the kernel where the time actually goes, and try to make it faster, whether by fusing it, tuning it, or proving that the library version is already at the roofline.

-----

# appendix A

## Installation and setup

---

This appendix provides everything you need to set up your environment and run the code examples from this book. Whether you use a local GPU or a cloud instance, these steps will get you up and running.

| A.1 | Prerequisites Before installing anything, confirm your machine meets the hardware bar and that your OS and toolchain are supported. |
|---|---|
| A.1.1 | Hardware requirements Minimum requirements (for basic chapters): |

¡ NVIDIA GPU with CUDA support (compute capability 7.0 or later) ¡ 8 GB or more of GPU memory (VRAM) ¡ Examples: RTX 3060, RTX 3070, or newer Recommended for most chapters:

¡ NVIDIA GPU with tensor cores (compute capability 8.0 or later) ¡ 16 GB or more of GPU memory ¡ Examples: RTX 3090, RTX 4090, A100, H100 Chapter-specific requirements:

¡ *Chapters 1 to 6-GPU with compute capability 7.0 or later* ¡ *Chapter7 (WMMA tensor cores)-Volta or later (V100, RTX 20+ series)*

-----

###### Core installation 327

¡ *Chapter7 (WGMMA tensor cores)-Hopper architecture (H100)* ¡ *Chapter10 (distributed, single node)-8× GPU server (H100 recommended)* ¡ *Chapter 11 (CUTLASS)-Hopper (H100) for most examples; an SM100-based*

B200 or GB200 GPU for the FP4 example, which targets sm\_100a and TCGen05 block-scaled tensor core instructions

###### A.1.2 Software prerequisites

You will need the following on the host before installing CUDA:

¡ *Operating system-Linux (Ubuntu 24.04 LTS recommended) or WSL2 on Windows* ¡ *Python-3.10 or later* ¡ *C/C++ Compiler-A host compiler supported by your CUDA Toolkit, such as GCC*

13 on Ubuntu 24.04 ¡ *Build tools-GNU Make, CMake (optional)* Knowledge prerequisites:

¡ Intermediate Python programming ¡ Basic C/C++ understanding (we cover CUDA-specific syntax) ¡ Command line familiarity ¡ Basic understanding of linear algebra (matrix multiplication, vectors) ¡ Familiarity with deep learning concepts (optional but helpful)

###### A.2 Core installation

The install is split into three steps: the CUDA Toolkit, a Python environment, and the deep learning libraries layered on top.

###### A.2.1 Step 1: CUDA Toolkit

The CUDA Toolkit includes the NVCC compiler, core libraries such as cuBLAS, and debugging tools. The PyTorch installation in step 2 supplies cuDNN for the Python examples.

Check if CUDA is already installed:

```
nvcc --version
nvidia-smi
```

If both commands work, you can skip to verification. Otherwise, proceed with installation. If nvidia-smi fails, install a supported NVIDIA driver and reboot before continuing. The toolkit package below does not install a GPU driver.

Installation (Ubuntu/Debian):

```
# Add NVIDIA package repositories
```

➥x86\_64/cuda-keyring\_1.1-1\_all.deb

-----

**328 Appendix A** ***Installation and setup***

```
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
# Install the current CUDA 13 toolkit
sudo apt-get install -y cuda-toolkit-13-3
# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-13.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64${LD_LIBRARY_PATH:+:${LD_LI
```

➥BRARY\_PATH}} Verify installation:

```
nvcc --version # Shows the installed CUDA Toolkit version
nvidia-smi # Shows the GPU, driver, memory, and utilization
```

The exact output depends on your driver and GPU. nvidia-smi reports the newest CUDA version supported by the driver, which can differ from the toolkit version reported by nvcc.

CUDA 13 no longer generates native machine code offline for compute capability 7.0 GPUs such as V100. If you use one of those GPUs for the Volta examples, install the final CUDA 12 release instead. The rest of this appendix uses the current CUDA 13 toolchain.

###### A.2.2 Step 2: Python environment

We use uv for fast, reliable Python package management throughout this book.

Install uv (if not already installed):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment:

```
cd book.cu
uv venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```

Install Python dependencies:

```
# Core dependencies for all chapters
uv pip install torch torchvision torchaudio
# Additional packages
uv pip install numpy matplotlib pybind11 requests
```

Verify PyTorch CUDA support:

```python
uv run python -c "import torch; print(f'CUDA available:
{torch.cuda.is_available()}'); print(f'CUDA version:
{torch.version.cuda}'); print(f'Device:
{torch.cuda.get_device_name(0)}')"
```

-----

###### Cloud GPU options 329

Expected output resembles the following (PyTorch packages include their own CUDA runtime, so the minor version can differ from the separately installed toolkit):

```
CUDA available: True
CUDA version: 13.0
Device: NVIDIA H100 80GB HBM3
```

###### A.2.3 Step 3: Additional libraries

For distributed computing (chapter 10):

```
# OpenMPI and NCCL for single-node multi-GPU examples
sudo apt-get install -y openmpi-bin libopenmpi-dev libnccl2 libnccl-dev
# Verify installation
mpirun --version
```

For CUTLASS (chapter 11): CUTLASS is a header-only library, so there is nothing to build or install. Clone it and point an environment variable at the checkout:

```
git clone https://github.com/NVIDIA/cutlass.git cutlass
export CUTLASS_DIR=$(pwd)/cutlass
```

###### A.3 Cloud GPU options

If you do not have local GPU access, the services in table A.1 provide CUDA-capable instances. Hardware, regions, availability, and hourly rates change too often for a printed price table to remain accurate. Check each provider's current catalog before you create an instance.

###### Table A.1 Cloud GPU providers and where each fits

| Provider | Access model | Best fit |
|---|---|---|
| Lambda Cloud CoreWeave vast.ai Runpod Google Colab | Managed GPU instances Managed cloud infrastructure Third-party host marketplace Managed and community instances Managed notebook sessions | A simple preconfigured environment Larger or container-based workloads Flexible hardware and spot availability Container-based development Short experiments without server setup |

For the shortest setup, start with a managed instance or notebook. For chapter 10, confirm that one instance exposes all required GPUs through nvidia-smi topo -m before you rent it.

-----

# appendix B

## Running the code examples

---

All code examples live in the public book.cu repository, organized by chapter. Clone it once, and the per-chapter commands that follow will run from inside that directory:

```
git clone https://github.com/infatoshi/book.cu
cd book.cu
```

Each chapter directory is self-contained with build scripts and README files.

###### B.1 Chapter 2: Vector addition (0_vecadd/)

What you'll learn: Basic CUDA kernel launches, memory management, and thread-indexing patterns.

Build and run the following:

```
cd book.cu/0_vecadd
# Using Makefile (recommended)
make all
make run
# Or compile individually
nvcc vecadd.cu -o vecadd
./vecadd
nvcc vecadd_scalable.cu -o vecadd_scalable
```

-----

***Chapter 3: Naive neural network operations (1\_naive/)*** **331**

```
./vecadd_scalable
nvcc tensor_add_3d.cu -o tensor_add_3d
./tensor_add_3d
```

Expected output:

```
Vector addition successful!
All 8 elements match.
```

Learning path:

**1** Start with vecadd.cu, the simplest CUDA program (eight elements, eight threads). **2** Move to vecadd\_scalable.cu to handle large arrays (1 million elements). **3** Explore tensor\_add\_3d.cu for multidimensional indexing in deep learning.

###### B.2 Chapter 3: Naive neural network operations (1_naive/)

What you'll learn: CPU-first development, fundamental neural network operations, correctness verification.

Build and run the following:

```
cd book.cu/1_naive
# Build all examples
make all
# Run all with verification
make run
# Or build/run individual operations
make vector_add
./vector_add
make gemm
./gemm
make softmax
./softmax
make conv2d
./conv2d
```

Expected output:

```
CPU result sample: [12.34, 45.67, ...]
GPU result sample: [12.34, 45.67, ...]
✓ Results match! Max difference: 0.000001
GPU kernel time: 1.23 ms
```

Available operations: `vector_add.cu` (element-wise vector addition); `matrix_add.cu` (2D matrix addition); transpose.cu (matrix transposition); gemm.cu (GEMM kernels);

-----

**332 Appendix B** ***Running the code examples***

```
softmax.cu (softmax activation [naive]); conv1d.cu (1D convolution); conv2d.cu (2D
```

convolution [convolutional neural network core]); maxpool2d.cu (2D max pooling)

###### B.3 Chapter 4: MLP training on MNIST (2_mnist/)

What you'll learn: An end-to-end training pipeline from PyTorch → NumPy → CUDA.

**NOTE** This chapter has Python scripts (v0-v2), a plain C implementation (v3), and CUDA implementations (v4-v5). Run the CPU and PyTorch versions:

```
cd book.cu/2_mnist
uv run v0.py # Download and preprocess data (run this first)
uv run v1.py # PyTorch (CUDA)
uv run v2.py # NumPy from scratch
gcc v3.c -lm -o mnist_c && ./mnist_c # Plain C from scratch
```

Run CUDA implementations:

```
# Compile and run naive CUDA version (custom kernels only)
nvcc v4.cu -o mnist_cuda
./mnist_cuda
# Compile the cuBLAS-optimized version
nvcc v5.cu -o mnist_optimized -lcublas
./mnist_optimized
```

###### B.4 Chapter 5: Transformer architecture (3_transformer/)

What you'll learn: Custom CUDA kernels integrated with PyTorch autograd, training, and inference.

Setup:

```
cd book.cu/3_transformer
# Create virtual environment
uv venv
source .venv/bin/activate
# Install dependencies
uv pip install torch torchvision torchaudio pybind11 requests
# Build CUDA extensions
uv run python setup.py build_ext --inplace
```

Run training (dense only):

```
uv run train.py
```

-----

###### Chapter 6: CUDA core optimizations (4_optim/) 333

Expected output:

```
Using device: cuda
Batch size: 16, Block size: 64, Embedding dim: 128
PyTorch baseline training...
iter 0/1000 | loss 4.3512
iter 100/1000 | loss 2.5310
...
PyTorch training time: ~20s
Custom CUDA training...
iter 0/1000 | loss 4.3772
iter 100/1000 | loss 2.5535
...
Custom CUDA training time: ~15s
✓ Custom CUDA achieves similar loss with 25% speedup
```

Run inference (dense + Mixture of Experts [MoE]):

```
uv run inference.py
```

Expected output:

```
=== PyTorch Dense vs CUDA Dense ===
✓ SUCCESS: Dense implementations match exactly!
=== PyTorch MoE vs CUDA MoE ===
NOTICE: MoE implementations show expected differences
(This is educational - demonstrates numerical precision challenges)
```

Custom operations implemented:

¡ *Training-Matmul, LayerNorm, softmax, GELU; embedding with backpropaga-*

tion; element-wise operations ¡ *Inference-GEMV (optimized for single-token generation); top-k (for MoE rout-*

ing); KV caching

###### B.5 Chapter 6: CUDA core optimizations (4_optim/)

What you'll learn: Progressive kernel optimization from naive to highly optimized implementations.

Structure:

```
4_optim/
```

| ├── layernorm/ | # 3 kernels |
|---|---|
| ├── softmax/ | # 5 kernels |
| ├── gemv/ | # 5 kernels + cuBLAS |
| ├── topK/ | # 4 kernels |
| └── gemm/ | # 7 CUDA core kernels + cuBLAS |

-----

**334 Appendix B** ***Running the code examples***

Run benchmarks:

```
cd book.cu/4_optim
# Benchmark softmax optimizations
cd softmax
uv run main.py
# Output: softmax_performance.png
# Benchmark GEMV
cd ../gemv
uv run main.py
# Benchmark full GEMM optimization progression
cd ../gemm
uv run main.py
# Output: gemm_performance.png
```

Expected output:

```
=== Correctness Verification ===
✓ Kernel 0 (Naive): PASS
✓ Kernel 1 (Coalesced): PASS
✓ Kernel 2 (Shared Memory): PASS
...
=== LATENCY SUMMARY (ms) ===
```

| Shape | Naive | Coalesced | Shared | 1D-Tile | 2D-Tile |
|---|---|---|---|---|---|
| 512² | 0.124 | 0.087 | 0.042 | 0.028 | 0.018 |
| 1024² | 0.912 | 0.643 | 0.301 | 0.189 | 0.098 |
| 2048² | 7.234 | 5.102 | 2.287 | 1.432 | 0.723 |
| === SPEEDUP SUMMARY (vs PyTorch) === |  |  |  |  |  |
| Shape | Naive | Coalesced | Shared | 1D-Tile | 2D-Tile |
| 512² | 0.3x | 0.4x | 0.9x | 1.3x | 2.1x |
| 1024² | 0.3x | 0.4x | 0.9x | 1.4x | 2.8x |
| 2048² | 0.2x | 0.3x | 0.7x | 1.1x | 2.2x |

Optimization techniques covered:

¡ Memory coalescing ¡ Shared memory tiling ¡ 1D and 2D block tiling ¡ Vectorized memory access ¡ Warp-level reductions

###### B.6 Chapter 7: Tensor core programming (5_tensor_cores/)

What you'll learn: Hardware-accelerated matrix operations using WGMMA.

```
Vectorized
0.012
0.067
0.501
```

```
Vectorized
3.2x
4.1x
3.2x
```

WMMA and

-----

###### Chapter 8: Flash attention (6_flash/) 335

Hardware requirements: kernels 7-8 (WMMA): Volta or later (V100, RTX 20+ series); kernels 9-12 (WGMMA): Hopper only (H100)

Build and run the following:

```
cd book.cu/5_tensor_cores/gemm
# Build all kernels
make
# Run benchmarks
uv run main.py
```

Expected output (on H100):

```
=== Tensor Core Performance ===
```

| Kernel 7 (cuBLAS TC): | ~810 TFLOPS |
|---|---|
| Kernel 8 (WMMA): | ~450 TFLOPS |
| Kernel 9 (WGMMA): | ~790 TFLOPS |
| Kernel 10 (WGMMA Large): | ~805 TFLOPS |
| Kernel 11 (WGMMA Async): | ~815 TFLOPS |
| Kernel 12 (WGMMA Max): | ~825 TFLOPS |

```
Speedup vs best CUDA core kernel (Ch 6): 10-15x
```

Kernel progression:

**1** Kernel 7: cuBLAS with tensor cores (baseline) **2** Kernel 8: WMMA (manual tensor core programming) **3** Kernel 9: WGMMA (Hopper asynchronous programming) **4** Kernels 10-12: WGMMA optimizations (tile sizes, async loads, TMA)

###### B.7 Chapter 8: Flash attention (6_flash/)

What you'll learn: kernel fusion, online softmax, memory-efficient attention with tensor cores.

Hardware requirements: Volta or later (for WMMA tensor cores) Build and run the following:

```
cd book.cu/6_flash
# Run all implementations
uv run main.py
# Test specific kernel
uv run main.py --kernels fa
# Different configuration
uv run main.py --seq-len 1024 --batch-size 8
```

-----

**336 Appendix B** ***Running the code examples***

| Expected output (on H100, B | = 16, H = 8, | N = 512, D = | 64): |
|---|---|---|---|
| Implementation | Time (ms) ------------------------------------------------------------ | Speedup | Notes |
| PyTorch Flash (bf16) | 0.087 | 1.00x | Reference |
| PyTorch Naive (f32) | 0.465 | 0.19x | 3 separate matmuls |
| fa (fp16 + WMMA) | 5.314 | 0.02x | Custom tensor cores |
| naive (fp32) | 68.832 ✓ Custom FA achieves 13x speedup over naive baseline! | 0.00x | 3 separate kernels |

Implementations:

```
¡ naive.cu-Three separate kernels (Q@K^T, softmax, S@V)
¡ fa.cu-Fused Flash Attention with WMMA tensor cores
```

Key concepts:

¡ Tiling (Br = 16, Bc = 16) ¡ Online softmax (no N×N materialization) ¡ Mixed precision (FP16 input, FP32 accumulation) ¡ WMMA 16 × 16 × 16 tiles (the most common shape; 32 × 8 × 16 and 8 × 32 × 16

also exist)

###### B.8 Chapter 9: Quantization (7_quant/)

What you'll learn: FP32/FP16 → INT8/INT4 quantization, calibration, different granularity schemes.

Build and run all examples:

```
cd book.cu/7_quant
# Compile all
nvcc -O3 -o fp32_int8 fp32_int8.cu && ./fp32_int8
nvcc -O3 -o fp32_int4 fp32_int4.cu && ./fp32_int4
nvcc -O3 -o dynamic_static dynamic_static.cu && ./dynamic_static
nvcc -O3 -o int8_uint8 int8_uint8.cu && ./int8_uint8
nvcc -O3 -o calibrator calibrator.cu && ./calibrator
nvcc -O3 -o tensorwise tensorwise.cu && ./tensorwise
nvcc -O3 -o groupwise groupwise.cu && ./groupwise
nvcc -O3 -o blockwise blockwise.cu && ./blockwise
nvcc -O3 -o channelwise channelwise.cu && ./channelwise
```

Or use a Makefile:

```
make all
make run
```

Expected output:

```
=== FP32 → INT8 Quantization ===
Scale: 0.0392157
```

-----

***Chapter 10: Distributed computing (8\_distributed/)*** **337**

```
Original: [12.45, -8.23, 5.67, ...]
Quantized: [127, -84, 58, ...]
Dequantized: [12.42, -8.20, 5.64, ...]
Max error: 0.03, Mean error: 0.01
Memory savings: 4x
```

Fundamental concepts covered:

**1** fp32\_int8.cu: Basic symmetric quantization **2** fp32\_int4.cu: 8× compression with packing **3** dynamic\_static.cu: Runtime versus precomputed scales **4** int8\_uint8.cu: Signed versus unsigned storage (symmetric pins the zero-point

at 0, asymmetric shifts it; either can use int8 or uint8, but the common pairing is signed int8 for zero-centered weights, unsigned uint8 for one-tailed activations) **5** calibrator.cu: Min-max versus percentile calibration **6** tensorwise.cu: Single scale per tensor **7** groupwise.cu: Group-based scales **8** blockwise.cu: Spatial block scales **9** channelwise.cu: Per-channel/feature scales

###### B.9 Chapter 10: Distributed computing (8_distributed/)

What you'll learn: Tensor parallelism, pipeline parallelism, and multi-GPU coordination.

Hardware requirements: 8× GPU server (H100 recommended): Setup:

```
cd book.cu/8_distributed/setup_scripts
chmod +x node_setup.sh
./node_setup.sh
```

Run tensor parallelism (automated):

```
cd book.cu/8_distributed/tensor_parallel
chmod +x run_all.sh
./run_all.sh
```

Run tensor parallelism (manual):

```
cd tensor_parallel
# 8 GPUs, single node
make 8gpu
mpirun -np 8 --mca btl tcp,self ./8gpu_single_node
```

-----

**338 Appendix B** ***Running the code examples***

Expected output (eight GPUs, single node):

```
=== 8-GPU Single-Node Tensor Parallel Results ===
Rank 0: ~770000 GFLOPS, ~0.18 ms
Rank 1: ~770000 GFLOPS, ~0.18 ms
...
TOTAL: ~6,160,000 GFLOPS
Avg per GPU: ~770,000 GFLOPS
Efficiency: ~100%
```

Run pipeline parallelism:

```
cd book.cu/8_distributed/pipeline
make
./run_benchmark.sh
```

Expected output:

| === | Naive | pipeline | (blocking) === |
|---|---|---|---|
| Throughput: | ~110 | batches/s |  |

```
Speedup: 1.1x
Efficiency: 27%
=== Optimized pipeline (async with streams) ===
Throughput: ~396 batches/s
Speedup: 7.8x
Efficiency: 98%
✓ CUDA streams enable near-linear scaling!
```

###### B.10 Chapter 11: CUTLASS (9_cutlass/)

What you'll learn: Production-ready tensor core programming with the CUTLASS library.

Hardware requirements: Hopper (H100) is recommended for advanced features. Official CUTLASS distributed GEMM (two GPUs):

```
cd book.cu/9_cutlass/official/official_multi_gpu_gemm
# Performance-only benchmark (recommended)
uv run perf_only.py
# Custom problem size
uv run perf_only.py --m=4096 --n=4096 --k=4096 --iterations=50
# Full benchmark with verification
uv run benchmark.py
```

-----

###### Chapter 11: CUTLASS (9_cutlass/) 339

Expected output (2× H100):

```
=== 2-GPU Distributed GEMM Performance ===
```

| Problem Size | Kernel | Time | TFLOPS | Per-GPU TFLOPS |
|---|---|---|---|---|
| 4096³ | 3.81 | ms | 36.09 | 18.04 |
| 8192³ | 5.89 | ms | 186.70 | 93.35 |
| Speedup | vs | single-GPU cuBLAS: | 1.9x |  |

```
Efficiency: 95%
```

FP4 quantization examples:

```
cd book.cu/9_cutlass/fp4
uv run main.py
```

-----

# appendix C

## Troubleshooting and development practices

---

###### C.1 Troubleshooting

This section collects the failures that bite most often, grouped by where they show up: installation, compilation, runtime, PyTorch, distributed execution, and performance.

###### C.1.1 CUDA installation problems

Problem: nvcc: command not `found` Solution:

```
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Make permanent (add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >>
~/.bashrc
source ~/.bashrc
```

Problem: CUDA error: no CUDA-capable device detected Solution:

```
# Check GPU is detected
lspci | grep -i nvidia
```

-----

###### Troubleshooting 341

```
# Check NVIDIA driver
nvidia-smi
# If the driver is missing, install the recommended Ubuntu package
sudo ubuntu-drivers install
sudo reboot
```

###### C.1.2 Compilation errors

Problem: fatal error: cuda\_runtime.h: No such file or directory Solution: Ensure that the CUDA Toolkit is installed and nvcc is in your PATH (see appendix A, section A.2.1). Problem: fatal error: mpi.h: No such file or directory (chapter 10) Solution:

```
sudo apt-get install -y openmpi-bin libopenmpi-dev
```

Problem: undefined reference to cublasCreate Solution: Add -lcublas to your nvcc command:

```
nvcc your_file.cu -o output -lcublas
```

###### C.1.3 Runtime errors

Problem: CUDA error: out `of memory` Solution:

¡ Reduce batch size or matrix dimensions in the code. ¡ Close other GPU-using applications. ¡ Check available memory: nvidia-smi. Problem: CUDA error: invalid device ordinal Solution: You're trying to use more GPUs than are available. Check:

```
nvidia-smi --list-gpus | wc -l
```

Ensure MPI ranks match available GPUs. Problem: cudaErrorInsufficientDriver or version mismatch Solution: Update the NVIDIA driver:

```
sudo apt-get update
sudo ubuntu-drivers install
sudo reboot
```

###### C.1.4 PyTorch problems

Problem: torch.cuda.is\_available() returns False Solution: Compare the toolkit and PyTorch runtime versions:

-----

**342 Appendix C** ***Troubleshooting and development practices***

```
# Reinstall the current PyTorch packages
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio
```

The major versions must be compatible. The minor versions can differ because PyTorch packages include their own CUDA runtime.

```python
nvcc --version
uv run python -c "import torch; print(torch.version.cuda)"
```

Problem: RuntimeError: CUDA error: CUBLAS\_STATUS\_EXECUTION\_FAILED Solution: This usually means something around the matmul failed, not the math itself. The common causes: the GPU ran out of memory while cuBLAS allocated its workspace, a driver/CUDA-library version mismatch, or an unsupported GPU architecture for the installed build. Check free memory first:

```python
----
import torch
print(torch.cuda.memory_summary()) # look for near-full memory
torch.cuda.empty_cache()
----
```

If memory is fine, verify the driver and PyTorch CUDA versions match (see the versionmismatch entry above).

###### C.1.5 MPI and distributed problems

Problem: mpirun: command not found Solution:

```
sudo apt-get install -y openmpi-bin libopenmpi-dev
which mpirun # Verify installation
```

Problem: NCCL initialization hangs or times out Solution: Run with debug logging and confirm that every rank can see its GPU:

```
# Print NCCL setup and routing decisions
NCCL_DEBUG=INFO mpirun -np 8 ./program
# Verify the GPUs and the links between them
nvidia-smi topo -m
```

###### C.1.6 Performance debugging

Problem: Kernel runs but is very slow Solution: The debugging strategy depends on what level you're optimizing:

-----

###### Development best practices 343

¡ For kernel-level optimization (use Nsight Compute):

```
ncu --set full -o kernel_profile ./your_program
```

This provides detailed metrics for individual kernels: memory bandwidth utilization, compute throughput (TFLOPS), occupancy and warp stall reasons, register and shared memory usage, and instruction mix analysis. ¡ For application-level optimization (use Nsight Systems):

```
nsys profile --trace=cuda,nvtx -o profile ./your_program
```

Open the resulting profile.nsys-rep in the Nsight Systems GUI to analyze:

- Kernel launch timeline and dependencies
- CPU-GPU synchronization overhead
- Memory transfer bottlenecks
- CUDA stream utilization
- Multikernel coordination problems When to use each tool:
- *Nsight Compute (ncu)-Optimizing a specific kernel's memory access, compute* efficiency, or occupancy
- *Nsight Systems (nsys)-Understanding end-to-end application flow, identify-* ing the kernels that take the most time, and finding CPU-GPU coordination problems ¡ Check GPU utilization:

```
nvidia-smi dmon -s u
# Should show high GPU utilization during kernel execution
```

¡ Verify compute mode:

```
nvidia-smi -q | grep "Compute Mode"
# Should be "Default" not "Prohibited"
```

###### C.2 Development best practices

A few habits separate kernels that ship from kernels that linger in the optimization phase. The next three subsections cover the workflow, debugging tools, and testing approach worth internalizing early.

###### C.2.1 Kernel development workflow

**1** Implement CPU version first

- Write clear, sequential code

-----

**344 Appendix C** ***Troubleshooting and development practices***

- Verify correctness with small test cases
- Use this as your reference **2** Parallelize naively
- Direct translation of CPU code to CUDA
- One thread per output element
- Focus on correctness, not performance **3** Verify correctness
- Compare GPU output with CPU reference
- Use tolerance-based checks for floating point: max\_diff < 1e-5
- Test multiple input sizes **4** Optimize incrementally
- Memory coalescing
- Shared memory tiling
- Vectorization
- Tensor cores (if applicable)
- Measure after each change **5** Profile and iterate
- Use nsys or ncu to find bottlenecks
- Focus on the slowest parts
- Know when to stop (diminishing returns)

###### C.2.2 Debugging CUDA kernels

Use CUDA\_LAUNCH\_BLOCKING for synchronous errors:

```
export CUDA_LAUNCH_BLOCKING=1
./your_program
```

This makes kernel launches synchronous, pinpointing which kernel fails. Serialize kernel launches for exact error locations: TORCH\_USE\_CUDA\_DSA is a buildtime flag: device-side assertions only exist if PyTorch itself was compiled with it, so exporting it against a stock wheel does nothing. For a stock install, use blocking launches instead:

```
----
export CUDA_LAUNCH_BLOCKING=1
uv run your_script.py
----
```

-----

###### Development best practices 345

Add debug prints in kernels:

```python
__global__ void my_kernel() {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx == 0) { // Print from one thread only
printf("Debug: value = %f\n", some_value);
}
}
```

Check for errors after CUDA calls:

```
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
printf("CUDA error: %s\n", cudaGetErrorString(err));
exit(1);
}
// After kernel launch
my_kernel<<<blocks, threads>>>(args);
err = cudaGetLastError();
if (err != cudaSuccess) {
printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}
```

###### C.2.3 Testing strategy

Unit test individual operations:

```python
# test_embedding.py
import torch
from wrapper.training import Embedding
vocab_size, embed_dim = 100, 32
indices = torch.randint(0, vocab_size, (4, 10))
embed = Embedding(vocab_size, embed_dim)
output = embed(indices)
assert output.shape == (4, 10, 32)
loss = output.sum()
loss.backward()
assert embed.weight.grad is not None
print("✓ Embedding test passed!")
```

Integration-test full pipelines:

```
# Compare custom CUDA vs PyTorch end-to-end
pytorch_loss = train_step_pytorch(model_pt, batch)
cuda_loss = train_step_cuda(model_cuda, batch)
assert abs(pytorch_loss - cuda_loss) < 0.01
print("✓ Integration test passed!")
```

-----

**346 Appendix C** ***Troubleshooting and development practices***

Performance regression tests:

```python
# Ensure optimizations don't regress
import time
start = time.time()
for _ in range(100):
my_kernel(inputs)
torch.cuda.synchronize()
elapsed = time.time() - start
assert elapsed < 1.0, f"Performance regression: {elapsed:.3f}s > 1.0s"
```

-----

# appendix D

## CUDA reference: commands, hardware, templates, and bindings

---

###### D.1 Quick command reference

Check GPU information:

```
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

CUDA compilation:

```
nvcc file.cu -o output # Basic
nvcc file.cu -o output -lcublas # With cuBLAS
nvcc file.cu -o output -lcublas -lcudnn # With cuDNN
nvcc file.cu -o output -O3 # Optimized
nvcc file.cu -o output -arch=sm_86 # Specific architecture
```

Python CUDA extension:

```
uv run python setup.py build_ext --inplace
```

MPI execution: `mpirun -np 8 ./program` **# Single node, 8 GPUs** ➥GPUs

```
mpirun -np 8 --mca btl tcp,self ./program # Force TCP
```

-----

**348 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

Profiling:

```
# Kernel-level profiling (detailed metrics)
ncu --set full -o kernel_profile ./program
# Application-level profiling (timeline view)
nsys profile --trace=cuda,nvtx -o app_profile ./program
# Memory debugging
compute-sanitizer ./program
```

###### D.2 Hardware architecture quick reference

The tables below give the memory sizes, latencies, and compute capabilities you will look up most often.

###### D.2.1 Memory hierarchy

Table D.1 summarizes the GPU memory levels from fastest to slowest, pinned to the two GPUs this book's examples run on: the RTX 3090 (Ampere, consumer) and the H100 SXM (Hopper, data center). Note that shared memory and the L1 cache are not separate pools-each SM has one block of SRAM, and CUDA lets you choose how much of it behaves as programmer-managed shared memory versus hardwaremanaged L1.

###### Table D.1 GPU memory hierarchy on the RTX 3090 and H100 SXM

**Level RTX 3090 H100 SXM Latency Scope**

| Registers | 256 KB/SM | 256 KB/SM | ~1 cycle | Per thread |
|---|---|---|---|---|
| Shared memory | 128 KB/SM (up to | 256 KB/SM (up to | ~30 cycles | Per block / |
| / L1 (one SRAM | 100 KB as shared) | 228 KB as shared) |  | per SM |
| pool) |  |  |  |  |
| L2 cache | 6 MB | 50 MB | ~260-275 cycles | Whole GPU |
| Global memory | 24 GB GDDR6X, 936 GB/s | 80 GB HBM3, 3,350 GB/s | ~400-600 cycles | Whole GPU |

NVIDIA publishes official bandwidth figures only for global memory; the on-chip tiers are best reasoned about from the hardware itself. Shared memory delivers up to 128 bytes per clock per SM (32 banks, 4 bytes each), which, across all SMs at boost clock, works out to roughly 18 TB/s aggregate on the RTX 3090 and roughly 33 TB/s on the H100: an order of magnitude above global memory, which is exactly why tiling into shared memory pays off. Registers are faster still, feeding operands effectively every cycle. The optimization principle is to keep data in fast memory (registers → shared → L1) as long as possible.

-----

###### Hardware architecture quick reference 349

###### D.2.2 GPU compute capabilities

Table D.2 maps each recent NVIDIA architecture to its compute capability, representative GPUs, and the headline features it introduced.

###### Table D.2 NVIDIA architectures, compute capabilities, and headline features Compute Architecture Example GPUs Key features capability

| Volta | 7.0 | V100 | First tensor cores |
|---|---|---|---|
| Turing | 7.5 | RTX 20 series | RT cores, INT8/INT4 |
| Ampere | 8.0, 8.6 | A100, RTX 30 series | Improved tensor cores, TF32 |
| Ada | 8.9 | RTX 40 series | DLSS 3, shader execution reordering |
| Lovelace Hopper | 9.0 | H100 | WGMMA, TMA, FP8, Transformer Engine |
| Blackwell | 10.0 | B100, B200 | TCGen05, FP4, doubled tensor core throughput |

---

Check your GPU:

```
nvidia-smi --query-gpu=name,compute_cap
```

###### D.2.3 CUDA core vs. tensor core decision matrix

Use table D.3 to decide whether a given workload belongs on the CUDA cores or the tensor cores, along with the reason in each case.

###### Table D.3 CUDA cores versus tensor cores by workload

---

###### Use case

---

General compute (non-matrix) Small matrices (<64 × 64) Matrix multiply (training)

Matrix multiply (inference) Element-wise operations Reductions (softmax, LayerNorm) Attention (fused)

```
--format=csv
```

###### Recommended Rationale

CUDA cores Tensor cores only accelerate matrix ops.

CUDA cores Tensor cores

Tensor cores CUDA cores CUDA cores

Tensor core overhead is not worth it. Chapter 7 measured ~15x versus our naive kernel on the RTX 3090 at 4096^3. Especially with FP16/INT8 No benefit from tensor cores Warp-level primitives are more efficient.

Tensor cores Flash Attention uses mma.sync/WGMMA.

-----

| 350 | Appendix D | CUDA reference: commands, hardware, templates, and bindings |
|---|---|---|
| D.3 |  | CUDA templates and boilerplate guide This section collects the template and boilerplate patterns used in the book's code |

and in CUTLASS-style kernels.

###### D.3.1 CUDA template basics

Templates in CUDA work like C++ templates but with some GPU-specific considerations. They're essential for writing flexible, type-generic kernels.

###### Basic kernel template

The following template turns an ordinary element-wise kernel into one that works for any numeric type by adding a single typename T parameter:

```
// Simple templated kernel
template <typename T>
__global__ void add_kernel(T* a, T* b, T* c, int n) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
c[idx] = a[idx] + b[idx];
}
}
```

```
// Launch with different types
add_kernel<float><<<blocks, threads>>>(a_f32, b_f32, c_f32, n);
add_kernel<half><<<blocks, threads>>>(a_f16, b_f16, c_f16, n);
add_kernel<int><<<blocks, threads>>>(a_i32, b_i32, c_i32, n);
```

Key points:

¡ The template parameter T determines the data type at compile time. ¡ Each instantiation generates a separate kernel in the binary. ¡ The type must support the operations used (e.g., the + operator)

###### Templated tile sizes

One of the most common uses of templates in CUDA is compile-time tile sizes:

```
// GEMM with compile-time tile sizes
template <typename T, int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_tiled(const T* A, const T* B, T* C,
int M, int N, int K) {
// Shared memory sized at compile time
__shared__ T As[TILE_M][TILE_K];
__shared__ T Bs[TILE_K][TILE_N];
// Thread indices
int tx = threadIdx.x;
int ty = threadIdx.y;
// Block indices
int bx = blockIdx.x;
```

-----

###### CUDA templates and boilerplate guide 351

```
int by = blockIdx.y;
// Calculate output position
int row = by * TILE_M + ty;
int col = bx * TILE_N + tx;
T sum = 0;
// Tile loop
for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
// Load tiles into shared memory
if (row < M && (t * TILE_K + tx) < K)
As[ty][tx] = A[row * K + t * TILE_K + tx];
else
As[ty][tx] = 0;
if ((t * TILE_K + ty) < K && col < N)
Bs[ty][tx] = B[(t * TILE_K + ty) * N + col];
else
Bs[ty][tx] = 0;
__syncthreads();
// Compute partial dot product
#pragma unroll
for (int k = 0; k < TILE_K; k++) {
sum += As[ty][k] * Bs[k][tx];
}
```

```
__syncthreads();
}
```

```
// Write result
if (row < M && col < N) {
C[row * N + col] = sum;
}
}
```

```
// Launch with specific tile configuration
constexpr int TILE = 32;
dim3 threads(TILE, TILE);
dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
gemm_tiled<float, TILE, TILE, TILE><<<blocks, threads>>>(A, B, C, M, N, K);
```

Benefits of compile-time tile sizes:

¡ Shared memory arrays can be stack allocated (no dynamic allocation). ¡ The compiler can optimize loops with #pragma unroll. ¡ Better register allocation. ¡ Constant folding.

-----

**352 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

###### Template specialization

You can specialize templates for specific types to optimize for hardware features:

```python
// Generic implementation
template <typename T>
__device__ T load_with_cache(const T* ptr) {
return *ptr;
}
// Specialized for float4 (vectorized load)
template <>
__device__ float4 load_with_cache<float4>(const float4* ptr) {
return __ldg(ptr); // Use read-only cache
}
// Specialized for half2 (use tensor core types)
template <>
__device__ half2 load_with_cache<half2>(const half2* ptr) {
return *reinterpret_cast<const half2*>(ptr);
}
```

###### D.3.2 Common boilerplate patterns

These are code patterns you'll use repeatedly. Memorize them or keep this as a reference.

###### Pattern 1: Error-checking macro

Problem: CUDA API calls return error codes, but checking each one is wordy.

Solution: Use a macro for automatic error checking:

```
// Define once at the top of your file
#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA error at %s:%d: %s\n", \
__FILE__, __LINE__, cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
} while(0)
// Use throughout your code
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
kernel<<<blocks, threads>>>(args);
CUDA_CHECK(cudaGetLastError()); // Check kernel launch
CUDA_CHECK(cudaDeviceSynchronize()); // Check kernel execution
```

###### Pattern 2: Complete memory-management workflow

This pattern walks through a full host-to-device round trip: allocate, copy in, launch, copy back, and free, with error checking at each step:

-----

###### CUDA templates and boilerplate guide 353

```
void matrix_add_workflow(float* h_A, float* h_B, float* h_C, int N) {
size_t bytes = N * sizeof(float);
// 1. Allocate device memory
float *d_A, *d_B, *d_C;
CUDA_CHECK(cudaMalloc(&d_A, bytes));
CUDA_CHECK(cudaMalloc(&d_B, bytes));
CUDA_CHECK(cudaMalloc(&d_C, bytes));
// 2. Copy inputs to device
CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
// 3. Launch kernel
int threads = 256;
int blocks = (N + threads - 1) / threads;
add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
CUDA_CHECK(cudaGetLastError());
// 4. Copy result back to host
CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
// 5. Free device memory
CUDA_CHECK(cudaFree(d_A));
CUDA_CHECK(cudaFree(d_B));
CUDA_CHECK(cudaFree(d_C));
}
```

###### Pattern 3: RAII memory management (C++ style)

For cleaner code, use RAII to automatically free memory:

```
// Simple CUDA memory wrapper
template <typename T>
class CudaMemory {
private:
T* ptr_ = nullptr;
size_t size_ = 0;
public:
CudaMemory(size_t n) : size_(n * sizeof(T)) {
CUDA_CHECK(cudaMalloc(&ptr_, size_));
}
~CudaMemory() {
if (ptr_) cudaFree(ptr_);
}
// Delete copy operations (move-only type)
CudaMemory(const CudaMemory&) = delete;
CudaMemory& operator=(const CudaMemory&) = delete;
// Move operations
CudaMemory(CudaMemory&& other) noexcept
: ptr_(other.ptr_), size_(other.size_) {
```

-----

**354 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

```
other.ptr_ = nullptr;
other.size_ = 0;
}
T* get() { return ptr_; }
const T* get() const { return ptr_; }
void copy_from_host(const T* host_ptr) {
CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, size_,
```

➥cudaMemcpyHostToDevice));

```
}
void copy_to_host(T* host_ptr) const {
CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, size_,
```

➥cudaMemcpyDeviceToHost));

```
}
};
// Usage - automatic cleanup!
void example() {
CudaMemory<float> d_A(N);
CudaMemory<float> d_B(N);
CudaMemory<float> d_C(N);
d_A.copy_from_host(h_A);
d_B.copy_from_host(h_B);
add_kernel<<<blocks, threads>>>(d_A.get(), d_B.get(), d_C.get(), N);
d_C.copy_to_host(h_C);
// Memory automatically freed when function exits
}
```

###### Pattern 4: GPU timing

The following helper wraps CUDA events in a small timer class so you can measure kernel time without repeating the event boilerplate:

```
// Method 1: CUDA Events (recommended)
class CudaTimer {
private:
cudaEvent_t start_, stop_;
public:
CudaTimer() {
CUDA_CHECK(cudaEventCreate(&start_));
CUDA_CHECK(cudaEventCreate(&stop_));
}
~CudaTimer() {
cudaEventDestroy(start_);
cudaEventDestroy(stop_);
}
void start() {
```

-----

###### CUDA templates and boilerplate guide 355

```
CUDA_CHECK(cudaEventRecord(start_));
}
```

```
float stop() { // Returns elapsed time in milliseconds
CUDA_CHECK(cudaEventRecord(stop_));
CUDA_CHECK(cudaEventSynchronize(stop_));
float elapsed;
CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
return elapsed;
}
};
```

```
// Usage
CudaTimer timer;
timer.start();
my_kernel<<<blocks, threads>>>(args);
float elapsed_ms = timer.stop();
printf("Kernel took %.3f ms\n", elapsed_ms);
// Method 2: Multiple iterations for accuracy
float benchmark_kernel(int iterations = 100) {
CudaTimer timer;
// Warmup
for (int i = 0; i < 10; i++) {
my_kernel<<<blocks, threads>>>(args);
}
CUDA_CHECK(cudaDeviceSynchronize());
// Benchmark
timer.start();
for (int i = 0; i < iterations; i++) {
my_kernel<<<blocks, threads>>>(args);
}
float total_ms = timer.stop();
return total_ms / iterations; // Average time per iteration
}
```

###### Pattern 5: Kernel configuration calculator

This helper derives grid and block dimensions from a problem size, so launch configuration is computed rather than hardcoded:

```
// Helper to calculate optimal grid dimensions
struct KernelConfig {
dim3 blocks;
dim3 threads;
// 1D configuration
static KernelConfig make_1d(int n, int threads_per_block = 256) {
KernelConfig config;
config.threads = dim3(threads_per_block);
config.blocks = dim3((n + threads_per_block - 1) /
```

-----

**356 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

```
threads_per_block);
return config;
}
```

```
// 2D configuration
static KernelConfig make_2d(int rows, int cols,
int tile_x = 16, int tile_y = 16) {
KernelConfig config;
config.threads = dim3(tile_x, tile_y);
config.blocks = dim3((cols + tile_x - 1) / tile_x,
(rows + tile_y - 1) / tile_y);
return config;
}
```

```
// 3D configuration
static KernelConfig make_3d(int d, int h, int w,
int tile_x = 8, int tile_y = 8, int tile_z
= 8) {
KernelConfig config;
config.threads = dim3(tile_x, tile_y, tile_z);
config.blocks = dim3((w + tile_x - 1) / tile_x,
(h + tile_y - 1) / tile_y,
(d + tile_z - 1) / tile_z);
return config;
}
};
```

```
// Usage
auto config = KernelConfig::make_1d(N, 256);
my_kernel<<<config.blocks, config.threads>>>(args);
auto config2d = KernelConfig::make_2d(M, N);
my_2d_kernel<<<config2d.blocks, config2d.threads>>>(args);
```

###### Pattern 6: Device query and selection

The following code queries each visible GPU and prints the properties you most often need when choosing a device:

```
// Print GPU properties
void print_gpu_info() {
int device_count;
CUDA_CHECK(cudaGetDeviceCount(&device_count));
printf("Found %d CUDA device(s)\n\n", device_count);
for (int i = 0; i < device_count; i++) {
cudaDeviceProp prop;
CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
printf("Device %d: %s\n", i, prop.name);
printf(" Compute Capability: %d.%d\n", prop.major, prop.minor);
printf(" Total Global Memory: %.2f GB\n",
prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
```

-----

###### CUDA templates and boilerplate guide 357

```
printf(" Multiprocessors: %d\n", prop.multiProcessorCount);
printf(" Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
printf(" Max Threads per Multiprocessor: %d\n",
prop.maxThreadsPerMultiProcessor);
printf(" Warp Size: %d\n", prop.warpSize);
printf(" Shared Memory per Block: %zu KB\n",
prop.sharedMemPerBlock / 1024);
printf(" L2 Cache Size: %d MB\n", prop.l2CacheSize / (1024 *
```

➥1024));

```
printf(" Memory Bus Width: %d-bit\n\n", prop.memoryBusWidth);
}
}
```

```
// Select best GPU (highest compute capability)
int select_best_gpu() {
int device_count;
CUDA_CHECK(cudaGetDeviceCount(&device_count));
int best_device = 0;
int best_sm = 0;
for (int i = 0; i < device_count; i++) {
cudaDeviceProp prop;
CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
int sm = prop.major * 10 + prop.minor;
if (sm > best_sm) {
best_sm = sm;
best_device = i;
}
```

```
}
CUDA_CHECK(cudaSetDevice(best_device));
return best_device;
}
```

###### Pattern 7: Verification helper

This helper compares CPU and GPU results within a tolerance, which is the same check used to validate every kernel in this book:

```cpp
// Compare CPU and GPU results
template <typename T>
bool verify_results(const T* cpu_result, const T* gpu_result, int n,
T tolerance = 1e-5) {
T max_diff = 0;
T mean_diff = 0;
int error_count = 0;
for (int i = 0; i < n; i++) {
T diff = std::abs(cpu_result[i] - gpu_result[i]);
max_diff = std::max(max_diff, diff);
mean_diff += diff;
```

-----

**358 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

```
if (diff > tolerance) {
if (error_count < 10) { // Print first 10 errors
printf("Mismatch at %d: CPU=%.6f, GPU=%.6f, diff=%.6e\n",
i, (float)cpu_result[i], (float)gpu_result[i],
(float)diff);
}
error_count++;
}
```

```
}
mean_diff /= n;
printf("\nVerification Results:\n");
printf(" Elements: %d\n", n);
printf(" Errors: %d (%.2f%%)\n", error_count, 100.0 * error_count / n);
printf(" Max difference: %.6e\n", (float)max_diff);
printf(" Mean difference: %.6e\n", (float)mean_diff);
printf(" Tolerance: %.6e\n", (float)tolerance);
if (error_count == 0) {
printf(" ✓ PASS: All values within tolerance\n");
return true;
} else {
printf(" ✗ FAIL: %d values exceed tolerance\n", error_count);
return false;
}
```

```
}
// Usage
verify_results(cpu_output, gpu_output, N, 1e-5f);
```

###### D.3.3 Advanced template patterns

Beyond the basics, three template techniques show up repeatedly in production CUDA code: variadic templates, SFINAE for type dispatch, and constexpr-based compile-time configuration.

###### Variadic templates for multiple data types

The following template uses a parameter pack so a single kernel can accept several input tensors of different types:

```
// Kernel that works with multiple input/output types
template <typename OutputT, typename... InputTs>
__global__ void fused_ops_kernel(OutputT* output, const InputTs*... inputs,
int n) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
// Fold expression (C++17) to apply operation
output[idx] = (static_cast<OutputT>(inputs[idx]) + ...);
}
```

```
}
// Launch with different combinations
fused_ops_kernel<float><<<blocks, threads>>>(out, in1, in2, in3, N);
```

-----

###### CUDA templates and boilerplate guide 359

###### SFINAE for type-specific optimization

The following example uses SFINAE to enable a vectorized load path only for the types where it is valid, falling back to a scalar path otherwise:

```cpp
#include <type_traits>
// Use vectorized loads only for 32-bit types
template <typename T>
__device__ typename std::enable_if<sizeof(T) == 4, T>::type
load_element(const T* ptr) {
return __ldg(ptr); // Vectorized read-only cache load
}
```

```cpp
// Fallback for other sizes
template <typename T>
__device__ typename std::enable_if<sizeof(T) != 4, T>::type
load_element(const T* ptr) {
return *ptr; // Regular load
}
```

###### Compile-time configuration with constexpr

This pattern resolves tile sizes at compile time, letting the compiler specialize and unroll the kernel for each configuration:

```
// Determine optimal tile size at compile time
template <int M, int N>
struct TileConfig {
// Ensure tiles are multiples of warp size (32)
static constexpr int TILE_M = (M + 31) / 32 * 32;
static constexpr int TILE_N = (N + 31) / 32 * 32;
// Total threads must not exceed maximum
static_assert(TILE_M * TILE_N <= 1024,
"Tile size exceeds max threads per block");
// Calculate shared memory requirement
static constexpr size_t SHARED_MEM = TILE_M * TILE_N * 2 *
sizeof(float);
static_assert(SHARED_MEM <= 48 * 1024,
"Shared memory exceeds 48KB limit");
};
```

```
// Use in kernel
template <typename T, int M, int N>
__global__ void optimized_kernel(/* ... */) {
using Config = TileConfig<M, N>;
__shared__ T shared[Config::TILE_M][Config::TILE_N];
// ...
}
```

-----

| 360 | Appendix D | CUDA reference: commands, hardware, templates, and bindings |
|---|---|---|
| D.3.4 |  | PyTorch C++ extension boilerplate |

If you're binding CUDA kernels to PyTorch, here's the standard pattern:

```
// my_cuda_ops.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
// CUDA kernel (in .cu file or separate header)
template <typename T>
__global__ void my_kernel(T* output, const T* input, int n) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
output[idx] = input[idx] * 2.0f; // Example operation
}
}
```

```
// C++ wrapper function
torch::Tensor my_cuda_forward(torch::Tensor input) {
// Input validation
TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
// Allocate output
auto output = torch::empty_like(input);
// Launch configuration
int n = input.numel();
int threads = 256;
int blocks = (n + threads - 1) / threads;
// Dispatch based on dtype
AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_kernel", ([&] {
my_kernel<scalar_t><<<blocks, threads>>>(
output.data_ptr<scalar_t>(),
input.data_ptr<scalar_t>(),
n
);
}));
// Check for errors
cudaError_t err = cudaGetLastError();
TORCH_CHECK(err == cudaSuccess,
"CUDA kernel failed: ", cudaGetErrorString(err));
return output;
}
```

```
// PyBind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &my_cuda_forward, "My CUDA operation (CUDA)");
}
```

-----

###### CUDA templates and boilerplate guide 361

Build with setup.py:

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
name='my_cuda_ops',
ext_modules=[
CUDAExtension(
name='my_cuda_ops',
sources=['my_cuda_ops.cu'],
extra_compile_args={
'cxx': ['-O3'],
'nvcc': ['-O3', '--use_fast_math']
}
```

```
)
],
cmdclass={'build_ext': BuildExtension}
)
```

###### D.3.5 Template best practices

Do:

¡ Use templates for type genericity (float, half, int, etc.). ¡ Use compile-time constants for tile sizes (enables optimizations). ¡ Use constexpr for compile-time calculations. ¡ Add static\_assert to catch configuration errors early. ¡ Document template parameters clearly. Don't:

¡ Over-templatize simple kernels (adds compilation time). ¡ Use templates for runtime-variable parameters (use function arguments). ¡ Instantiate too many template combinations (binary bloat). ¡ Mix template metaprogramming with complex device code (which can be hard

to debug).

###### D.3.6 Complete example: Templated matrix multiply

Here's a complete example showing many patterns together:

```
#include <cuda_runtime.h>
#include <stdio.h>
// Error checking macro
#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
```

-----

**362 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

```
fprintf(stderr, "CUDA error at %s:%d: %s\n", \
__FILE__, __LINE__, cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
} while(0)
// Templated GEMM kernel
template <typename T, int TILE_SIZE>
__global__ void gemm_kernel(const T* A, const T* B, T* C,
int M, int N, int K) {
__shared__ T As[TILE_SIZE][TILE_SIZE];
__shared__ T Bs[TILE_SIZE][TILE_SIZE];
int tx = threadIdx.x;
int ty = threadIdx.y;
int row = blockIdx.y * TILE_SIZE + ty;
int col = blockIdx.x * TILE_SIZE + tx;
T sum = 0;
for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
// Load tiles
if (row < M && (t * TILE_SIZE + tx) < K)
As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
else
As[ty][tx] = 0;
if ((t * TILE_SIZE + ty) < K && col < N)
Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
else
Bs[ty][tx] = 0;
__syncthreads();
// Compute
#pragma unroll
for (int k = 0; k < TILE_SIZE; k++) {
sum += As[ty][k] * Bs[k][tx];
}
__syncthreads();
}
if (row < M && col < N) {
C[row * N + col] = sum;
}
}
```

```
// Host function with error checking and timing
template <typename T, int TILE_SIZE = 32>
void gemm_cuda(const T* h_A, const T* h_B, T* h_C,
int M, int N, int K) {
// Allocate device memory
T *d_A, *d_B, *d_C;
size_t size_A = M * K * sizeof(T);
```

-----

###### CUDA templates and boilerplate guide 363

```
size_t size_B = K * N * sizeof(T);
size_t size_C = M * N * sizeof(T);
CUDA_CHECK(cudaMalloc(&d_A, size_A));
CUDA_CHECK(cudaMalloc(&d_B, size_B));
CUDA_CHECK(cudaMalloc(&d_C, size_C));
// Copy inputs to device
CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
// Setup timing
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));
// Launch kernel
dim3 threads(TILE_SIZE, TILE_SIZE);
dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
(M + TILE_SIZE - 1) / TILE_SIZE);
CUDA_CHECK(cudaEventRecord(start));
gemm_kernel<T, TILE_SIZE><<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
CUDA_CHECK(cudaEventRecord(stop));
// Check for kernel errors
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
// Print timing
float elapsed_ms;
CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
double gflops = (2.0 * M * N * K) / (elapsed_ms * 1e6);
printf("GEMM: %.3f ms, %.2f GFLOPS\n", elapsed_ms, gflops);
// Copy result back
CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
// Cleanup
CUDA_CHECK(cudaFree(d_A));
CUDA_CHECK(cudaFree(d_B));
CUDA_CHECK(cudaFree(d_C));
CUDA_CHECK(cudaEventDestroy(start));
CUDA_CHECK(cudaEventDestroy(stop));
}
```

```
int main() {
const int M = 1024, N = 1024, K = 1024;
// Allocate and initialize host memory
float *h_A = new float[M * K];
float *h_B = new float[K * N];
float *h_C = new float[M * N];
```

-----

**364 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

```
// ... initialize h_A and h_B ...
// Run GEMM with different tile sizes
gemm_cuda<float, 16>(h_A, h_B, h_C, M, N, K);
gemm_cuda<float, 32>(h_A, h_B, h_C, M, N, K);
delete[] h_A;
delete[] h_B;
delete[] h_C;
return 0;
}
```

This example demonstrates the following:

¡ Type templates (<typename T>) ¡ Constant templates (<int TILE\_SIZE>) ¡ Error checking with macros ¡ Proper memory management ¡ GPU timing with CUDA events ¡ Kernel launch configuration ¡ Complete workflow from host to device and back

###### D.4 Python bindings for CUDA kernels

This section covers two main approaches for calling CUDA kernels from Python:

```
pybind11 and PyTorch's CUDAExtension. Both are essential tools for integrating custom
```

CUDA kernels into Python workflows.

###### D.4.1 pybind11: Pure Python bindings

```
pybind11 is a lightweight header-only library for creating Python bindings of C++ code.
```

It's perfect for standalone CUDA kernels that don't need PyTorch integration.

###### Simple vector addition example

**1** CUDA kernel (vector\_add.cu):

```
#include <cuda_runtime.h>
__global__ void vector_add_kernel(const float* a, const float* b, float* c,
int n) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
c[idx] = a[idx] + b[idx];
}
}
```

```
extern "C" {
void vector_add_cuda(const float* a, const float* b, float* c, int n) {
int threads = 256;
```

-----

###### Python bindings for CUDA kernels 365

```
int blocks = (n + threads - 1) / threads;
vector_add_kernel<<<blocks, threads>>>(a, b, c, n);
cudaDeviceSynchronize();
}
}
```

```cpp
2 pybind11 binding (binding.cpp):
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
extern "C" {
void vector_add_cuda(const float* a, const float* b, float* c, int n);
}
void vector_add(pybind11::array_t<float> a,
pybind11::array_t<float> b,
pybind11::array_t<float> c) {
// Input validation
if (a.ndim() != 1 || b.ndim() != 1 || c.ndim() != 1) {
throw std::runtime_error("Arrays must be 1D");
}
if (a.size() != b.size() || a.size() != c.size()) {
throw std::runtime_error("Arrays must have same size");
}
// Get raw pointers
auto a_ptr = static_cast<float*>(a.mutable_unchecked().data());
auto b_ptr = static_cast<float*>(b.mutable_unchecked().data());
auto c_ptr = static_cast<float*>(c.mutable_unchecked().data());
// Call CUDA kernel
vector_add_cuda(a_ptr, b_ptr, c_ptr, a.size());
}
PYBIND11_MODULE(vector_add_ext, m) {
m.doc() = "CUDA vector addition extension";
m.def("vector_add", &vector_add, "Add two vectors using CUDA");
}
```

**3** Setup script (setup.py):

```python
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
ext_modules = [
Pybind11Extension(
"vector_add_ext",
sources=[
"binding.cpp",
"vector_add.cu",
```

-----

**366 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

```
],
include_dirs=[
pybind11.get_include(),
"/usr/local/cuda/include"
],
libraries=["cudart"],
language='c++'
),
]
```

```
setup(
name="vector_add_ext",
ext_modules=ext_modules,
cmdclass={"build_ext": build_ext},
zip_safe=False,
)
```

**4** Python usage:

```python
import numpy as np
import vector_add_ext
# Create test data
n = 1000000
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)
# Call CUDA kernel
vector_add_ext.vector_add(a, b, c)
# Verify result
expected = a + b
print(f"Max difference: {np.max(np.abs(c - expected))}")
```

Build and run:

```
# Install pybind11
uv pip install pybind11
# Build extension
uv run python setup.py build_ext --inplace
# Test
uv run test_vector_add.py
```

###### D.4.2 PyTorch integration with PyTorch CUDAExtension

PyTorch's CUDAExtension is designed for kernels that integrate with PyTorch's autograd system and tensor operations.

-----

###### Python bindings for CUDA kernels 367

###### Simple matrix multiplication example

**1** CUDA kernel (matmul.cu):

```
#include <torch/extension.h>
#include <cuda_runtime.h>
template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C,
int M, int N, int K) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
T sum = 0;
for (int k = 0; k < K; k++) {
sum += A[row * K + k] * B[k * N + col];
}
C[row * N + col] = sum;
}
```

```
}
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
// Input validation
TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensors must be 2D");
TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must be
[CA]compatible");
// Allocate output
auto C = torch::zeros({A.size(0), B.size(1)}, A.options());
// Launch configuration
dim3 threads(16, 16);
dim3 blocks((B.size(1) + threads.x - 1) / threads.x,
(A.size(0) + threads.y - 1) / threads.y);
// Launch kernel
AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul", ([&] {
matmul_kernel<scalar_t><<<blocks, threads>>>(
A.data_ptr<scalar_t>(),
B.data_ptr<scalar_t>(),
C.data_ptr<scalar_t>(),
A.size(0), B.size(1), A.size(1)
);
}));
return C;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("matmul", &matmul_cuda, "Matrix multiplication (CUDA)");
}
```

-----

**368 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

**2** Setup script (setup.py):

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
name='matmul_cuda',
ext_modules=[
CUDAExtension(
name='matmul_cuda',
sources=['matmul.cu'],
extra_compile_args={
'cxx': ['-O3'],
'nvcc': ['-O3', '--expt-extended-lambda']
}
)
],
cmdclass={
'build_ext': BuildExtension
}
)
```

**3** Python usage:

```python
import torch
import matmul_cuda
# Create test data
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')
# Call CUDA kernel
C = matmul_cuda.matmul(A, B)
# Verify result
expected = torch.mm(A, B)
print(f"Max difference: {torch.max(torch.abs(C - expected))}")
```

Build and run:

```
# Build extension
uv run python setup.py build_ext --inplace
# Test
uv run test_matmul.py
```

###### D.4.3 pybind11 vs. CUDAExtension: When to use each

The two approaches solve overlapping but distinct problems. Pick based on whether you need standalone bindings or first-class PyTorch interop.

-----

###### Python bindings for CUDA kernels 369

Use pybind11 in the following circumstances: ¡ *Standalone kernels-You have CUDA kernels that don't need PyTorch integration.* ¡ *NumPy compatibility-You want to work directly with NumPy arrays.* ¡ *Lightweight-You want minimal dependencies and faster compilation.* ¡ *Custom workflows-You're building tools outside the PyTorch ecosystem.* ¡ *Performance-You need maximum control over memory management.* Example use cases are scientific computing kernels, image-processing pipelines, custom numerical libraries, and standalone GPU utilities.

Use CUDAExtension in the following circumstances: ¡ *PyTorch integration-Your kernels need to work with PyTorch tensors.* ¡ *Autograd support-You need automatic differentiation.* ¡ *Production models-You're building production machine learning models.* ¡ *Tensor operations-You want seamless tensor type handling.* ¡ *GPU memory management-You want PyTorch's memory pool benefits.* Example use cases are custom neural network layers, optimized training operations, model inference engines, and research prototypes.

###### Performance comparison

Both approaches have similar performance characteristics:

```python
# pybind11 benchmark
import time
import numpy as np
import vector_add_ext
n = 10_000_000
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)
# Warmup
for _ in range(10):
vector_add_ext.vector_add(a, b, c)
# Benchmark
start = time.time()
for _ in range(1000):
vector_add_ext.vector_add(a, b, c)
pybind11_time = (time.time() - start) / 1000
print(f"pybind11: {pybind11_time*1000:.3f} ms")
# CUDAExtension benchmark
import torch
import matmul_cuda
```

-----

**370 Appendix D** ***CUDA reference: commands, hardware, templates, and bindings***

```python
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')
# Warmup
for _ in range(10):
C = matmul_cuda.matmul(A, B)
# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
C = matmul_cuda.matmul(A, B)
torch.cuda.synchronize()
cudaext_time = (time.time() - start) / 1000
print(f"CUDAExtension: {cudaext_time*1000:.3f} ms")
```

###### Migration guide

From pybind11 to CUDAExtension: **1** Replace pybind11::array\_t<T> with torch::Tensor. **2** Use TORCH\_CHECK instead of manual validation. **3** Use torch::empty\_like() for output allocation. **4** Add AT\_DISPATCH\_FLOATING\_TYPES for type dispatch. **5** Update setup.py to use CUDAExtension. From CUDAExtension to pybind11: **1** Replace torch::Tensor with pybind11::array\_t<T>. **2** Add manual input validation. **3** Use pybind11::array\_t<T> for output allocation. **4** Remove AT\_DISPATCH\_FLOATING\_TYPES (handle types manually). **5** Update setup.py to use Pybind11Extension.

-----

# appendix E

## Additional resources

---

When you outgrow this book, these are the references and tools to reach for next:

¡ *CUDA programming guide-https://docs.nvidia.com/cuda/*

[cuda-programming-guide/](https://docs.nvidia.com/cuda/cuda-programming-guide/) ¡ *cuBLAS documentation-https://docs.nvidia.com/cuda/cublas/* ¡ *cuDNN documentation-https://docs.nvidia.com/deeplearning/cudnn/* ¡ *CUTLASS GitHub-https://github.com/NVIDIA/cutlass* ¡ *NCCL documentation-https://docs.nvidia.com/deeplearning/nccl/* ¡ *OpenMPI documentation-https://www.open-mpi.org/doc/*

###### E.1 Profiling and debugging tools

¡ *Nsight Systems-System-wide performance analysis*

- Download: https://developer.nvidia.com/nsight-systems
- Use for application timelines, CPU-GPU coordination, memory transfers, stream analysis ¡ *Nsight Compute (NCU)-Kernel-level profile*
- Download https://developer.nvidia.com/nsight-compute
- Use for memory bandwidth, compute utilization, occupancy, warp stalls

-----

**372 Appendix E** ***Additional resources***

¡ *Compute Sanitizer-Memory error detection*

- Included with CUDA Toolkit
- Use for out-of-bounds access, uninitialized memory, race conditions

###### E.1.1 Nsight Systems: Application-level profiling

Nsys helps identify bottlenecks in the overall application flow. It's particularly useful for the following cases:

¡ *Finding which kernels take the most time-Identify hotspots in your application.* ¡ *Analyzing CPU-GPU synchronization-Detect unnecessary cudaDeviceSynchronize()*

calls. ¡ *Memory transfer analysis-Find slow cudaMemcpy operations.* ¡ *Stream utilization-Verify parallel streams are working as expected.* ¡ *Multi-kernel coordination-Understand kernel launch dependencies.* Basic profiling commands:

```
# Basic CUDA profiling
nsys profile -o output ./program
# With NVTX markers for custom regions
nsys profile --trace=cuda,nvtx -o output ./program
# Profile specific duration (useful for long-running apps)
nsys profile --duration=10 -o output ./program
# Include CPU sampling for CPU bottlenecks
nsys profile --trace=cuda,nvtx,osrt --sample=cpu -o output ./program
```

NVTX (NVIDIA Tools Extension) markers annotate your code so the timeline shows named regions, as in the following example:

```
#include <nvtx3/nvToolsExt.h>
void training_loop() {
nvtxRangePushA("Training Loop");
for (int epoch = 0; epoch < num_epochs; epoch++) {
nvtxRangePushA("Forward Pass");
forward_pass();
nvtxRangePop();
nvtxRangePushA("Backward Pass");
backward_pass();
nvtxRangePop();
nvtxRangePushA("Optimizer Step");
optimizer_step();
nvtxRangePop();
}
nvtxRangePop();
}
```

-----

###### Profiling and debugging tools 373

NVTX 3 is header-only, so no extra link flag is required.

When you open the .nsys-rep file in the Nsight Systems GUI, look for ¡ *CUDA kernels row-Shows the kernel execution timeline*

- Look for gaps (GPU idle time)
- Check for sequential execution that could be parallel ¡ *Memory transfers row-Shows cudaMemcpy operations*
- Large transfers may indicate inefficient CPU-GPU communication
- Overlapping transfers with compute is ideal ¡ *CPU timeline-Shows host code execution*
- Identify if CPU is blocking GPU work
- Find unnecessary synchronization points ¡ *Streams row-Shows different CUDA streams*
- Verify parallel streams actually overlap in time
- Ensure stream dependencies are correct Common issues found by Nsight Systems: ¡ *GPU idle time-Kernels aren't keeping the GPU busy (launch overhead, small* kernels) ¡ *Synchronization overhead-Too many cudaDeviceSynchronize() calls* ¡ *Serial execution-Kernels that could run in parallel are sequential* ¡ *Memory transfer bottlenecks-Large CPU-GPU transfers dominate runtime* ¡ *CPU bottlenecks-Host code taking longer than GPU kernels* Best practices: ¡ Profile after major optimizations to verify improvements. ¡ Use NVTX markers to annotate logical sections of code. ¡ Focus on the longest operations first (Pareto principle). ¡ Compare profiles before and after changes to validate optimizations. ¡ Profile realistic workloads, not toy examples.

###### E.1.2 Nsight Compute: Kernel-level profiling

NCU provides detailed metrics for optimizing individual kernels.

Basic profiling command:

```
ncu --set full \
--section SpeedOfLight_HierarchicalDoubleRooflineChart \
--section MemoryWorkloadAnalysis \
--section Occupancy \
--print-summary per-kernel \
./your_program
```

-----

**374 Appendix E** ***Additional resources***

Key metrics and what they mean:

¡ GPU speed of light throughput:

- *Memory throughput percent-Percentage of peak memory bandwidth utilized*
  - High (>80%): Memory-bound kernel, optimize memory access
  - Low (<20%): Compute-bound or poorly optimized **WARNING** High doesn't always mean good! Uncoalesced access can show high utilization while wasting bandwidth
- *Compute (SM) throughput percent-Percentage of peak compute utilized*
  - High (>80%): Compute-bound, focus on arithmetic intensity
  - Low (<20%): Memory-bound or low occupancy
- *Duration-Actual kernel execution time (most important metric!)* ¡ Memory workload analysis:
- *Memory throughput (GB/s)-Actual bandwidth achieved*
- *L1/TEX hit rate percent-Percentage of L1 cache hits*
- *L2 hit rate percent-Percentage of L2 cache hits*
  - High (>95%): Excellent reuse, data fits in L2
  - Low (<50%): Working set exceeds L2, going to DRAM ¡ Occupancy:
- *Theoretical occupancy percent-Maximum* possible occupancy given resource constraints:
  - Limited by registers, shared memory, block size, or hardware limits
- *Achieved occupancy percent-Actual average active warps/maximum warps:*
  - Gap from theoretical indicates runtime inefficiencies(divergence, imbalance)
- *Block limit-Shows what resource limits occupancy*
  - Block Limit Registers: Too many registers per thread
  - Block Limit Shared Mem: Too much shared memory per block
  - Block Limit Warps: Block size is too small ¡ Warp state statistics:
- *Warp cycles per issued instruction-How many cycles between instructions*
  - Low (<10): Good instruction-level parallelism
  - High (>30): Stalled waiting for dependencies
- *Shows stall reasons (available with --section WarpStateStats)*
  - Long Scoreboard: Waiting for memory/long-latency operations
  - Short Scoreboard: Waiting for shared memory or special-math (MUFU) results
  - MIO Throttle: Memory input/output pipeline saturated
  - No Eligible: No warps ready to execute (common in low-occupancy kernels)

-----

###### Profiling and debugging tools 375

Three profiles recur often enough to be worth recognizing on sight:

¡ Memory-bound kernels (GEMV, softmax, LayerNorm):

```
Memory Throughput: 30-90%
Compute Throughput: <30%
L1 Hit Rate: Variable
Occupancy: Can be high or low
Action: Optimize memory access (coalescing, vectorization, fusion)
```

¡ Compute-bound kernel (GEMM with tiling):

```
Memory Throughput: 20-40%
Compute Throughput: 60-95%
L1 Hit Rate: Low (10-30%)
Occupancy: Moderate (15-40%)
Action: Optimize arithmetic intensity (register blocking, tile sizes)
```

¡ Under-parallelized kernel:

```
Memory Throughput: <5%
Compute Throughput: <5%
Achieved Occupancy: <10%
No Eligible Warps: >70%
Action: Increase parallelism (more threads, larger blocks)
```

As shown in chapter 6, profiling metrics can be misleading:

¡ *High memory throughput ≠ good-Naive kernels often show high utilization from*

wasteful, uncoalesced access. ¡ *Low occupancy ≠ bad-GEMM with 16% occupancy can outperform 66% occu-*

pancy if arithmetic intensity is high. ¡ *Cache hit rate context-High L1 hits in naive GEMM (95%) mean the working set*

fits in cache; low hits in tiled GEMM (12%) mean we're accessing more unique data (better!) Profiling workflow:

**1** *Baseline-Profile a naive implementation to identify the bottleneck type.* **2** *Optimize-Make a targeted change (coalescing, tiling, etc.).* **3** *Measure-Re-profile to verify improvement in relevant metrics.* **4** *Iterate-Focus on the new bottleneck revealed by profiling.* **5** *Validate-Compare final kernel duration against baseline and libraries (cuBLAS,*

cuDNN). Advanced NCU options:

```
# Profile specific kernel by name
ncu --kernel-name my_kernel_name ./program
```

-----

**376 Appendix E** ***Additional resources***

```
# Get only specific sections (faster)
ncu --section SpeedOfLight_HierarchicalDoubleRooflineChart ./program
# Export to file for later analysis
ncu --set full -o profile_output ./program
# Compare two kernels side-by-side
ncu --set full -o kernel_v1 ./program_v1
ncu --set full -o kernel_v2 ./program_v2
# Open both in Nsight Compute GUI for comparison
```

###### E.2 Community and learning

¡ *NVIDIA Developer Forums* [*(CUDA)-https://forums.developer.nvidia.com/c/*](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/)

[accelerated-computing/cuda/](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/) ¡ *CUDA Subreddit-https://www.reddit.com/r/CUDA/* ¡ *GitHub issues-For this book's code: https://github.com/infatoshi/book.cu*

###### E.3 Recommended papers

¡ *Flash Attention-https://arxiv.org/abs/2205.14135* ¡ *Flash Attention 2-https://arxiv.org/abs/2307.08691* ¡ *Megatron-LM(model parallelism)-https://arxiv.org/abs/1909.08053* ¡ *ZeRO (Data Parallelism)-https://arxiv.org/abs/1910.02054*

###### E.4 Glossary of key terms

¡ *Block-Group of threads that execute on the same SM and can share memory*

and synchronize. ¡ *cuBLAS-NVIDIA's* optimized BLAS (Basic Linear Algebra Subprograms)

library for CUDA. ¡ *cuDNN-NVIDIA's library of primitives for deep neural networks.* ¡ *CUTLASS-CUDA Templates for Linear Algebra Subroutines and Solvers, a*

header-only C++ template library for building high-performance GEMM and related kernels. ¡ *DMA-Direct memory access. A hardware mechanism that moves data between*

memory regions without CPU or GPU thread intervention. TMA is NVIDIA's tensor-aware DMA engine. ¡ *FMA-Fused multiply-add. A single hardware operation that computes a \* b + c with*

one rounding step, providing higher precision and throughput than separate multiply and add instructions. Tensor cores execute thousands of FMAs per instruction. ¡ *Global memory (HBM)-Large, slower GPU memory accessible by all threads (40*

GB on an A100 to 180 GB or more on current Blackwell parts; the H100 SXM used in this book has 80 GB).

-----

###### Glossary of key terms 377

¡ *Grid-The collection of all blocks launched for a kernel.* ¡ *Kernel-A GPU function launched from the CPU and executed by many threads*

in parallel. ¡ *MMA-Matrix Multiply-Accumulate instruction(PTX-level tensor core operation).* ¡ *NCCL-NVIDIA Collective Communication Library for multi-GPU communication.* ¡ *PTX-Parallel Thread Execution. NVIDIA's virtual instruction set architecture*

(ISA), which sits between C++/CUDA and the hardware-specific SASS. PTX is portable across GPU generations; the driver JIT-compiles it to SASS at load time. ¡ *RAII-Resource* Acquisition Is Initialization, a C++ pattern for automatic

resource management. ¡ *SASS-The native machine code that runs directly on NVIDIA GPU hardware.*

Generated from PTX by ptxas ahead of time or by the driver at load time. Architecture-specific and not portable. ¡ *SFINAE-Substitution Failure Is Not An Error, a C++ template technique for con-*

ditional compilation. ¡ *Shared memory (SRAM)-Fast, small memory shared by threads in a block (100*

KB/SM on the RTX 3090, up to 228 KB/SM on the H100). ¡ *SM(streaming multiprocessor)-A GPU processing unit that executes thread blocks.* ¡ *TCGen05-Fifth-generation tensor core instruction set, introduced with Black-*

well (B200). Adds native FP4 support and improved TMA integration. ¡ *Tensor cores-Specialized hardware units for accelerated matrix operations.* ¡ *Thread-Individual execution unit in a CUDA kernel.* ¡ *TMA (Tensor Memory Accelerator)-Hopper hardware for asynchronous memory*

transfers. ¡ *Warp-Group of32 threads scheduled together on an SM; they share instruction*

issue but may diverge and execute independently (SIMT). ¡ *WGMMA-Warp Group Matrix Multiply-Accumulate* (Hopper tensor core

instruction). ¡ *WMMA-Warp-level Matrix Multiply-Accumulate (Volta+ tensor core API).*

-----

# index

| Numbers 1D block tiling 174-175 | Symbols .backward() method 64-65 | asynchronous warp group operations 193-202 |
|---|---|---|
| arithmetic intensity | __global__ qualifier 20, 24, 76 | attention 12. See also Flash |

improvement 174 Attention

\_\_half type 167 compiler optimizations 175 HBM bottleneck in standard

\_\_shfl\_down\_sync() 133, 147 1D convolution 35, 45-46 attention 213

\_\_syncthreads() 147, 200 CPU implementation 45 multihead 96

| CUDA kernel for 45 |  | A |  | attention head, defined 96 |  |
|---|---|---|---|---|---|
| 2D block tiling 175-176 |  |  |  | AWQ (activation-aware weight |  |
| hierarchy of tiles 176 |  | absmax 265 |  |  | quantization) 258 |
| implementation details 176 |  | abstraction layers 316-317 |  |  |  |
| maximum register reuse 175 |  | Python-level GPU |  | B |  |
| 2D convolution 35, 46-48 |  |  | programming 316 |  |  |
|  |  | torch.compile 317 |  | backpropagation theory 58-63 |  |
| CPU implementation 47 |  |  |  |  |  |
| CUDA kernel for 47 |  | accuracy validation 242 |  | common pitfalls and debugging |  |
|  |  |  |  |  | tips 63 |
| kernel | launch | Alignment | parameters |  |  |
|  | configuration 48 |  | (CUTLASS) 302 | from slopes to gradients |  |
| 2D max pooling 3, 50-52 |  | AITemplate 323 |  |  | 59-61 |
| CPU implementation 50 |  | AllReduce, combining results |  | layer-by-layer gradient |  |
| CUDA kernel for 50 |  |  | with 272, 283-285 |  | derivations 62-63 |
| kernel launch |  | ALUs (arithmetic logic units) 6, |  |  | linear layer gradients 62 |
|  | configuration 51 |  | 18 |  | ReLU activation |
| 3D arrays 29-32 |  | arithmetic intensity 11, 114, |  |  | gradients 62 |
| 3D | kernel launch |  | 118-120,124, 153, 156, |  | softmax +cross-entropy |
|  | configuration 30 |  | 167-168,173-174,177, |  | combined gradient 63 |
| 3D tensor addition |  |  | 181, 198, 203, 304, 312, | scaling to vectors and |  |
| CPU implementation 29 |  |  | 375-376 |  | matrices 61 |
| CUDA kernel 30 |  | asm keyword 196 |  | vectorized implementation for |  |
| batch image processing 31 |  | asymmetrical quantization 256 |  |  | batches 63 |

-----

**index 379**

| backward pass 65, 82 | cublasCreate() function 87 | CUDA streams 288-290 |
|---|---|---|
| in C 71 | cuBLAS (CUDA Basic Linear | why they work 290 |
| kernels 77 | Algebra | cudaStreamWaitEvent |
| bank conflicts 147, 156, 159, | Subroutines) 83-90 | function 29 |
| 173, 179, 201, 204, | cublasSgemm API call 86 | cuDNN (CUDA Deep Neural |
| 207-208 | debugging gradient | Network library) 84 |
| binding pipeline 93-95 | instability 88 | CuTe (CUDA Templates for |
| blockReduceSum helper | integrating cuBLAS 84 | Tensors) 303 |
| function 137 | necessary lie and path | CUTLASS (CUDA Templates for |
| block reduction 164 | forward 89 | Linear Algebra |
| block-scaled computation 308 BlockScaledLayout 309 | optimizing training loop 84 performance and final | Subroutines and Solvers) 84, 297-312, 321 |
| block-wise quantization 248 | verification 88 | auto-tuning 322 |
| book.cu repository 331 | cublasHgemm function 283 | building kernels 300-303 |
| boundary check 27 | CUBLAS_OP_T 85, 87 cublasSaxpy function 87 | composing kernel with CollectiveBuilder 301 |
| C | cublasSgemm API call 86 | defining kernel shape 300 |
| cache lines 130 calibration quality 251 | CUDA (Compute Unified Device Architecture) 5-7. See also neural networks, | launching kernel from host 302 |
| causal mask 99 | training in raw CUDA | CuTe 303 |
| channel-wise quantization 245 | 3D arrays 29-32 | decomposing GEMM 299 |
| CNNs (convolutional neural | anatomy of CUDA | distributed GEMM 311 |
| networks) 12, 38, 84 | program 20-23 | importance of 298-300 |
| coalescing, improving through | compared with PyTorch 1-2 | performance tuning 304-306 |
| thread remapping 169 | deep learning and 11-13 | profiler 310, 322 |
| CollectiveBuilder | host and device 5 | Python CuTe DSL 322 |
| (CUTLASS) 301, 303 CollectiveEpilogue 302 | kernels 6 next steps 313 | cutlass_profiler 310 |
| CollectiveMainloop 302 column-major order 87 compiler workflows 320 | parallelism 23 running programs 19 | D Data Center GPU Manager |
| computational graph 60 | scaling up 25-29 | (DCGM) 314 datacenter-level |
| compute-bound operations 167, | template libraries 321-322 | considerations 314 |
| 213. See also memory- | throughput vs. latency 18 | DataLoader pipeline 54 |
| bound operations | universal pattern 32 | data parallelism 292 |
| consumer threads 199 | when custom CUDA is | DCGM (Data Center GPU |
| contiguity 117 | needed 3-5 | Manager) 314 |
| convolution and pooling | your first program 17 | debugging |
| toolkit 45-52 | cudaDeviceSynchronize() 80, | common pitfalls and tips 63 |
| 1D convolution 45 | 82, 287, 290 | gradient instability 88 |
| 2D convolution 46-48 | CUDAExtension 94, 365-371 | deep learning 11-13 |
| 2D max pooling 50-52 | cudaFree() 22, 82 | Blackwell TCGen05 |
| convolution-heavy | CUDAGuard 95 | instructions 315 |
| architectures 315 | cudaMalloc() 21-22, 75, 82 | computation 12 |
| convolutions, defined 12 | cudaMemcpyAsync function 290 | convolution-heavy |
| CPUs (central processing units), | cudaMemcpy function 75, 80 | architectures 315 |
| 3D tensor addition 29 | cudaMemcpyHostToDevice | datacenter-level |
| cross-entropy loss function 63 | flag 22 | considerations 314 |
| CUDA_CHECK macro 75 | cuda_runtime.h header 75 | GPUs 13 |

-----

###### 380 index

| next steps 324 |  | element-wise operations 12 |  | G |  |
|---|---|---|---|---|---|
| resources 325 |  | embedding 99-105 |  | GELU (Gaussian Error Linear |  |
| dequantization 240 |  | end-to-end performance |  |  | Unit) 12 |
| devices 5 |  |  | 317-320 | GEMM (general matrix |  |
| overview of 5 |  | ONNX 319 |  |  | multiplication) 12, |
| dim3 type 28, 31 |  | SGLang 318 |  |  | 166-167 |
| distributed computing 269 |  | TensorRT-LLM 318 |  | compute-bound |  |
| AllGather 276 |  | vLLM 318 |  |  | operation 167 |
| AllReduce 272 |  | exp() function 141 |  | CPU implementation 41 |  |
| Broadcast 272 |  |  |  | CUDA kernel 42 |  |
| Gather 273 |  | F |  | decomposing in |  |
| NCCL's primitives 271-277 |  | Flash Attention 212 |  |  | CUTLASS 299 |
| pipeline parallelism 285-290 |  | compute-bound to memory- |  | distributed 311 |  |
| CUDA streams 288-290 |  |  | bound 213 | kernel launch |  |
| sequential processing |  |  | HBM bottleneck in standard |  | configuration 42 |
|  | 286-290 |  | attention 213 | memory access problem 168 |  |
| Reduce 274 |  | designing 214-218 |  | naive kernel 34, 41-42 |  |
| ReduceScatter 275 |  |  | challenges 214 | optimization summary 180 |  |
| ring vs. tree algorithms 277 |  |  | tiling and online | profiling 179 |  |
| scaling strategies 291-296 |  |  | softmax 215 | switching to FP16 for 167 |  |
| parallelism strategies 292 |  | implementing 218-229 |  | tiling and shared memory 170 |  |
| when to use each 294 |  |  | benchmarking kernel 228 | GemmUniversal 302 |  |
| tensor parallelism 280-285 |  |  | dissecting naive | GemmUniversalAdapter |  |
|  |  |  | implementation |  | wrapper 302 |
|  | combining results with |  | 218-220 |  |  |
|  | AllReduce 283-285 |  |  | GEMV-dominated inference |  |
|  |  |  | fused Flash Attention |  | operations 107 |
| initialization and GPU |  |  | kernel 223-227 |  |  |
|  | assignment 280 |  |  | GEMV(general matrix-vector |  |
|  |  |  | three-level loop |  | multiplication) |
| local matrix |  |  | hierarchy 221 |  |  |
|  | multiplication 282 |  |  |  | 118-122,128-141 |
|  |  |  | work distribution |  |  |
| memory allocation 282 |  |  | strategy 220 | coalesced access 131-134 |  |
| DistributedDataParallel |  | kernel evolution 229-232 |  | final performance |  |
|  | wrapper 293 |  |  |  | analysis 139 |
|  |  |  | Flash Attention 1and 2 229 |  |  |
| distributed GEMM 311 |  |  |  | maximizing bandwidth with |  |
|  |  |  | Flash Attention 3 231 |  | vectorization 138 |
| DMA(Direct Memory |  |  |  |  |  |
|  | Access) 193 |  | Flash Attention 4 231 |  |  |
|  |  |  |  | memory access problem 129 |  |
| dynamic quantization 253-255 |  |  | lessons from 231 |  |  |
|  |  |  |  | optimization summary 140 |  |
|  |  | FMA (fused multiply-add) 173, |  |  |  |
|  |  |  | 191 | profiling | coalescing |

**E** improvement 134

forward method 65 edge hardware 323 scaling up with block-level

forward pass 65 reduction 135-138 elementwise addition 35-40 in C 70

matrix transpose 38-40 GEMV vs. GEMM 114

kernels 77

FP16 (16-bit floating-point

| CPU reference 39 |  | generated-kernel workflows 320 |
|---|---|---|
| launching transpose | format) 167, 223 | GPTQ (GPT quantization) 264 |
| kernel 40 | switching GEMM to 167 | GPU Mode 322 |
| sequential loops to parallel | FP32 to INT8 quantization 240 | GPUs (graphics processing units) |
| kernel 39 | FSDP (Fully Sharded Data | deep learning 13 |
| parallel threads 36-37 | Parallel) 275, 293 | initialization and |
| launching kernel 37 | Fully Sharded Data Parallel | assignment 280 |
| line-by-line walk-through 37 | (FSDP) 275, 293 | gradient computations 12 |

-----

**index 381 gradients**

from slopes to 59-61 layer-by-layer gradient

derivations 62-63

linear layer gradients 62 ReLU activation

gradients 62

softmax +cross-entropy

combined gradient 63 group-wise quantization 247

###### H

HBM (high-bandwidth

bottleneck in standard He initialization hierarchical reductions high-level frameworks and hosts HPC (high-performance memory) 9

146

| attention | 213 55 |
|---|---|
| languages | 320-321 |

5 computing) 11 dataset and tokenization 95 LeetGPU 323 establishing baseline linear layers 62

scenario 92-105 LLMs (large language models) 1 reference local matrix multiplication 282

implementation 92 tokenization and **M**

embedding 99-105

| inference kernels 117-122 | m64n64k16 instruction 195 |
|---|---|
| GEMV 118-122 | mad.lo.s32 instruction 197 |
| memory layout and | MAE (mean absolute error) 242 |

contiguity 117 matmul kernel 76 installation and setup 327-330 matrices INT4 quantization 242 transposing 34, 38-40

CPU reference 39

**J** launching transpose

JAX 321 kernel 40 JIT (just-in-time) sequential loops to parallel

compilation 321 kernel 39 matrix multiplication 12

**K** matrix transpose 34

| kernels 6. See also transformer memory kernels access analysis 177 building with CUTLASS access, vectorizing for 300-303 tiling 178-179 |  | kernels 6. See also transformer kernels building with CUTLASS 300-303 | kernels 6. See also transformer kernels building with CUTLASS 300-303 | memory access analysis 177 access, vectorizing for tiling 178-179 | access analysis 177 access, vectorizing for tiling 178-179 |
|---|---|---|---|---|---|
| I forward and backward pass 77 allocation 282 |  |  |  |  |  |
| inference 105-117 launching 37 bandwidth utilization 128 |  |  |  |  |  |
| deploying optimized |  | line-by-line walk-through 37 |  | efficiency 14 |  |
|  | stack 115-117 | KV cache 106-108,323 |  | hierarchy 9-11 |  |
| complete generation |  |  |  | layout 117 |  |
|  | example 115 | L |  | management, training and |  |
| expanding | CUDA kernel |  |  |  | verifying neural nets in |
|  | coverage 117 | L1 cache 9 |  |  |  |
|  |  | L2 cache 9 |  |  | raw CUDA 75 |
| production inference |  |  |  | min-heap 162-164 |  |
|  | results 115 | latency 14, 18 |  |  |  |
|  |  | LayerNorm 152-160 |  | min-max calibration 250 |  |
| high-level tricks 323 |  |  |  | Mixture of Experts (MoE) |  |
| optimizing | autoregressive generation 105-115 | concrete | numerical example 154 |  | 110-114,117, 161 |
| cache vs. recomputation |  | fused kernel 157-159 |  | MLP (Multilayer Perceptron) 52 |  |
|  | tradeoffs 109 | fused with warp |  | MMA (Matrix Multiply- |  |
| GEMV-dominated inference |  |  | operations 155-157 |  | Accumulate) |
|  | operations 107 |  | multikernel parallel |  | instructions 185, 186 |
| GEMV vs. GEMM 114 |  |  | approach 155 | mma_sync intrinsic 225 |  |
| KV cache 106-108 |  |  | performance of warp-based | MoE (Mixture of Experts) |  |
| MoE (Mixture of |  |  | LayerNorm 157 |  | 110-114,117, 161 |
|  | Experts) 110-114 | mapping to GPU kernel |  | routing 112, 120-122 |  |
| transformers in CUDA |  |  | logic 155 | Mojo 321 |  |
|  | 92-105 | optimization summary 159 |  | MPI (Message Passing |  |
| architecture overview |  | performance issues in |  |  | Interface) 280 |
|  | 96-99 |  | naive 153 | MSE (mean squared error) 242 |  |
| binding pipeline 93-95 |  | profiling results 159 |  | multihead attention 96 |  |

-----

###### 382 index

| N |  | integrating cuBLAS 84 | online softmax 143-145 |  |
|---|---|---|---|---|
| naive kernels 13, 34 |  | layer-by-layer gradient | concrete numerical |  |
| derivations 62 example 144 |  |  |  |  |
| convolution and pooling |  |  |  |  |
|  | toolkit 45-52 | matmul kernel 76 | key insight 143 |  |
|  | 1D convolution 45 | memory management 75 | ONNX (Open Neural Network |  |
|  |  | memory management and |  | Exchange) 319 |
|  | 2D convolution 46-48 |  |  |  |

initialization 69 2D max pooling 50-52 **P**

naive CUDA approach 74 elementwise addition 35-40

necessary lie and path parallelism 23 parallel threads 36 forward 89 neural network kernels 41-44 data 292

NumPy approach 64-68 FSDP 293 general matrix multiplication optimizing training loop 84

(GEMM) 41-42 hardware 294

performance and final strategies 292-294 softmax 43-44 verification 88 NCCL (NVIDIA Collective parallel

profiling 82 Communications kernels 39 Library), scaling to vectors and

matrices 61 opportunities 8-11 primitives 271-277 threads 36-37 AllGather 276 setup 69

setup, training, and PCIe (Peripheral Component AllReduce 272 Interconnect Broadcast 272 benchmarking 54-57

Express) 270 Gather 273 training loop and

performance 72 percentile calibration 251 Reduce 274 performance tuning 304-306

training loops 80 ring vs. tree algorithms pipeline parallelism 285-290

277-279 vectorized implementation for

batches 63 naive pipeline parallelism nccl-tests suite 271 implementation,

nn.Module 55

general matrix multiplication

| neural network kernels 41-44 | NormalFloat4 (NF4) 264 | sequential processing 286-290 |
|---|---|---|
| (GEMM) 41-42 | Nsight Compute (ncu) 127 | optimized implementation, |
| softmax 43-44 neural networks, training and | Nsight Systems (nsys) 127, 266, 326, 344, 372-374 | CUDA streams 288-290 pooling, convolution and pooling |
| verifying in raw | numerical precision 112 | toolkit 45-52 |
| CUDA 53-90 | NumPy | 1D convolution 45 |
| backpropagation theory | implementation 64-68 | 2D convolution 46-48 |
| 58-63 | manual implementation in | 2D max pooling 50-52 |
| backward pass in C 71 | NumPy 65 | precision, architecture and |
| C implementation 68-74 | performance 68 | 306-309 |
| common pitfalls and debugging | verification technique 67 | across generations 306 |
| tips 63 | nvidia-smi command 271 | evolution from CUTLASS 2.x to |
| cuBLAS approach 83-90 | NVLink 270 | 3.x 306 |
| cublasSgemm API call 86 CUDA setup and memory | NVTX 344, 349, 373-374 | FP4 performance and use cases 309 |
| management 75 debugging gradient | O occupancy 10, 126-127, | FP4 technology and Blackwell architecture 308 |
| instability 88 | 134-135,139-140,145, | implementing FP4 GEMMs |
| forward and backward pass | 150-151,160, 162, | with CUTLASS 308 |
| kernels 77 | 165-166,179-181,204, | narrow precision |
| forward pass in C 70 | 208, 266, 344, 372, | operations 308-309 |
| from slopes to gradients | 374-376 | prefix caching 323 |
| 59-61 | online algorithm extension 155 | producer threads 199 |

-----

**index 383**

| profiling | symmetric and asymmetric | S |
|---|---|---|
| CUDA kernels 127 | range placement | scale computation 241 |

CUTLASS kernels 310 237-239 common pitfalls 262 scaling strategies 291-296 softmax optimizations 150 parallelism strategies 292-294 top-k 165 deployment strategies

244-257 data parallelism 292 wall-clock vs. GPU time 82 FSDP 293 PTX (Parallel Thread Execution) block-wise quantization 248

calibration 250-252 hardware 294

inline assembly 196-198 when to use each 294

| pybind 11, 94, 124, 365-371 Python CuTe DSL 322 | channel-wise quantization 245 | scaling up 25-29 SDPA (scaled dot-product |
|---|---|---|
| Python-level GPU programming 316 | granularity schemes 244-250 group-wise | attention) 99 sequential loops 39 |
| PyTorch building 15 | quantization 247 static vs. dynamic | sequential processing 286-290 serving systems 317-320 |
| CUDA vs. 1-2 optimization stack 13-15 | quantization 252-255 symmetric vs. asymmetric | ONNX 319 SGLang 318 |
| compounding effect 14 | quantization 255-257 | TensorRT-LLM 318 |
| importance of 15 | tensor-wise | vLLM 318 |
| naive kernels 13 | quantization 244 | SGLang 318 |
| optimization layers 13 | exercises 267 | shared memory 9, 170 |
| scaling to multiple GPUs 14 | GPTQ (GPT | becoming implicit 207 |
| when custom CUDA is needed 3-5 | quantization) 264 NormalFloat4 (NF4) 264 | fused softmax with 147-149 SiLU (Sigmoid Linear Unit) 12 |
| Q | quantization kernel profiling 265 | SIMD (Single Instruction, Multiple Data) 8 |
| QAT (quantization-aware | tradeoffs 262 | SIMT(Single Instruction, |
| training) 265 | workflow 260 | Multiple Threads) 8, |
| QLoRA (Quantized Low-Rank | quantization-aware training | 130 |
| Adaptation) 264-265 | (QAT) 265 | Single Instruction, Multiple |
| quantization 234-268 | quantization kernel | Threads (SIMT) 8, 130 |
| accuracy validation strategy 261 activation-aware weight | profiling 265 | SMs (streaming multiprocessors) 18, 31, 184, 266, 298 |
| quantization 258 | R | softmax 34, 141-151,215 |
| advanced techniques 266 | recurrent neural networks | cross-entropy and 63 |
| big vs. small inputs 259 | (RNNs) 84 | CPU implementation 43 |
| building blocks of 235-244 | registers 9 | fused with shared |
| accuracy validation 242 | maximum reuse 175 | memory 147-149 |
| basic quant/dequant | ReLU (rectified linear | kernel launch |
| operations 239-244 | unit) 8-12 | configuration 44 |
| data types 235-239 | gradients 62 | naive CUDA kernel 44 |
| dequantization 240 | ring algorithms (NCCL) | online 143-145,215-216 |
| FP32 to INT8 | 277-279.See also tree | optimization summary 151 |
| quantization 240 | algorithms | parallel with warp |
| INT4 quantization 242 | RNNs (recurrent neural | shuffles 146-147 |
| memory savings 236 | networks) 84 | profiling optimizations 150 |
| precision loss and | routing in PyTorch 112 | vectorized 149 |
| quantization noise 236 | RowMajor operands 302 | software pipelining 305 |
| scale computation 241 | row-major order 87 | StageCountAutoCarveout 305 |

-----

###### 384 index

| static quantization 253 |  |  | storing results 193 | top-k |  |
|---|---|---|---|---|---|
| store_matrix_sync 225 |  |  | with asynchronous memory | for MoE routing 120-122 |  |
| streaming multiprocessors |  |  | operations 199-201 | min-heap 162-164 |  |
|  | (SMs) 18, 31, 184, 266, | tensor parallelism 280-285 |  | parallel with block |  |
|  | 298 | combining results with |  |  | reduction 164 |
| symmetrical quantization 255 |  |  | AllReduce 283-285 | profiling analysis 165 |  |
|  |  | initialization and GPU |  | torch.compile 317, 323 |  |
| T |  |  | assignment 280 | torch/extension.h header 94 |  |
| TCGen05 (fifth-generation tensor |  | local matrix |  | torch.matmul 311 |  |
|  | core instructions) 186, |  | multiplication 282 | torch.nn.functional.scaled_dot_ |  |
|  | 209-210,315, 324 | memory allocation 282 |  |  | product_attention() |
| challenges and |  | TensorRT 323 |  |  | function 10 |
|  | considerations 210 | TensorRT-LLM 318 |  | TPS (tokens per second) 115 |  |
| overview 209 |  | tensor-wise quantization 244 |  | training cost 14 |  |
| template libraries 321-322 |  | threadIdx 24 |  | training loops 80 |  |
| CUTLASS profiler and |  | thread remapping 169 |  | optimizing 84 |  |
|  | auto-tuning 322 |  |  |  |  |
| Python CuTe DSL 322 |  | threads 6 |  | transformer block 96 |  |
| tensor cores 183-187 |  | parallel 36-37 |  | transformer inference 91 |  |
| decision framework for when to |  |  | launching kernel 37 | transformer kernels |  |
|  | use 205 |  | line-by-line walkthrough 37 | 1D blocktiling 174-175 |  |
| history of instructions 186 |  | three-pass problem 141 |  |  | arithmetic intensity |
| overview of 184 |  | throughput 14, 18 |  |  | improvement 174 |
| pattern mapping 205-209 |  | tiled GEMM 172-173 |  |  | compiler optimizations 175 |
|  | manual tiling becomes | performance | of basic | 2D | block tiling |
|  | hardware tiling 205 |  | tiling 173 |  | maximum register |
| shared memory becomes |  | setup 172 |  |  | reuse 175 |
|  | implicit 207 | tile multiplication 172 |  | improving coalescing through |  |
| vectorized loads become |  |  |  |  | thread remapping 169 |
|  | TMA 206 | tiling 170, 215 |  | optimizing |  |
| what you still optimize |  | manual tiling becomes |  |  | coalesced access 131-134 |
|  | manually 208 |  | hardware tiling 205 |  |  |
|  |  | overview of 215 |  |  | final performance |
| performance across GPU |  |  |  |  | analysis 139 |
|  | generations 186 | vectorizing memory access |  |  | GEMM memory access |
| performance analysis |  |  | for 178-179 |  | problem 168 |
|  | 202-205 |  | compute-bound |  | GEMM optimization with |
| progression analysis 203 |  |  | kernels 179 |  | tiling and shared |
| TCGen05 209-210 |  |  | reaching peak |  | memory 170 |
| challenges and |  |  | performance 179 |  | GEMM profiling 179 |
|  | considerations 210 | TMA (Tensor Memory |  |  | GEMV 128-141 |
| overview 209 |  |  | Accelerator) 186, |  | maximizing bandwidth with |
| WGMMA 193-202 |  |  | 193-194,199, 206 |  | vectorization 138 |
| when to use 185 |  | vectorized loads become 206 |  |  | memory access |
| WMMA 188-193,196-201 |  | TMEM (Tensor Memory) 308 |  |  | problem 129 |
| fragments and tiling |  | tokenization 95-105 |  |  | optimization summary 140 |
|  | strategy 188 | toolkit, convolution and |  |  | profiling coalescing |
| loading data with 190-191 |  |  | pooling 45-52 |  | improvement 134 |
| matrix multiply with 191 |  | 1D convolution 45 |  |  | preparing to optimize 127 |
| PTX inline assembly |  | 2D convolution 46-48 |  |  | profiling CUDA |
|  | 196-198 | 2D max pooling 50-52 |  |  | kernels 127 |

-----

**index 385 scaling up with block-level memory layout and **W** reduction 135-138 contiguity 117**

optimizing kernels 126 warp, defined 129-130

top-k 161- 166

compute-bound warp divergence 8, 11, 375 min-heap 162-164

| parallel top-k with block |  | operation 167 | warpgroup_arrive fence |  |
|---|---|---|---|---|
| reduction 164 |  | GEMM 166 |  | instruction 194 |
| profiling analysis 165 |  | LayerNorm 152 | warp groups 193 |  |
| vectorizing memory access for Softmax function 141-160 warpgroup_wait fence |  |  |  |  |
| tiling 178 |  | switching to FP16 for 167 |  | instruction 194 |
| transformers transposing matrices 38-40 warp operations, fused |  |  |  |  |
| inference 105-117 | CPU reference 39 launching transpose kernel 40 |  |  | LayerNorm with 155-157 |
| cache vs. recomputation |  |  |  |  |
| tradeoffs 109 | sequential loops to parallel |  | multikernel parallel |  |
| complete generation |  | kernel 39 |  | approach 155 |
| example 115 | tree algorithms (NCCL) |  | performance of warp-based |  |
| deploying optimized |  | 277-279.See also ring |  | LayerNorm 157 |
|  |  | algorithms | warpReduceSum | helper |
| stack 115-117 |  |  |  |  |
|  | Triton 316 |  |  | function 132 |
| expanding CUDA kernel |  |  |  |  |
| coverage 117 | troubleshooting 341-347 |  | warp shuffles 146-147 |  |
| GEMV-dominated inference | TTFT (time to first token) 115 |  |  |  |
|  |  |  | wavefront, defined 130 |  |
| operations 107 |  |  | weight quantization 323 |  |
|  | U |  |  |  |
| GEMV vs. GEMM 114 |  |  | WGMMA (Warp Group Matrix |  |
| KV cache 106-108 | Ultra-Scale Playbook 323 |  |  | Multiply- |
| MoE (Mixture of | universal pattern 32 |  |  | Accumulate) 186, |
| Experts) 110-114 |  |  |  | 193-202 |
| optimizing autoregressive | V |  |  |  |
|  |  |  | basic 194 |  |
| generation 105-115 | vecadd.cu file 20-23 |  | PTX inline assembly 196-198 |  |
| production inference | vectorAdd function 23 |  |  |  |
| results 115 |  |  | with asynchronous memory |  |
|  | vectorization 138 |  |  | operations 199-201 |
| inference in CUDA 92-105 | vectorized loads |  |  |  |
|  |  |  | with larger tiles 198 |  |
| architecture overview | becoming TMA 206 |  |  |  |
| 96-99 |  |  | with maximum tiles 201 |  |
|  | vectorized softmax 149 |  |  |  |
| binding pipeline 93-95 |  |  | WMMA(Warp Matrix Multiply |  |
|  | vectorizing memory access for |  |  | Accumulate) 186, |
| dataset and tokenization 95 |  | tiling 178-179 |  |  |
|  |  |  |  | 188-193 |
| establishing baseline | compute-bound kernels 179 |  |  |  |
| scenario 92-105 | reaching peak |  | fragments and tiling |  |
| reference |  | performance 179 |  | strategy 188 |
| implementation 92 | verification technique 67 |  | loading data with 190-191 |  |
| tokenization and | isolating and conquering 67 |  | matrix multiply with 191 |  |
| embedding 99-105 | vLLM 318 |  | storing results 193 |  |
| inference kernels 117-122 | volatile qualifier 196 |  | workflows, compiler and |  |
| GEMV 118-122 | VRAM (video RAM) 5 |  |  | generated-kernel 320 |
---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<!-- # About Me -->

Hello! My name is **Li He**, a Master’s student in **Computer Science** at the **University of Victoria** with [Prof. Jianping Pan](https://webhome.cs.uvic.ca/~pan/).

I’m currently a **Full-stack Developer (Co-op)** with the **Ministry of Housing and Municipal Affairs (Information Systems)**, where I build **LLM-powered** features for a production grant management system—covering **natural-language-to-SQL (NL2SQL) reporting**, **summarization**, **analysis**, and **automated scoring**. I developed and integrated an AI reporting module using **Flask + Angular**, **LangChain**, **Azure OpenAI**, and **PostgreSQL**, embedding it into a **.NET** platform and enabling natural-language querying with **Metabase** dashboards.

My broader interests sit at the intersection of **machine learning** and **large-scale systems**, including **distributed training** and performance bottleneck analysis. I enjoy turning research and engineering ideas into practical tools that make complex data and workflows more accessible to non-technical users.

<!-- 👉 Check out my [projects](projects.md) and [CV](cv.pdf) for more details. -->


Work Experience
======
**Full-stack Developer (Co-op)**  
Ministry of Housing and Municipal Affairs, Information Systems, Victoria, BC, Canada  
*2025 - present*  
- Developed LLM-powered AI features (NL2SQL reporting, summarization, analysis, scoring) for a production grant management system.  
- Built and integrated an AI reporting module (Flask + Angular, LangChain, Azure OpenAI, PostgreSQL) into a .NET platform, collaborating using Azure DevOps (task tracking) and GitHub (version control, CI/CD).  

**Software Developer (Part-time)**  
Pigeon Communication Limited, Victoria, BC, Canada  
*2020 – 2021*  
- Programmed and deployed C-based firmware for Arduino Uno microcontrollers to control experimental hardware.  
- Configured and operated UAVs for field experiments, contributing to data collection and system validation.  

**Postal Clerk**  
Canada Post, Victoria, BC, Canada  
*2021 – 2024*  
- Processed and sorted high-volume mail and packages in a time-sensitive, team-based environment.  
- Ensured accuracy and efficiency while supporting daily operations under tight deadlines.  

Selected Project Experience
======

1. **AI Feature Development - Grant Management System (Unity)**  
   *Technologies: Python, Flask, Angular, PostgreSQL, LangChain, Azure OpenAI, Vector Database, Metabase, Azure DevOps, .NET*  
   - Built and deployed an LLM-powered AI system (LangChain + Azure OpenAI) that translates natural language into SQL, enabling non-technical users to query PostgreSQL databases.  
   - Developed AI pipelines for document summarization, application analysis, and automated scoring, providing structured insights and decision support for grant evaluation.  
   - Built and integrated an AI reporting module (Flask + Angular) into a .NET platform via iframe, enabling natural language querying and visualization through Metabase dashboards.  

2. **Traffic Pattern Analysis and Comparison of Distributed Deep Learning Models**  
   *Technologies: PyTorch (DistributedDataParallel), ASTRA-sim, Slurm, NVIDIA DGX-2, Google TPU*  
   - Architected and executed distributed training experiments for models like VGG16 and GPT-3 on Canada’s Alliance HPC clusters, achieving 91% model accuracy with minimal communication overhead.  
   - Utilized ASTRA-sim to model and analyze training performance across various network topologies, identifying fully connected layers as the primary communication bottleneck.  

3. **Large-Scale Text Embeddings & Clustering Using Metadata and Pinecone Embeddings**  
   *Technologies: Python, Pinecone, Sentence Transformers, Scikit-learn (PCA, K-Means), KeyBERT*  
   - Built a semantic search pipeline by generating high-dimensional embeddings from a large text corpus using Sentence Transformers and the Pinecone vector database.  
   - Implemented PCA for dimensionality reduction and leveraged KeyBERT for automated keyword extraction to enrich metadata.  
   - Deployed K-Means clustering to group related documents, significantly improving search result relevance and speed for text-based applications.  

4. **Analysis of RED Algorithm Using Markov Chain Model**  
   *Technologies: Python, NumPy, Matplotlib, Computer Networking, Performance Modeling, Markov Chains*  
   - Developed a Markov Chain model to simulate and analyze the queueing behavior of the Random Early Detection (RED) congestion control algorithm.  
   - Conducted a comparative performance analysis of Linear vs. Exponential RED, visualizing the results with Matplotlib.  
   - Demonstrated that Exponential RED achieves significantly higher throughput and lower packet latency under simulated network congestion.  

5. **Analysis and Prediction of Maternal Health Risk**  
   *Technologies: Python, Scikit-learn (SGD, Random Forest), TensorFlow, Pandas*  
   - Engineered and compared multiple machine learning models to classify maternal health risk levels based on physiological data.  
   - Validated the performance of SGD, Random Forest, and Neural Network classifiers to identify the most effective model for early risk detection.


Research Publications
======
- **[C1]** **L. He**, W. Yang, and J. Pan, “Communication bottleneck analysis for distributed MoE training.” *IEEE International Conference on Communications*, 2026.

- **[J1]** J. Wei, B. Yang, **L. He**, et al., “Design and optimization of low power cabinet lock circuit based on NB-IOT communication,” *Electronic Design Engineering*, 2019, 27(19): 19-24.

- **[C2]** W. Yang, **L. He**, J. Pan, L. Cai, and W. Tang, “StreamMoE: Dual-Stream Pipelining for Accelerated Distributed MoE Training.” [submitted to JSAC]

- **[O1]** W. Yang, **L. He**, L. Cai, A. Sepahi, and J. Pan, “Tile scheduling across multiple paths for smooth interactive 360-degree video streaming.” [to be submitted]


Honors and Awards
======

- **2025**: Bronze Medal of the 2024 China International Innovation Competition, standing out among more than 11,000 research teams worldwide.  
- **2018–2019**: First Prize Scholarship awarded by HNNU to students ranking in the Top 3 in their department based on GPA.  
- **2016**: Outstanding Graduate awarded by HNUST in recognition of outstanding academic record and promising future.  
- **2015**: Outstanding Student awarded by HNUST for demonstrated leadership and dedicated volunteering service.  
- **2014**: Second Prize in Energy Conservation and Emission Reduction Competition awarded by HNUST to students with a novel idea or product beneficial for the environment.


Practical Skills
======

- **Programming Languages:** Python, C, MATLAB, SQL, C#, JavaScript, HTML, CSS  
- **AI / Machine Learning:** Scikit-learn, TensorFlow/Keras, Pandas, NumPy, Matplotlib, LangChain, Vector Databases (Pinecone, VG), LLM Integration (Azure OpenAI)  
- **Backend & Web Development:** Flask, .NET, REST APIs, Angular  
- **Databases & Data:** PostgreSQL, MySQL, Pinecone (Vector DB), Data Analysis, Feature Engineering  
- **Tools & Platforms:** Azure DevOps, Git, GitHub, Postman, pgAdmin, Metabase, Anaconda, Google Colab  
- **Developer Tools:** Visual Studio Code, Microsoft Visual Studio, GitHub Copilot, Claude Code  
- **Knowledge:** Machine Learning, Data Mining, Computer Networking, Communication Principles, Optimization Algorithms

<!-- Example: editing a Markdown file for a talk
![Editing a Markdown file for a talk](/images/editing-talk.png)

More info about configuring Academic Pages can be found in [the guide](https://academicpages.github.io/markdown/), the [growing wiki](https://github.com/academicpages/academicpages.github.io/wiki), and you can always [ask a question on GitHub](https://github.com/academicpages/academicpages.github.io/discussions). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful. -->

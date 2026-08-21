# LLM Knowledge Graphs & Ontologies


---

## Table of Contents 

1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [Acknowledgements](#acknowledgements)
4. [License](#license)



---

## Project Description


---

## Project Structure

```
kg-ontology-llm/
│
├── data/
│   └── raw/                        # Raw text data
│       ├── janis                       # Example text
│       ├── transcripts                 # Video transcripts 
│
├── docs/                           # Documentation 
│   └── prompts/                        # LLM prompts 
│       ├── kg-gen                          # kg-gen library
│
├── results/
│   └── graphs/                         # KG visualizations
│       ├── kg-gen                          # kg-gen library 
│           └── text_janis                      # Example text
│           └── transcripts                     # Video transcripts
│
├── scripts/
│   ├── build_kg_kg-gen.py             # CLI: build and visualize KG with kg-gen library
│   └── collect_yt_transcripts.py                 # CLI: collect YouTube video transcripts
│
├── .gitignore
├── LICENSE
├── README.md
```

---

## Acknowledgements 

This project builds on the following open-source libraries:

- **[kg-gen](https://github.com/stair-lab/kg-gen)** — used to construct and visualize knowledge graphs from unstructured text. Licensed under MIT.

---


## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

--- 
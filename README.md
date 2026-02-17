# 🔍 Retrieval Semantic Search using FAISS

A high-performance semantic search implementation using **FAISS** (Facebook AI Similarity Search) and **Sentence Transformers** to enable efficient similarity-based text retrieval.

## 📖 Overview

This project demonstrates how to build a semantic search system that can find the most relevant text passages based on the meaning of a query, rather than simple keyword matching. It uses dense vector embeddings and approximate nearest neighbor search to achieve fast and accurate results.

## ✨ Features

- **Semantic Understanding**: Uses sentence transformer models for meaningful text embeddings
- **Efficient Search**: Leverages FAISS IndexFlatL2 for fast similarity search
- **Top-K Retrieval**: Returns the most relevant results with distance scores
- **Persistent Storage**: Pre-computed embeddings for quick access

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-00ADD8?style=for-the-badge&logo=meta&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge&logoColor=black)

**Core Libraries:**
- [FAISS](https://github.com/facebookresearch/faiss) - High-performance similarity search
- [Sentence Transformers](https://www.sbert.net/) - State-of-the-art text embeddings
- [NumPy](https://numpy.org/) - Numerical computations

## 📺 About the Dataset

<div align="center">
  <img src="Succession_image.jpg" alt="Succession TV Show Cast" width="100%" style="max-width: 800px;"/>
  <br/>
  <em>Demo dataset based on HBO's Succession TV series</em>
</div>

<br/>

This project uses text data about **HBO's Succession**, an Emmy Award-winning drama series created by Jesse Armstrong. The dataset includes information about the series overview, cast, characters, production details, and critical reception.

**Example Queries:**
```python
"What is Succession about and who created it?"
"Tell me about Logan Roy and his children"
"Which studio produced Succession?"
```

## 🧠 What is FAISS?

**FAISS** (Facebook AI Similarity Search) is a library by Meta AI for efficient similarity search and clustering of dense vectors.

**Why use it?**
- Traditional search matches exact keywords
- Semantic search understands **meaning**
- FAISS finds similar vectors (embeddings) quickly

**Example:** Query "What is Succession about?" matches "TV show about media family" even without shared keywords!

**Key Benefits:**
- ⚡ Searches millions of vectors in milliseconds
- 📈 Scales from thousands to billions of vectors
- 🎯 Multiple index types for different use cases
- 💾 Memory-efficient data structures
- 🚀 GPU support available

## 📁 Project Structure

```
├── main.py                    # Main search implementation
├── embeddings.py              # Generate embeddings from text
├── retreive_embeddings.py     # Load and inspect embeddings
├── embeddings.npy             # Pre-computed embeddings
└── texts.json                 # Source text data
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/KEYUR141/Retreival-Semantic-Search-using-FAISS.git
cd Retreival-Semantic-Search-using-FAISS

# Install dependencies
pip install numpy faiss-cpu sentence-transformers
```

### Usage

**1. Generate Embeddings**
```bash
python embeddings.py
```

**2. Run Search**
```bash
python main.py
```

**Sample Output:**
```
Distances: [[0.8679179 0.8679179 1.0980705 1.1064415 1.2139347]]
Indexes: [[ 3  8  0 21 15]]

Text 1: Succession, American comedy-drama television series created by British writer 
and producer Jesse Armstrong that aired on HBO from 2018 to 2023...
```

**Understanding Results:**
- **Lower distance = Higher similarity** (0.867 is more similar than 1.213)
- **Indexes**: Position of matched texts in the dataset

## 🔍 How It Works

```python
# 1. Load embeddings
embeddings = np.load("embeddings.npy")

# 2. Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 3. Encode query
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embeddings = model.encode([query])

# 4. Search
distances, indexes = index.search(query_embeddings, k=5)
```

## 🎯 Use Cases

- Question Answering Systems
- Document Retrieval
- Recommendation Systems
- Knowledge Base Search
- Chatbot Context Retrieval
- Duplicate Detection

## 🔧 Customization

**Change number of results:**
```python
k = 10  # in main.py
```

**Use different model:**
```python
model = SentenceTransformer('all-mpnet-base-v2')  # Better performance
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Multilingual
```

**Add your own data:**
1. Update `texts.json` with your text passages
2. Run `python embeddings.py`
3. Run `python main.py`

## 📊 FAISS Index Types

| Index Type | Best For | Speed | Accuracy |
|------------|----------|-------|----------|
| **IndexFlatL2** | Small datasets (<1M) | Medium | 100% |
| **IndexIVFFlat** | Medium datasets (1M-10M) | Fast | ~95% |
| **IndexHNSW** | Fast retrieval needed | Very Fast | ~98% |
| **IndexIVFPQ** | Large datasets (>10M) | Very Fast | ~90% |

**For larger datasets:**
```python
# IVF Index
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, 100)
index.train(embeddings)
index.add(embeddings)

# HNSW Index
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(embeddings)
```

## 🐛 Troubleshooting

**Warning: Unauthenticated requests to HF Hub**
```bash
export HF_TOKEN="your_huggingface_token"
```

**Import errors**
```bash
pip install numpy faiss-cpu sentence-transformers
```

**FAISS installation issues**
```bash
conda install -c conda-forge faiss-cpu
```

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
- [Sentence Transformers](https://www.sbert.net/) by UKP Lab
- Dataset from Britannica's Succession article

## 📧 Contact

**KEYUR141** - [GitHub Profile](https://github.com/KEYUR141)

---

⭐ If you find this project helpful, please give it a star!

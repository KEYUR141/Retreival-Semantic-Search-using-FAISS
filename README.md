# 🔍 Retrieval Semantic Search using FAISS

A high-performance semantic search implementation using **FAISS** (Facebook AI Similarity Search) and **Sentence Transformers** to enable efficient similarity-based text retrieval.


## 📖 Overview

This project demonstrates how to build a semantic search system that can find the most relevant text passages based on the meaning of a query, rather than simple keyword matching. It uses dense vector embeddings and approximate nearest neighbor search to achieve fast and accurate results.

**Dataset**: This implementation uses text data from the critically acclaimed HBO series **Succession** as a demonstration corpus, showcasing how semantic search can retrieve relevant information about the show's plot, characters, and production details.

## ✨ Features

- **Semantic Understanding**: Uses `all-MiniLM-L6-v2` sentence transformer model for creating meaningful text embeddings
- **Efficient Search**: Leverages FAISS IndexFlatL2 for fast similarity search
- **Top-K Retrieval**: Returns the most relevant results with distance scores
- **Persistent Storage**: Pre-computed embeddings stored for quick access
- **Demo Dataset**: Includes real-world text data about HBO's Succession TV series

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-00ADD8?style=for-the-badge&logo=meta&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge&logoColor=black)

### Core Libraries

- **[FAISS](https://github.com/facebookresearch/faiss)**: High-performance similarity search library by Meta AI
- **[Sentence Transformers](https://www.sbert.net/)**: State-of-the-art sentence and text embeddings
- **[NumPy](https://numpy.org/)**: Efficient numerical computations
- **[Python 3.x](https://www.python.org/)**: Programming language

## 🧠 What is FAISS?

**FAISS** (Facebook AI Similarity Search) is a library developed by Meta AI Research for efficient similarity search and clustering of dense vectors. It's designed to handle large-scale vector search operations that are common in machine learning applications.

### Why FAISS?

Traditional keyword search matches exact words, but semantic search understands **meaning**. When you convert text into embeddings (numerical vectors), similar meanings produce similar vectors. FAISS helps you find these similar vectors quickly.

**Example:**
- Query: "What is Succession about?"
- Similar matches: "TV show about media family", "HBO drama series"
- These phrases don't share exact keywords but are semantically similar!

### Key Features of FAISS

1. **Speed**: Can search through millions of vectors in milliseconds
2. **Scalability**: Handles datasets from thousands to billions of vectors
3. **Flexibility**: Multiple index types for different use cases (exact vs. approximate search)
4. **Memory Efficiency**: Optimized data structures to minimize RAM usage
5. **GPU Support**: Can leverage GPU acceleration for even faster searches

### How FAISS Works in This Project

```python
# 1. Create index with vector dimension (384 for our model)
index = faiss.IndexFlatL2(dimension)

# 2. Add all pre-computed embeddings to the index
index.add(embeddings)

# 3. Search for k nearest neighbors
distances, indexes = index.search(query_embedding, k=5)
```

**IndexFlatL2** used in this project:
- **Flat**: Performs exhaustive search (checks every vector)
- **L2**: Uses L2 (Euclidean) distance metric
- **Accuracy**: 100% accurate (exact search, no approximation)
- **Best for**: Small to medium datasets (up to ~1M vectors)

### Distance Metrics

FAISS calculates similarity using distance metrics. **Lower distance = Higher similarity**

- **L2 Distance (Euclidean)**: `√((x1-x2)² + (y1-y2)² + ...)`
- In our output: `0.867` is more similar than `1.213`

### When to Use Different FAISS Indexes

| Index Type | Use Case | Speed | Accuracy | Memory |
|------------|----------|-------|----------|--------|
| **IndexFlatL2** | Small datasets (<1M vectors) | Medium | 100% | High |
| **IndexIVFFlat** | Medium datasets (1M-10M) | Fast | ~95% | Medium |
| **IndexHNSW** | Fast retrieval needed | Very Fast | ~98% | High |
| **IndexIVFPQ** | Large datasets (>10M) | Very Fast | ~90% | Low |

## 📺 About the Dataset

<div align="center">
  <img src="https://github.com/user-attachments/assets/b6520d49-92cd-4f2d-b19d-d83744131ca7" alt="Succession TV Show" width="600"/>
  <p><em>Demo dataset based on HBO's Succession TV series</em></p>
</div>

This project uses text data about **HBO's Succession**, an Emmy Award-winning drama series created by Jesse Armstrong. The dataset includes:

- **Series Overview**: Plot summary and premise
- **Cast Information**: Details about main characters and actors
- **Production Details**: Creator, studio, and broadcast information
- **Critical Reception**: Awards and accolades
- **Character Descriptions**: Information about the Roy family members

### Example Queries You Can Try

```python
# Query about the plot
"What is Succession about and who created it?"

# Query about characters
"Tell me about Logan Roy and his children"

# Query about production
"Which studio produced Succession and when did it air?"

# Query about awards
"What awards did Succession win?"
```

## 📁 Project Structure

```
├── main.py                    # Main search implementation
├── embeddings.py              # Script for generating embeddings
├── retreive_embeddings.py     # Utility to load and inspect embeddings
├── embeddings.npy             # Pre-computed text embeddings
├── texts.json                 # Source text data (Succession info)
└── assets/                    # Images and media
    └── succession-poster.jpg  # TV show poster
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy faiss-cpu sentence-transformers
```

> **Note**: For GPU support, install `faiss-gpu` instead of `faiss-cpu`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/KEYUR141/Retreival-Semantic-Search-using-FAISS.git
cd Retreival-Semantic-Search-using-FAISS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **(Optional)** Add the Succession poster image:
```bash
# Create assets folder
mkdir assets

# Download and add succession-poster.jpg to the assets folder
# Or use any image URL in the README
```

## 💻 Usage

### Step 1: Generate Embeddings

First, create embeddings from your text data:

```bash
python embeddings.py
```

**Output:**
```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████| 103/103 [00:00<00:00, 4110.50it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
```

### Step 2: Inspect Embeddings (Optional)

View the generated embeddings and text data:

```bash
python retreive_embeddings.py
```

**Output:**
```
[[-0.07467586 -0.11685868 -0.02425971 ... -0.00394061  0.0834312  -0.0616134 ]
 [-0.01888517 -0.06129563  0.00924178 ... -0.02298332  0.05890682  0.01150667]
 [-0.05811108 -0.02864878 -0.02723223 ...  0.0158628   0.01457181  0.00713025]
 ...
 [ 0.01879968 -0.08314444  0.08157625 ... -0.01742974  0.03530176  0.01411057]]

['Succession, American comedy-drama television series created by British writer and producer Jesse Armstrong...',
 'Widely praised for its imaginative profanity-laden dialogue...',
 ...]
```

### Step 3: Run Semantic Search

Execute the main search script:

```bash
python main.py
```

**Example Query:**
```python
query = "What is Succession About and studio who created it?"
```

**Output:**
```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████| 103/103 [00:00<00:00, 2878.46it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.

Distances: [[0.8679179 0.8679179 1.0980705 1.1064415 1.2139347]]
Indexes: [[ 3  8  0 21 15]]

Text 4: Succession
Text 9: Succession
Text 1: Succession, American comedy-drama television series created by British writer and producer Jesse Armstrong that aired on HBO from 2018 to 2023. The series focuses on the Roy family, whose aging patriarch, Logan Roy, owns the entertainment and media conglomerate Waystar Royco, one of the last surviving legacy media concerns, and struggles to pick a successor from among his power-hungry children, advisers, and investors. While Roy reluctantly acknowledges the need to choose a successor, he cannot seem to find one that satisfies both his desire to maintain family control over his company and to leave his life's work to someone as mercilessly ambitious as he is.
Text 22: Although Succession has drawn fewer viewers than HBO's most popular shows, it has received extensive critical attention from national outlets covering news and culture. In 2023 reporting on media trends found that Succession spawned six times as many online articles in one 30-day period in summer 2023 as any other highly watched television show and seemed to confirm that Succession sparked outsized media coverage compared with reader interest. Some critics have speculated that the media's fascination with the show stems from its own involvement in the volatile and sometimes toxic media industry that the show portrays.
Text 16: Napoleonic succession, Succession style
```

**Understanding the Results:**
- **Distances**: Lower values indicate higher similarity (0.867 is more similar than 1.213)
- **Indexes**: Positions of the matched texts in the original dataset
- **Texts**: The actual content retrieved, ranked by relevance

## 🔍 How It Works

### 1. Load Pre-computed Embeddings
The system loads pre-generated embeddings from `embeddings.npy`:
```python
embeddings = np.load("embeddings.npy")
with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)
```

### 2. Initialize FAISS Index
Creates an L2 distance index for similarity search:
```python
dimension = embeddings.shape[1]  # 384 dimensions
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```

### 3. Encode Query
Transforms the search query into a dense vector:
```python
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
query_embeddings = model.encode([query]).astype('float32')
```

### 4. Perform Search
Finds the k most similar texts:
```python
k = 5  # Number of results to return
distances, indexes = index.search(query_embeddings, k)
```

### 5. Display Results
Retrieves and displays the matched texts:
```python
for idex in indexes[0]:
    print(f"Text {idex+1}:", texts[idex])
```

## 🎯 Use Cases

- **Question Answering Systems**: Find relevant context for answering questions
- **Document Retrieval**: Search through large document collections semantically
- **Recommendation Systems**: Suggest similar content based on meaning
- **Knowledge Base Search**: Semantic search over FAQ or documentation
- **Content Discovery**: Find related articles, posts, or documents
- **Duplicate Detection**: Identify similar or duplicate content
- **Chatbot Context Retrieval**: Fetch relevant information for conversational AI
- **Media Database Search**: Search through TV shows, movies, or entertainment content

## 🔧 Customization

### Changing the Number of Results

Modify the `k` parameter in `main.py`:
```python
k = 10  # Returns top 10 results instead of 5
```

### Updating the Search Query

Edit the query variable in `main.py`:
```python
query = "Your custom search query here"
```

### Using a Different Model

Replace the model in both `embeddings.py` and `main.py`:
```python
# For multilingual support
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# For better performance (larger model)
model = SentenceTransformer('all-mpnet-base-v2')
```

### Adding Your Own Data

Want to use your own dataset instead of Succession data?

1. Update `texts.json` with your text corpus:
```json
[
    "Your first text passage...",
    "Your second text passage...",
    "Your third text passage..."
]
```

2. Regenerate embeddings:
```bash
python embeddings.py
```

3. Run the search:
```bash
python main.py
```

**Ideas for Other Datasets:**
- Movie scripts or reviews
- Product descriptions for e-commerce
- Research paper abstracts
- News articles
- Customer support FAQs
- Book summaries

### Switching to GPU

If you have a CUDA-enabled GPU for faster processing:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

## 📊 Performance

- **Index Type**: IndexFlatL2 (exact search)
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **Search Complexity**: O(n) for flat index
- **Embedding Model Size**: ~80MB
- **Dataset Size**: 23 text passages about Succession

### Optimization Tips

For larger datasets (10,000+ documents), consider:

**1. IVF (Inverted File) Index - For Medium Datasets:**
```python
# Create quantizer
quantizer = faiss.IndexFlatL2(dimension)
# Create IVF index with 100 clusters
index = faiss.IndexIVFFlat(quantizer, dimension, 100)
# Train the index
index.train(embeddings)
# Add vectors
index.add(embeddings)
# Set number of clusters to search
index.nprobe = 10  # Higher = more accurate but slower
```

**2. HNSW Index - For Fast Retrieval:**
```python
# Create HNSW index
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = number of neighbors
# Add vectors (no training needed)
index.add(embeddings)
```

**3. Product Quantization - For Very Large Datasets:**
```python
# Compress vectors while searching
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, 100, 8, 8)
# 100 clusters, 8 sub-quantizers, 8 bits per code
index.train(embeddings)
index.add(embeddings)
```

### Benchmark Comparison

| Dataset Size | Index Type | Search Time | Accuracy | Memory Usage |
|--------------|-----------|-------------|----------|--------------|
| 1K vectors | IndexFlatL2 | ~1ms | 100% | Low |
| 100K vectors | IndexFlatL2 | ~10ms | 100% | Medium |
| 1M vectors | IndexIVFFlat | ~5ms | 95% | Medium |
| 10M+ vectors | IndexIVFPQ | ~3ms | 90% | Low |

## 🐛 Troubleshooting

### Warning: Unauthenticated requests to HF Hub
To increase rate limits and download speeds, set your Hugging Face token:
```bash
export HF_TOKEN="your_huggingface_token"
```

### UNEXPECTED embeddings.position_ids
This warning can be safely ignored—it's related to model architecture differences and doesn't affect functionality.

### Import errors
Ensure all dependencies are installed:
```bash
pip install numpy faiss-cpu sentence-transformers
```

### FAISS installation issues
If you encounter issues installing FAISS:
```bash
# For conda users
conda install -c conda-forge faiss-cpu

# For GPU version
conda install -c conda-forge faiss-gpu
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation
- Add more diverse datasets

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
- [Sentence Transformers](https://www.sbert.net/) by UKP Lab
- Dataset information sourced from Britannica's Succession article
- HBO's Succession TV series for providing an excellent demo dataset

## 📚 Further Reading

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Understanding Vector Search](https://www.pinecone.io/learn/vector-search/)
- [Semantic Search Explained](https://www.elastic.co/what-is/semantic-search)

## 📧 Contact

**KEYUR141** - [GitHub Profile](https://github.com/KEYUR141)

---

⭐ If you find this project helpful, please give it a star!

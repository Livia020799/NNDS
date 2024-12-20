# Neural Network for Data Science
This repository hosts both the first homework (***`NNDS_2024_Homework1`***) and the final project (***``***) for Neural Network for Data Science exam, held by Professor Simone Scardapane, as part of the Master’s degree in Data Science at Sapienza University of Rome.

-------------------------------------------------------------------------------------------------------------------------------------

### **Exam Structure**

The exam in Neural Network for Data Science consisted of one homework and a final project, plus and oral exam.<br>
This repository details both the homework and the final project component of the course.

-------------------------------------------------------------------------------------------------------------------------------------

### **Project Overview**

Directly from the course page here you can see the overview of the points assigned to each section of the exam (as of 2024):

![image](https://github.com/user-attachments/assets/ca0e57cb-8d4b-4efb-a466-aaa32082470b)

**TO DO : PROJECT**

-------------------------------------------------------------------------------------------------------------------------------------

### **Exam Score and Project Usage**
**The first homework received a score of 5 out of 5, and the project received a  score of  out of 10. The overall score, including the oral part fo the exam resulted ina  score of...** Feel free to use it as a reference if you are planning to take the exam in the upcoming years.<br> 
Please do not hesitate to contact me if you need further explanations or encounter any issues with the materials.

---
### **Idea progetto - RNNs with haiku**
Certainly! Here’s a step-by-step outline of how you can approach the **Japanese Haiku Generation** task using a **Recurrent Neural Network (RNN)** from scratch in **JAX**. We will focus on generating **Haiku**—a form of Japanese poetry with a specific syllable pattern (5-7-5), which makes it ideal for an RNN-based sequence generation task.

### Outline for **Japanese Haiku Generation** using an RNN in JAX

---

### **1. Problem Understanding**
Haiku is a form of traditional Japanese poetry consisting of three lines, following a 5-7-5 syllable structure. Your task is to train an RNN to generate new Haiku given a seed word or prompt.

---

### **2. Dataset Preparation**
You need a dataset of Japanese Haiku poems. This dataset will serve as the training data for your RNN model.

#### **Steps**:
- **Collect Haiku Data**: Gather a small dataset of Haiku poems. You can scrape them from websites or use public datasets (like those from poetry collections).
  - Example dataset sources:
    - A collection of classic Haiku (e.g., Bashō’s works).
    - Modern Haiku collections from Japanese literature databases.
  - Alternatively, you can use a text dataset where the lines are already separated to easily create your training set.

- **Text Preprocessing**:
  - **Tokenization**: You can use a Japanese tokenizer like **MeCab** or **spaCy** for Japanese to split the Haiku text into tokens (either at the word or subword level). This will prepare your text for input to the RNN.
  - **Normalization**: Normalize the text by removing unnecessary characters, punctuation, or special symbols.
  - **Sequence Preparation**: Prepare sequences where each Haiku’s line is a sequence of tokens. You’ll also need to map these tokens to integers using a vocabulary dictionary.

  **Example Tokenization**:
  - Original Haiku:
    - "古池や 蛙飛びこむ 水の音"
    - Tokenized: ["古池", "や", "蛙", "飛びこむ", "水", "の", "音"]
  
---

### **3. Model Architecture**

#### **RNN Model (Look at group of slide 11)**:
Since this is a text generation task, an RNN (or a simple **LSTM**/**GRU**) would work well for learning the sequence of tokens.

- **Embedding Layer**: Convert each token into a vector representation.
- **RNN Layer**: Use a basic RNN layer, LSTM, or GRU to capture the dependencies between tokens in the sequence.
- **Dense Output Layer**: A softmax output layer to predict the next token at each time step.
  
**Input and Output**:
- **Input**: The input to the model will be a sequence of tokens from the Haiku (e.g., a previous token or a context like "春" for spring).
- **Output**: At each step, the model generates the next token in the sequence, eventually forming a full Haiku with the 5-7-5 structure.

**Architecture Details**:
- **Embedding Layer**: Convert each token (syllable/word) into a dense vector of size `embedding_dim`.
- **RNN/LSTM/GRU Layer**: Implement an RNN to process sequences, capturing dependencies between syllables/words in the Haiku.
- **Output Layer**: A **softmax** layer that outputs a probability distribution over the entire vocabulary for the next token.
  
---

### **4. Training the Model**

#### **Loss Function**:
- Use **categorical cross-entropy** as the loss function. This will measure how close the model’s predicted token is to the actual next token in the Haiku.

#### **Optimizer**:
- Use **Adam** or **SGD** as the optimizer to update the weights of the model during training.

#### **Training Steps**:
- **Sequence Prediction**: The model will be trained to predict the next token in a sequence of tokens (e.g., from one syllable to the next).
- **Teacher Forcing**: During training, use teacher forcing (i.e., feeding the true previous token as input to the model during training), so the model learns from the actual sequence rather than its own generated output.
  
---

### **5. Generating Haiku**

#### **Text Generation Process**:
- **Seed Prompt**: Start with a seed token, such as a word or syllable (e.g., "春" for spring).
- **Autoregressive Generation**: After generating one token, feed that token as input to predict the next token, repeating this process until the full Haiku (with 5-7-5 structure) is generated.
  
**Generation Example**:
- **Prompt**: "春" (Spring)
  - The model generates a sequence like: 
    - "春の風" ("Spring wind")
    - "花が舞い散る" ("Flowers swirl and scatter")
    - "月明かり" ("Moonlight")

#### **Considerations**:
- Ensure the model generates lines that follow the 5-7-5 syllable structure. One way to do this is to limit the number of tokens per line during training (e.g., 5 tokens for the first line, 7 for the second, and 5 for the third).
- Implement a stopping condition where the model stops generating after reaching the end of a Haiku (i.e., after generating the 3 lines with the required syllables).

---

### **6. Evaluation and Fine-tuning**

- **Qualitative Evaluation**: Since this is a creative task, evaluate the generated Haiku subjectively by checking their poetic quality and adherence to the Haiku structure.
- **Fine-tuning**: Based on the generated results, you may want to fine-tune the model on a more specific subset of Haiku or adjust the architecture to better fit the desired output.

---

### **7. Performance Considerations**

- **Memory Constraints**: Since you have limited memory, start with a smaller model (e.g., a small embedding size, fewer RNN units). Use **batch size of 1** or small batches to minimize memory usage.
- **Training Time**: You may want to use a smaller training dataset or fewer epochs to avoid long training times. A **small model** will reduce the computational cost.

---

### **8. JAX-specific Considerations**

- **JAX Setup**: Use JAX's `jax.nn` and `jax.optimizers` for building the RNN, and take advantage of **JIT compilation** to speed up training by optimizing the computation graph.
- **Autograd**: Use JAX's automatic differentiation (`jax.grad`) for backpropagation during training.

---

### **Summary of Steps**:

1. **Dataset Preparation**:
   - Collect and preprocess Haiku poems into sequences of tokens (using MeCab or similar tokenizer).
2. **Model Construction**:
   - Build an RNN (or LSTM/GRU) in JAX, with embedding, RNN, and output layers.
3. **Training**:
   - Train the model with categorical cross-entropy loss and optimize using Adam.
4. **Generation**:
   - Use seed prompts and autoregressive generation to produce Haiku.
5. **Evaluation and Fine-tuning**:
   - Fine-tune the model based on the quality of generated Haiku.

---

### **Dataset and repositories**

[Possible repository to have a look for the code](https://huggingface.co/datasets/davanstrien/haiku_dpo)
[Possible Dataset of haiku in japanese](https://github.com/Livia020799/Haiku_NNDS) 

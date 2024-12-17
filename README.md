# Myntra Fashion Product Dataset - Recommender System

## Overview
This project implements two types of recommender systems for the **Myntra Fashion Product Dataset**:
1. **Text-Based Recommender System**: Leverages product descriptions, names, and other metadata.
2. **Image-Based Recommender System**: Uses deep learning models to recommend visually similar products based on image data.

The dataset used is sourced from Kaggle: [Myntra Fashion Product Dataset](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset).

---

## Dataset Description
The dataset consists of two key components:
1. **CSV File**: Contains metadata about fashion products such as:
   - Product ID
   - Product Name
   - Description
   - Price
   - Colour
   - Brand
   - Ratings and Reviews
   - Image URLs
2. **Images Folder**: A collection of images for fashion products used for the image-based recommender system.

---

## Project Highlights
### 1. **Text-Based Recommender System**
This system uses product names and descriptions to recommend similar products based on text similarity.

#### **Technologies Used**:
- **TF-IDF Vectorization**: Converts text data into numerical representations.
- **Cosine Similarity**: Measures the similarity between product descriptions.
- **Python Libraries**: `pandas`, `scikit-learn`.

#### **Steps**:
1. Preprocess product metadata (cleaning, handling missing values).
2. Convert `name` and `description` columns into TF-IDF features.
3. Compute pairwise cosine similarity between products.
4. Provide recommendations based on product similarity.

---

### 2. **Image-Based Recommender System**
This system recommends visually similar products by comparing the features extracted from product images.

#### **Technologies Used**:
- **ResNet50**: A pre-trained Convolutional Neural Network (CNN) from the ImageNet dataset.
- **Cosine Similarity**: Measures similarity between image embeddings.
- **Python Libraries**: `TensorFlow/Keras`, `Pillow`, `matplotlib`, `scikit-learn`.

#### **Steps**:
1. Preprocess product images (resize, normalize).
2. Use ResNet50 to extract high-dimensional embeddings for all images.
3. Compute pairwise cosine similarity between image embeddings.
4. Provide recommendations for visually similar products.

---

## How to Run the Project
### Prerequisites
- Python 3.7+
- Required Libraries:
  ```bash
  pip install pandas numpy scikit-learn tensorflow pillow matplotlib
  ```

### Steps to Run
#### **For Text-Based Recommender**:
1. Open `Recommender_System_Myntra_Dataset_Text.ipynb`.
2. Update the dataset path to point to the CSV file.
3. Run the notebook to generate text-based product recommendations.

#### **For Image-Based Recommender**:
1. Open `Recommender_System_Myntra_Dataset_Images.ipynb`.
2. Update the image folder path.
3. Run the notebook to generate visually similar product recommendations.

---

## Example Output
### Text-Based Recommendations:
Input: **Product ID 12345**  
Output:
| Product Name                       | Similarity Score |
|------------------------------------|------------------|
| Blue Cotton Kurta                  | 0.89             |
| Floral Printed Cotton Dress        | 0.85             |
| Navy Blue Anarkali Kurta           | 0.82             |

### Image-Based Recommendations:
Input: **Image of a Red Printed Kurta**  
Output: Top 5 visually similar images displayed side-by-side using Matplotlib.

---

## Improvements
1. Save image embeddings for faster retrieval.
2. Add clustering methods (e.g., K-Means) to organize products into categories.
3. Combine text and image-based recommenders for a **Hybrid Recommendation System**.

---

## Acknowledgements
Special thanks to:
- **Kaggle Dataset Creator**: Hitesh Suthar
- Pre-trained models from **TensorFlow/Keras**.

---

## Contact
For any queries or suggestions, feel free to reach out at:
- **Email**: namanrawal132@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/naman-rawal-11b233246/

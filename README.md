✍️ Handwritten Text Recognition using CRNN & CTC

<div align="center">










🧠 AI-powered text recognition from image input

A Flask web application that recognizes lowercase English text from a single text-line image using a Convolutional Recurrent Neural Network (CRNN) trained with Connectionist Temporal Classification (CTC) loss.

</div>

🌟 Overview

This project combines Computer Vision and Deep Learning to convert an image containing a line of text into a machine-readable text string.

The pipeline:

Image → Preprocessing → CNN → BiLSTM → Character Classifier → CTC Decoding → Recognized Text

The CNN extracts visual features, the Bidirectional LSTM learns the left-to-right character sequence, and CTC handles alignment between image features and the final text sequence. fileciteturn0file0L3-L5

💡 Example

Input image:  hello world
Output text: hello world

✨ Features

🖼️ Upload a text image through a Flask web interface

🔤 Recognize lowercase English characters and spaces

📁 Supports PNG, JPG, JPEG, BMP, and WEBP

✂️ Automatically crops dark text from a light background

🧠 CNN + Bidirectional LSTM + CTC decoding

📸 Displays the uploaded image and recognized result

📋 Copy recognized text to clipboard

🧪 Trains using synthetic multi-font text-line images

🎛️ Uses light noise, blur, contrast, position, and font-size augmentation fileciteturn0file0L16-L26

🏗️ Project Architecture

                    ┌──────────────────────────┐
                    │      🖼️ Input Image      │
                    │       "hello world"      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   🔧 Preprocessing       │
                    │ Grayscale • Crop • Resize│
                    │       / Pad to 64×256    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      🧠 CNN Encoder      │
                    │ Visual Feature Extraction│
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    🔄 BiLSTM Sequence    │
                    │ Left → Right + Reverse   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    🎯 Character Classifier│
                    │       28 output classes  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       🔗 CTC Decoder     │
                    │ Blank removal + merging  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       ✅ Final Text      │
                    │       "hello world"      │
                    └──────────────────────────┘

The source project describes the same processing flow from image preprocessing through CNN feature extraction, Bidirectional LSTM sequence learning, classification, CTC decoding, and recognized text output. fileciteturn0file0L92-L148

🧩 Model Architecture

The CRNN consists of four main components:

Component

Purpose

🖼️ CNN

Extracts visual features from the input image

🔄 Bidirectional LSTM

Learns character sequence context in both directions

🎯 Fully Connected Layer

Maps sequence features to character classes

🔗 CTC Loss / Decoder

Learns alignment and converts predictions into text

fileciteturn0file0L154-L163

📐 Input / Output

Parameter

Value

Input

Grayscale image

Input shape

(batch_size, 1, 64, 256)

Image height

64 px

Image width

256 px

Vocabulary

a-z + space

CTC blank

Index 0

Output classes

28

Typical time steps

64

Output

Recognized text string

The 28 classes consist of 26 lowercase letters + 1 space + 1 CTC blank class. fileciteturn0file0L165-L185

🧠 CNN Feature Extractor

The CNN progressively extracts higher-level visual representations using convolution, batch normalization, ReLU activation, and pooling. Its output is converted into a horizontal sequence for the recurrent network. fileciteturn0file0L189-L240

🔄 Bidirectional LSTM

The recurrent stage uses a 2-layer Bidirectional LSTM with:

Input size: 512

Hidden size: 256

Layers: 2

Directions: 2

Output features: 512

Dropout: 0.2

This allows the model to use context from both directions of the text sequence. fileciteturn0file0L243-L266

🔗 How CTC Decoding Works

CTC is useful for text recognition because the model does not need explicit character-level bounding boxes.

For example, the model can produce:

blank → h → h → blank → e → l → blank → l → o → blank

CTC then:

Removes blank tokens

Merges consecutive repeated predictions

Result:

hello

fileciteturn0file0L287-L304

📂 Project Structure

handwritten_char_recognition/
│
├── 📄 app.py
├── 📄 train_word_crnn.py
├── 📄 predict_word.py
├── 📄 test_training_sample.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
│
├── 📁 configs/
│   └── 📄 config.py
│
├── 📁 models/
│   └── 📄 crnn_word_model.py
│
├── 📁 utils/
│   └── 📄 crnn_dataset.py
│
├── 📁 templates/
│   └── 📄 index.html
│
├── 📁 static/
│   └── 🎨 style.css
│
├── 📁 data/
│   └── 🖼️ sample_line.png
│
├── 📁 outputs/
│   ├── 🧠 word_crnn_best.pth
│   ├── 🔤 symbols.txt
│   └── 🖼️ generated_test_sample.png
│
└── 📁 uploads/
    └── .gitkeep

The directory and file responsibilities are based on the supplied project documentation. fileciteturn0file0L30-L67

🛠️ Technologies Used

Technology

Role

🐍 Python

Core programming language

🔥 PyTorch

Deep learning model and training

👁️ Torchvision

Computer vision utilities

🖼️ Pillow

Image processing

🌐 Flask

Web application backend

🧱 HTML

Web interface

🎨 CSS

Frontend styling

⚡ JavaScript

Frontend interactions

fileciteturn0file0L476-L484

⚙️ Installation

1️⃣ Clone the repository

git clone https://github.com/amitkumar15x/handwritten_char_recognition.git
cd handwritten_char_recognition

Replace the repository URL with your actual GitHub repository URL if the repository uses a different name.

2️⃣ Create a virtual environment

Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate.ps1

3️⃣ Install dependencies

pip install -r requirements.txt

If Flask is not already included:

pip install flask

fileciteturn0file0L337-L355

🏋️ Train the Model

Train the CRNN using the provided synthetic text-line dataset:

python train_word_crnn.py --epochs 30 --batch_size 16 --num_samples 12000

The best model is saved to:

outputs/word_crnn_best.pth

The character vocabulary is saved to:

outputs/symbols.txt

fileciteturn0file0L360-L377

🧪 Test the Model

Generate a synthetic test image and evaluate the trained model:

python test_training_sample.py

Example:

Expected text:  'data data neural'
Predicted text: 'data data neural'
Saved image: outputs/generated_test_sample.png

fileciteturn0file0L382-L395

🔍 Terminal Prediction

Run prediction on a single image:

python predict_word.py `
  --model_path outputs/word_crnn_best.pth `
  --image_path data/sample_line.png `
  --symbols_path outputs/symbols.txt

Example:

Recognized text: 'hello world'

fileciteturn0file0L400-L411

🌐 Run the Flask Web App

Start the application:

python app.py

Then open:

http://127.0.0.1:5000

Upload a clear image containing a single line of lowercase text and click Recognize text. fileciteturn0file0L416-L430

🖼️ Supported Input

The model works best with:

✍️ One line of text

🔡 Lowercase English characters

⚫ Black / dark text

⚪ White / light plain background

🔎 Clear and sharp images

↔️ Horizontal text

🚫 Minimal borders

Example inputs:

hello world
machine learning
data data neural
text recognition

fileciteturn0file0L434-L452

⚠️ Limitations

The current system may have difficulty with:

Uppercase letters

Numbers

Punctuation

Multi-line paragraphs

Images containing multiple text regions

Decorative, metallic, glowing, or 3D text

Dark textured backgrounds

Cursive handwriting

Heavily rotated or blurred text

Languages other than English

For genuine handwritten-text recognition, the project documentation recommends training or fine-tuning the model with labeled handwriting images. fileciteturn0file0L457-L472

🚀 Possible Future Improvements

✍️ Add support for uppercase characters

🔢 Add digits and punctuation

🌍 Extend recognition to additional languages

📝 Support multi-line documents

🧠 Train on real handwritten datasets

🎯 Add confidence scores

📄 Add document-level OCR

📱 Build a responsive mobile interface

⚡ Optimize inference for faster prediction

☁️ Deploy the Flask application to a cloud platform

👨‍💻 Author

Amit Kumar

💼 GitHub: @amitkumar15x

📄 License

This project is intended for academic, learning, and portfolio use. fileciteturn0file0L489-L497

<div align="center">

⭐ If you found this project useful, consider giving it a star!

Built with 🧠 Deep Learning + 👁️ Computer Vision + 🐍 Python

</div>
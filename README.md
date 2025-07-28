# SRGAN Image 4x Upscaler

This project uses **SRGAN (Super-Resolution Generative Adversarial Network)** to upscale images by 4x while preserving intricate details. The model is trained on the DIV2K dataset, and the web interface is built using Streamlit. Try it live on Hugging Face!

## 🚀 Features

- **4x Image Upscaling**: Transform low-resolution images into detailed, high-res outputs.
- **Cutting-Edge GAN Tech**: Utilizes SRGAN for photorealistic super-resolution.
- **Easy to Use**: Simple Streamlit interface for image upload and instant enhancement.
- **Open Source**: Full code available for learning and extending.

## Demo

<!-- Replace the URL below with your actual demo video link -->
<!--[![Demo Video](https://github.com/arnavbhatiamait/ESRGAN-4x/blob/45bf3042103b903c0fe959eae0685ac20fec851d/Demo%20images/sragn_final.mp4) -->


https://github.com/user-attachments/assets/6b15c1ef-bca7-4fd2-8994-f1673fb4aab6

Take a look at the results below:

- **Original vs. Upscaled (4x):**

<!-- Replace URL with your actual images -->

| Original Image                                    | Upscaled Image (4x)                             |
|-------------------------------------------------|------------------------------------------------|
| ![Original](https://github.com/arnavbhatiamait/ESRGAN-4x/blob/45bf3042103b903c0fe959eae0685ac20fec851d/Demo%20images/person.jpg) | ![Upscaled](https://github.com/arnavbhatiamait/ESRGAN-4x/blob/45bf3042103b903c0fe959eae0685ac20fec851d/Demo%20images/person%20Output.jpg) |

## Live Demo

🔗 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Arnavbhatia/SRGAN)**

## Installation & Usage

### 1. Clone the Repository
```
git clone https://github.com/yourusername/srgan-4x-upscaler.git
cd srgan-4x-upscaler
```

### 2. Set Up Environment (Optional)

It is recommended to use a virtual environment:
```
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```


### 3. Install Requirements
```
pip install -r requirements.txt
```


### 4. Run the App
```
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501/`.

## How It Works

- **Upload** your low-res image.
- The **SRGAN model** processes the image and generates a high-res, 4x upscaled version.
- **View and download** the upscaled result directly from the web interface.

## Screenshots

**Main Page:**  

![Main page screenshot](https://github.com/arnavbhatiamait/ESRGAN-4x/blob/0f66fc19e3ebc3ecea072a3483667ffcd57a71e7/Demo%20images/Screenshot%202025-07-28%20174646.png)

**Result Example:**  

![Result screenshot](https://github.com/arnavbhatiamait/ESRGAN-4x/blob/0f66fc19e3ebc3ecea072a3483667ffcd57a71e7/Demo%20images/Screenshot%202025-07-28%20174654.png)

## References

- **SRGAN Paper:** Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" [(arXiv)](https://arxiv.org/abs/1609.04802)
- **Dataset:** [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Arnavbhatia/SRGAN)

## License

This project is open source under the MIT License.

## Connect

- Questions, feedback, or want to collaborate? Raise an issue or reach out via GitHub!

---

<!-- #### Hashtags

`#SRGAN #ImageSuperResolution #DeepLearning #GANs #DIV2K #ComputerVision #AI #HuggingFace #Streamlit #OpenSource #ImageUpscaler #4xUpscale` -->









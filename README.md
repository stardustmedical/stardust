# Stardust
Building Pathology AI applications.

## Setup

To set up the project, follow these steps:

1. **Create and activate a virtual environment:**
   - For Windows:
     ```
     python -m venv venv
     .\\venv\\Scripts\\activate
     ```
   - For macOS and Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

2. **Install required packages:**
   ```
   pip install -r requirements.txt
   ```

   ```
   pip install git+https://github.com/leandermaerkisch/hover_net.git 
   ```

   OR

   ```
   git clone https://github.com/leandermaerkisch/hover_net.git 
   cd hover_net
   pip install -e .
   ```

3. **Set up the environment variables:**
   - Create a `.env` file in the root directory of the project.
   - Add the following line to the `.env` file:
     ```
     PINECONE_API_KEY="your_pinecone_api_key_here"
     ```
   Replace `your_pinecone_api_key_here` with your actual Pinecone API key.


## Demo
1. Run the demo script to get started:
   ```
   python src/demo.py
   ```

2. Go in your browser on localhost
    ```
    http://127.0.0.1:7860
    ```

3. Download this exemplary image to use for the demo:
    https://www.webpathology.com/slides-13/slides/Pancreas_AcinarCellCA_11B_resized.jpg

4. Upload in the image in the demo and click the button.

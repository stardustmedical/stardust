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
To install the target package from test.pypi but all other dependencies from normal pypi
   ```
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple hover-net==0.0.6```
   ```


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

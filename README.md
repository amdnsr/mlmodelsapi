# ML Models API

This projects aims to serve as the ML Service provider for the main website, and helps decouple the Deep Learning Algorithms from Web Development. 
## Requirements

This project is built using `fastapi`

To run the service, create a virtual environment and run:

```setup
pip install -r requirements.txt
```
## Usage


```usage
python main.py
```

## API Endpoints

- `/captiongeneration`: serves as the endpoint for the captiongeneration service
- `/cartoonization`: serves as the endpoint for the cartoonization service
- `/textsummarization`: serves as the endpoint for the textsummarization service


## Test

- for the captiongeneration service, provide the base64 string of the image
- for the cartoonization service, provide the base64 string of the image
- for the textsummarization service, provide the text as a string



# FindYourFit Demo

### Steps:

1. Download this data: https://github.com/alexeygrigorev/clothing-dataset into a floder called "clothing-dataset" within the FindYourFit Repository.
2. Run ```python feature_extraction```, this will take a while.
3. Run ```python run_inference.py``` and not the second to last number that is printed out, this corresponds to the printed image file.
4. Create a new folder "output" within the FindYourFit Repository.
5. Run ```python run_inference_to_index.py [the number from earlier] 10 output/```, this will put corresponding images in the output folder.
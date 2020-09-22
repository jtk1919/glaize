# Nails detection and scale analysis

- Download the model weights with this commman:
```bash
wget --verbose --continue --timestamping https://www.maths.tcd.ie/~bouracha/weights/mask_rcnn_nails_v1.h5
```
and then place it in the mask_RCNN-master directory. It is not on the github due to its size.

- Place the desired image in the nail_images directory and run the code with:
```python
python3 main.py --image_name test_image1.jpg
```
where you must replace test_image1.jpg with your image of choice.
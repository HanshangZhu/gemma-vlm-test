from PIL import Image
import numpy as np

# Create a blank white image
width, height = 224, 224
white_image = Image.new('RGB', (width, height), 'white')
white_image.save('test.png')
print("✅ Created dummy image at test.png") 
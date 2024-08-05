import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from mpi import load_mpi

root_dir = "checkpoints/mpi-small"
device = "cuda:0"
model = load_mpi(root_dir, device, freeze=True)
model.eval()
transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()])
image = cv2.imread("assets/example_franka_kitchen.jpg")
image = transforms(Image.fromarray(image.astype(np.uint8))).unsqueeze(0)
visual_input = torch.stack((image, image), dim=1) # simply repeat the current observation in downstream downstask
visual_input = visual_input.to(device=device)
lang_input = ("turn on the knob", )
embedding_with_lang_tokens = model.get_representations(visual_input, lang_input, with_lang_tokens = True)
embedding_without_lang_tokens = model.get_representations(visual_input, None, with_lang_tokens = False)
print(embedding_with_lang_tokens.shape, embedding_without_lang_tokens.shape) # (1, 218, 384), (1, 197, 384)

from model.basemodel import Basemodel
from PIL import Image
import torch,torchvision
img = "/workspace/00Dataset/BSDS-yc/test/100007.jpg"
ckpt = "best.pth"
encoder = "DUL-M36"
decoder = "UNETP"
head = "default"
model = Basemodel(encoder_name=encoder,
                  decoder_name=decoder,
                  head_name=head).cuda()
ckpt = torch.load(ckpt, map_location='cpu')['state_dict']

value = ckpt.pop('encoder.conv2.1.weight')
ckpt['encoder.conv2.0.weight'] = value
#
value = ckpt.pop('encoder.conv2.1.bias')
ckpt['encoder.conv2.0.bias'] = value


model.load_state_dict(ckpt)
model.eval()

img = torchvision.transforms.ToTensor()(Image.open(img)).unsqueeze(0)
img = img*2-1

edge = model(img.cuda())

torchvision.utils.save_image(edge,"edge.png")

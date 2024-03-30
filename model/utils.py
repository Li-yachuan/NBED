from torch import nn
def get_encoder(nm,Dulbrn=16):
    if "CAFORMER-M36" == nm.upper():
        from model.caformer import caformer_m36_384_in21ft1k
        encoder = caformer_m36_384_in21ft1k(pretrained=True)
    elif "DUL-M36" == nm.upper():
        from model.caformer import caformer_m36_384_in21ft1k
        encoder = caformer_m36_384_in21ft1k(pretrained=True, Dulbrn=Dulbrn)
    elif "DUL-S18" == nm.upper():
        from model.caformer import caformer_s18_384_in21ft1k
        encoder = caformer_s18_384_in21ft1k(pretrained=True, Dulbrn=Dulbrn)
    elif "VGG-16" == nm.upper():
        from model.vgg import VGG16_C
        encoder = VGG16_C(pretrain="model/vgg16.pth")
    elif "LCAL" == nm.upper():
        from model.localextro import LCAL
        encoder = LCAL(Dulbrn=Dulbrn)
    else:
        raise Exception("Error encoder")
    return encoder

def get_head(nm,channels):
    from model.detect_head import CoFusion_head,CSAM_head,CDCM_head,Default_head,Fusion_head
    if nm == "aspp":
        head = CDCM_head(channels)
    elif nm == "atten":
        head = CSAM_head(channels)
    elif nm == "cofusion":
        head = CoFusion_head(channels)
    elif nm == "fusion":
        head = Fusion_head(channels)
    elif nm == "default":
        head = Default_head(channels)
    else:
        raise Exception("Error head")
    return head


def get_decoder(nm, incs, oucs=None):
    if oucs is None:
        # oucs = (32, 32, 64, 128, 256, 512)
        oucs = (32, 32, 64, 128, 384)

    if nm.upper() == "UNETP":
        from model.unetp import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "UNET":
        from model.unet import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "DEFAULT":
        from model.unet import Identity
        decoder = Identity(incs, oucs[-len(incs):])

    else:
        raise Exception("Error decoder")
    return decoder,oucs[-len(incs):]

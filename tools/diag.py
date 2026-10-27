import os, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from segment_anything import build_sam_vit_l
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DS(Dataset):
    def __init__(self, root):
        self.i = os.path.join(root, 'images'); self.m = os.path.join(root, 'masks')
        self.files = []
        for f in os.listdir(self.i):
            b = os.path.splitext(f)[0]
            if any(os.path.exists(os.path.join(self.m, b+e)) for e in ('.png','.jpg','.jpeg','.tif','.tiff')):
                self.files.append(f)
        self.files.sort()
    def __len__(self): return len(self.files)
    def __getitem__(self, k):
        f = self.files[k]; b = os.path.splitext(f)[0]
        img = Image.open(os.path.join(self.i, f)).convert('RGB')
        m = Image.open(os.path.join(self.m, b+'.png')).convert('L')
        x = transforms.Compose([transforms.Resize((1024,1024)), transforms.ToTensor()])(img)
        return x, f, np.array(m), img.size

def dice(p, g):
    p = p.reshape(-1).astype(np.float64); g = g.reshape(-1).astype(np.float64)
    return (2*(p*g).sum()+1e-7)/(p.sum()+g.sum()+1e-7)

sam = build_sam_vit_l(checkpoint=None, use_adapter=True, adapter_dim_ratio=0.1).to(DEV)
ck = torch.load('best_model.pth', map_location=DEV)
print('ckpt epoch', ck.get('epoch'), 'f1', ck.get('f1_score'))
sam.load_state_dict(ck['model_state_dict']); sam.eval()
sp = torch.zeros((1,0,256), device=DEV); de = torch.zeros((1,256,64,64), device=DEV)

def run(root, name):
    ds = DS(root); dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    res = {k: [] for k in ['v255@.5','vany@.5','v127@.5','vany_best','v255_best']}
    viz = []
    with torch.no_grad():
        for x, f, mask, size in dl:
            x = x.to(DEV)
            xn = sam.preprocess(x)
            detail = sam.mask_decoder.encode_detail(xn)
            eo = sam.image_encoder(xn, return_features=True)
            ie, ef, af = (eo if len(eo)==3 else (*eo, None))
            out = sam.mask_decoder(image_embeddings=ie, image_pe=sam.prompt_encoder.get_dense_pe(),
                                   sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                                   multimask_output=False, adapter_features=af,
                                   detail_features=detail)
            lo = out['masks']
            h, w = int(size[1]), int(size[0])
            lo = F.interpolate(lo, size=(h, w), mode='bilinear', align_corners=False)
            prob = torch.sigmoid(lo)[0,0].cpu().numpy()
            mask = mask[0].numpy() if torch.is_tensor(mask) else mask[0]
            g255 = mask > 127.5
            g127 = (mask > 63) & (mask < 191)
            gany = mask > 63
            for th in [0.3,0.4,0.5,0.6,0.7]:
                pass
            res['v255@.5'].append(dice(prob>0.5, g255))
            res['vany@.5'].append(dice(prob>0.5, gany))
            res['v127@.5'].append(dice(prob>0.5, g127))
            # best fixed threshold scan for vany and v255
            best_any = max((dice(prob>t, gany), t) for t in np.arange(0.1,0.91,0.05))
            best_255 = max((dice(prob>t, g255), t) for t in np.arange(0.1,0.91,0.05))
            res['vany_best'].append(best_any[0]); res['v255_best'].append(best_255[0])
            viz.append((f[0], np.array(Image.open(os.path.join(ds.i,f[0])).convert('RGB')), gany, prob))
    print(f"\n===== {name} (n={len(ds)}) =====")
    for k,v in res.items(): print(f"  {k:12s}: {np.mean(v):.4f}  (std {np.std(v):.4f})")
    # save viz for 4 samples
    n = min(4, len(viz))
    fig,ax = plt.subplots(n,4,figsize=(16,4*n))
    for i in range(n):
        f,img,gany,prob = viz[i]
        ax[i,0].imshow(img); ax[i,0].set_title(f,fontsize=8)
        ax[i,1].imshow(prob>0.5,cmap='gray'); ax[i,1].set_title('pred>0.5',fontsize=8)
        ax[i,2].imshow(gany,cmap='gray'); ax[i,2].set_title('GT any(>63)',fontsize=8)
        ax[i,3].imshow(prob,cmap='gray'); ax[i,3].set_title('prob map',fontsize=8)
        for a in ax[i]: a.axis('off')
    plt.tight_layout(); plt.savefig(f'diag_{name}.png',dpi=80); plt.close()
    return res

run('datasets/DSCA','DSCA')
run('datasets/ROAD','ROAD')

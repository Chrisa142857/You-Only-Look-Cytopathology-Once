import torch
from tqdm import tqdm
from utils import repackage_hidden


def inference(run, loader, model, nepochs, model_name='lstm'):
    model.eval()
    hidden = model.init_hidden()
    probs = {}
    slides = []
    with torch.no_grad():
        for i, one in tqdm(enumerate(loader), desc='Inference\tEpoch: [%d/%d]\t' % (run+1, nepochs)):
            data = one['data']
            if 'transformer' not in model_name:
                output, hidden = model(data.cuda(), hidden)
                hidden = repackage_hidden(hidden)
            else:
                output = model(data.cuda())
            p = output.detach().clone()
            for j, slide in enumerate(one['info']):
                probs[slide] = p[j]
                slides.append(slide)
    return probs, slides
import fastai, pickle, sklearn
from fastai import *
from fastai.vision import *
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Sampler, SequentialSampler, RandomSampler
from random import shuffle


################################ HELPERS ################################

def write_pkl(obj, fname):
    '''Save a Pickle file'''
    with open(fname, 'wb') as f: pickle.dump(obj, f, protocol = pickle.HIGHEST_PROTOCOL)

def read_pkl(fname):
    '''Open a Pickle file'''
    with open(fname, 'rb') as f:
        file = pickle.load(f)
    return file

def flat(multi_dim_list):
    ''' flatten a multi-dimensional list '''
    return [l for t in multi_dim_list for l in t]


################################ TRAINING METRICS ################################

def top_k(input:Tensor, targs:Tensor, k:int, n:int)->Rank0Tensor:
    '''Computes the Top-k accuracy (target is in the top k predictions).'''
    input = input.topk(k, dim = -1)[1]
    targs_dict = {i: [] for i in range(targs.size()[0])}
    for t in targs.nonzero(): targs_dict[t[0].item()].append(t[1].item())

    match = 0
    for i, INP in enumerate(input):
        for true in targs_dict[i]:
            for inp in INP:
                if inp.item() == true: match += 1

    return torch.tensor(match / n)

def top_5(input:Tensor, targs:Tensor)->Rank0Tensor:
    '''Static Top 5 Accuracy metric (get around fastai init issues)'''
    k, n = 5, targs.size()[0]
    return top_k(input, targs, k, n)

def top_10(input:Tensor, targs:Tensor)->Rank0Tensor:
    '''Static Top 10 Accuracy metric (get around fastai init issues)'''
    k, n = 10, targs.size()[0]
    return top_k(input, targs, k, n)

def top_20(input:Tensor, targs:Tensor)->Rank0Tensor:
    '''Static Top 20 Accuracy metric (get around fastai init issues)'''
    k, n = 20, targs.size()[0]
    return top_k(input, targs, k, n)

def top_40(input:Tensor, targs:Tensor)->Rank0Tensor:
    '''Static Top 40 Accuracy metric (get around fastai init issues)'''
    k, n = 40, targs.size()[0]
    return top_k(input, targs, k, n)

def top_100(input:Tensor, targs:Tensor)->Rank0Tensor:
    '''Static Top 100 Accuracy metric (get around fastai init issues)'''
    k, n = 100, targs.size()[0]
    return top_k(input, targs, k, n)

def top_k_avg(input:Tensor, targs:Tensor)->Rank0Tensor:
    '''Get bullseye weighted Top K Accuracy'''
    accs = [top_5(input, targs), top_10(input, targs), top_20(input, targs), top_40(input, targs), top_100(input, targs)]
    return torch.mean(torch.stack(accs), dim = 0).float()

def avg_label_rank(input:Tensor, targs:Tensor)->Rank0Tensor:
    '''Computes average rank of multi-label prediction (1 being the best).'''
    n_batches, n_labels, ranks = targs.size()[0], targs.size()[1], []
    for i in range(n_batches):
        concat = torch.stack([input[i], targs[i]]).T
        ranks.append(n_labels - concat[concat[:,0].argsort()][:,1].nonzero().float().mean().item())
    return torch.tensor(np.mean(ranks)).float().mean()


################################ MIXUP ################################

# https://github.com/mnpinto/audiotagging2019
class AudioMixup(LearnerCallback):
    
    def __init__(self, learn):
        super().__init__(learn)
    
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        
        if train:
            bs = last_input.size()[0]
            lambd = np.random.uniform(0, 0.5, bs)
            shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
            x1, y1 = last_input[shuffle], last_target[shuffle]
            a = tensor(lambd).float().view(-1, 1, 1, 1).to(last_input.device)
            last_input = a*last_input + (1-a)*x1
            last_target = {'y0':last_target, 'y1':y1, 'a':a.view(-1)}
            return {'last_input': last_input, 'last_target': last_target}

class SpecMixUp(LearnerCallback):
    
    def __init__(self, learn:Learner):
        super().__init__(learn)
        self.masking_max_percentage = 0.5
        self.alpha = .6
    
    def _spec_augment(self, last_input, last_target):
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        batch_size, channels, height, width = last_input.size()
        h_percentage = np.random.uniform(low=0., high=self.masking_max_percentage, size=batch_size)
        w_percentage = np.random.uniform(low=0., high=self.masking_max_percentage, size=batch_size)
        alpha = (h_percentage + w_percentage) - (h_percentage * w_percentage)
        alpha = last_input.new(alpha)
        alpha = alpha.unsqueeze(1)
        new_input = last_input.clone()
        
        for i in range(batch_size):
            h_mask = int(h_percentage[i] * height)
            h = int(np.random.uniform(0.0, height - h_mask))
            new_input[i, :, h:h + h_mask, :] = x1[i, :, h:h + h_mask, :]
            w_mask = int(w_percentage[i] * width)
            w = int(np.random.uniform(0.0, width - w_mask))
            new_input[i, :, :, w:w + w_mask] = x1[i, :, :, w:w + w_mask]
        
        new_target = (1-alpha) * last_target + alpha*y1
        
        return new_input, new_target

    def _mixup(self, last_input, last_target):
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        new_input = (last_input * lambd.view(lambd.size(0),1,1,1) + x1 * (1-lambd).view(lambd.size(0),1,1,1))
        if len(last_target.shape) == 2:
            lambd = lambd.unsqueeze(1).float()
        new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return new_input, new_target
    
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if not train: return
        new_input, new_target = self._mixup(last_input, last_target)
        new_input, new_target = self._spec_augment(new_input, new_target)
        return {'last_input': new_input, 'last_target': new_target}

    
class StandardMixUp(LearnerCallback):
    
    def __init__(self, learn:Learner):
        super().__init__(learn)
        self.masking_max_percentage=0.25
        self.alpha = .4

    def _mixup(self, last_input, last_target):
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        new_input = (last_input * lambd.view(lambd.size(0),1,1,1) + x1 * (1-lambd).view(lambd.size(0),1,1,1))
        if len(last_target.shape) == 2:
            lambd = lambd.unsqueeze(1).float()
        new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return new_input, new_target
    
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if not train: return
        new_input, new_target = self._mixup(last_input, last_target)
        return {'last_input': new_input, 'last_target': new_target}


################################ OBJECTS ################################

LABELS = [
    'acid', 'acoustic', 'africa', 'afternoon', 'airplane', 'ambient', 'angelic', 'anger',
    'angst', 'arab', 'asia', 'atmospheric', 'autumn', 'bad mood', 'bbq', 'beach', 'beautiful',
    'bed', 'beer', 'berlin', 'biking', 'bleak', 'bliss', 'breakfast', 'breakup', 'bright',
    'broken', 'cabin', 'cafe', 'calm', 'caribbean', 'celebration', 'champagne', 'chaotic',
    'chill', 'choir', 'cinematic', 'city', 'city night', 'cleaning', 'clubbing', 'cocaine',
    'cocktail', 'coffee', 'cold', 'commuting', 'concentration', 'cool', 'cosmic', 'cowboy',
    'cozy', 'cry', 'cuddle', 'dance', 'dark', 'date', 'dawn', 'day', 'deep', 'depressed',
    'desire', 'despair', 'dirty', 'dramatic', 'dream', 'drinking', 'driving', 'drug', 'dusk',
    'early', 'ecstasy', 'eerie', 'encouraging', 'energetic', 'epic', 'erotic', 'ethereal',
    'euphoric', 'evening', 'excited', 'fearless', 'fight', 'fitness', 'flapper', 'focus',
    'forest', 'forgiveness', 'free spirit', 'friday', 'fuck', 'fucked up', 'fun', 'gaming',
    'garden', 'gentle', 'gloom', 'gloomy', 'goa', 'going out', 'good mood', 'good vibes',
    'grill', 'grimy', 'gritty', 'groovy', 'grunge', 'guilt', 'guitar', 'gym', 'hallucinating',
    'happiness', 'happy', 'havana', 'hazy', 'healing', 'heartbreak', 'heavy', 'hipster', 'home',
    'hopeless', 'horny', 'hype', 'hypnotic', 'ibiza', 'india', 'insane', 'inspiring', 'intense',
    'intimate', 'introspective', 'iran', 'irish', 'island', 'italy', 'jamaica', 'japan',
    'kaleidoscope', 'kiss', 'lake', 'late', 'latin america', 'lazy', 'lit', 'london',
    'loneliness', 'lonely', 'loud', 'love', 'lsd', 'magic', 'marathon', 'massage', 'meditation',
    'meditative', 'melancholic', 'mellow', 'memphis', 'memserizing', 'meth', 'mexico',
    'middle east', 'misery', 'monday', 'moon', 'morning', 'motivational', 'motorcycle',
    'mountain', 'moving on', 'mystical', 'nashville', 'nature', 'new orleans', 'new york',
    'night', 'nocturnal', 'noise', 'nomad', 'ocean', 'office', 'optimistic', 'painting',
    'paradise', 'paris', 'party', 'passion', 'peaceful', 'pensive', 'piano', 'porch', 'powerful',
    'psychedelic', 'rainy', 'rave', 'reading', 'reflective', 'relax', 'roadtrip', 'romantic',
    'running', 'sad', 'saturday', 'sea', 'sedating', 'seductive', 'sensual', 'sentimental',
    'serene', 'sex', 'sky', 'sleep', 'slow', 'slumber', 'smoking', 'smooth', 'snow', 'soft',
    'somber', 'soothing', 'sorrow', 'soulful', 'southern', 'space', 'spacey', 'spiritual',
    'spring', 'steamy', 'stimulating', 'stoner', 'storm', 'strings', 'stroll', 'study', 'summer',
    'sun', 'sunday', 'sunny', 'sunrise', 'sunset', 'sunshine', 'surf', 'surreal', 'tender',
    'tequila', 'thursday', 'tranquil', 'tranquilizer', 'travel', 'tribal', 'trip', 'trippy',
    'tropical', 'tuesday', 'upbeat', 'uplifting', 'urban', 'vibrant', 'violin', 'visceral',
    'vodka', 'walking', 'wandering', 'wanderlust', 'warm', 'wedding', 'wednesday', 'weed',
    'weekend', 'whiskey', 'wine', 'winter', 'woods', 'work', 'workout', 'writing', 'yacht',
    'yoga', 'zen'
]




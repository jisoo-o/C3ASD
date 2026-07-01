import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
from utils.corruption_utils import AudioCorruption, VisualCorruption

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)

def load_audio(data, dataPath, numFrames, audioAug, audioSet = None, audio_corruptor=None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    
    
    # Apply corruption if specified (for testing)
    if audio_corruptor is not None:
        audio = audio_corruptor.add_noise(audio)
    elif audioAug == True:
        # Original augmentation (for training)
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    return audio

def load_visual(data, dataPath, numFrames, visualAug, visual_corruptor=None): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    
    if visualAug == True and visual_corruptor is None:
        # Original augmentation (for training)
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    
    faces = numpy.array(faces)
    
    # Apply corruption if specified (for testing)
    if visual_corruptor is not None:
        faces = visual_corruptor.apply_corruption(faces)
    
    return faces


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)               
        start = 0       
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end     

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')            
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))  
            visualFeatures.append(load_visual(data, self.visualPath,numFrames, visualAug = True))
            labels.append(load_label(data, numFrames))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)


class corrupted_test_loader(object):
    """
    Test loader with corruption support for robustness evaluation.
    """
    def __init__(self, trialFileName, audioPath, visualPath,
                 audio_corruption_config=None,
                 visual_corruption_config=None,
                 **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()

        # Initialize corruption modules
        self.audio_corruptor = None
        self.visual_corruptor = None
        
        if audio_corruption_config is not None:
            from utils.corruption_utils import DEMAND_FOLDER_MAP
            noise_type = audio_corruption_config.get('noise_type', 'babble')
            snr = audio_corruption_config.get('snr', -10)
            snr_range = audio_corruption_config.get('snr_range', (snr, snr))
            
            # Build noise path:
            # DEMAND types mapped to actual folder names (e.g. demand_park -> NPARK)
            # MUSAN types used directly (e.g. babble -> babble/)
            noise_base = kwargs.get('noise_base_path', '/path/to/noise')
            folder_name = DEMAND_FOLDER_MAP.get(noise_type, noise_type)
            noise_path = os.path.join(noise_base, folder_name)
            
            self.audio_corruptor = AudioCorruption(noise_path, snr_range)
            print(f"Audio corruption: {noise_type} -> folder '{folder_name}', SNR range: {snr_range}")
        
        if visual_corruption_config is not None:
            corruption_type = visual_corruption_config.get('type', 'object_occlusion')
            corruption_prob = visual_corruption_config.get('prob', 1.0)
            max_freq = visual_corruption_config.get('max_freq', 1)
            patch_scale = visual_corruption_config.get('patch_scale', 0.5)
            occlusion_path = kwargs.get('occlusion_path', './occlusion_patch')

            self.visual_corruptor = VisualCorruption(
                corruption_type=corruption_type,
                occlusion_path=occlusion_path,
                corruption_prob=corruption_prob,
                max_freq=max_freq,
                patch_scale=patch_scale
            )
            print(f"Visual corruption enabled: {corruption_type}, prob: {corruption_prob}, max_freq: {max_freq}, patch_scale: {patch_scale}")

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        
        # Load data with corruption
        audioFeatures = [load_audio(data, self.audioPath, numFrames,
                                    audioAug=False, audioSet=audioSet,
                                    audio_corruptor=self.audio_corruptor)]
        visualFeatures = [load_visual(data, self.visualPath, numFrames,
                                      visualAug=False,
                                      visual_corruptor=self.visual_corruptor)]
        labels = [load_label(data, numFrames)]

        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
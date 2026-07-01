import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE

from loss import lossAV, lossV, lossA, InterModalityConsistency, IntraModalityConsistency, PredictionLevelConsistency
from model.Model import ASD_Model

class ASD(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, **kwargs):
        super(ASD, self).__init__()        
        self.model = ASD_Model().cuda()
        self.lossAV = lossAV().cuda()
        self.lossV = lossV().cuda()
        self.lossA = lossA().cuda()
        
        self.interModalityLoss = InterModalityConsistency().cuda()
        self.intraModalityLoss = IntraModalityConsistency(temperature=kwargs.get('intra_temperature', 0.07)).cuda()
        self.predictionLevelLoss = PredictionLevelConsistency().cuda()
        
        self.lambda_inter = kwargs.get('lambda_inter', 0.0)
        self.lambda_intra_audio = kwargs.get('lambda_intra_audio', 0.0)
        self.lambda_intra_visual = kwargs.get('lambda_intra_visual', 0.0)
        self.lambda_pred = kwargs.get('lambda_pred', 0.0)
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))
        if self.lambda_inter > 0:
            print(f"Inter-modality consistency enabled with weight: {self.lambda_inter}")
        if self.lambda_intra_audio > 0 or self.lambda_intra_visual > 0:
            print(f"Intra-modality consistency enabled - Audio: {self.lambda_intra_audio}, Visual: {self.lambda_intra_visual}")
        if self.lambda_pred > 0:
            print(f"Prediction-level consistency enabled with weight: {self.lambda_pred}")

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)  # StepLR
        index, top1, lossV, lossAV, loss = 0, 0, 0, 0, 0
        loss_inter_total, loss_intra_a_total, loss_intra_v_total, loss_pred_total = 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        r = 1.3 - 0.02 * (epoch - 1)
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()

            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda())
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())

            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsV = self.model.forward_visual_backend(visualEmbed)

            labels = labels[0].reshape((-1)).cuda()
            nlossAV, predScoreAV, _, prec = self.lossAV.forward(outsAV, labels, r)
            nlossV = self.lossV.forward(outsV, labels, r)
            nloss = nlossAV + 0.5 * nlossV
            
            loss_inter = torch.tensor(0.0).cuda()
            if self.lambda_inter > 0:
                audioEmbed_flat = torch.reshape(audioEmbed, (-1, 128))
                visualEmbed_flat = torch.reshape(visualEmbed, (-1, 128))
                loss_inter = self.interModalityLoss(audioEmbed_flat, visualEmbed_flat, labels)
                nloss += self.lambda_inter * loss_inter
            
            loss_intra_a = torch.tensor(0.0).cuda()
            loss_intra_v = torch.tensor(0.0).cuda()
            B, T, _ = audioEmbed.shape
            track_ids = torch.arange(B, device=labels.device).unsqueeze(1).expand(B, T).reshape(-1)
            
            if self.lambda_intra_audio > 0:
                audioEmbed_flat = torch.reshape(audioEmbed, (-1, 128))
                loss_intra_a = self.intraModalityLoss(audioEmbed_flat, labels, group_ids=track_ids)
                nloss += self.lambda_intra_audio * loss_intra_a
            
            if self.lambda_intra_visual > 0:
                visualEmbed_flat = torch.reshape(visualEmbed, (-1, 128))
                loss_intra_v = self.intraModalityLoss(visualEmbed_flat, labels, group_ids=track_ids)
                nloss += self.lambda_intra_visual * loss_intra_v
            
            loss_pred = torch.tensor(0.0).cuda()
            if self.lambda_pred > 0:
                outsA = self.model.forward_audio_backend(audioEmbed)
                predScoreA = F.softmax(self.lossA.FC(outsA), dim=-1)
                predScoreV = F.softmax(self.lossV.FC(outsV), dim=-1)
                loss_pred = self.predictionLevelLoss(predScoreAV, predScoreA, predScoreV)
                nloss += self.lambda_pred * loss_pred

            lossV += nlossV.detach().cpu().numpy()
            lossAV += nlossAV.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            if self.lambda_inter > 0:
                loss_inter_total += loss_inter.detach().cpu().numpy()
            if self.lambda_intra_audio > 0:
                loss_intra_a_total += loss_intra_a.detach().cpu().numpy()
            if self.lambda_intra_visual > 0:
                loss_intra_v_total += loss_intra_v.detach().cpu().numpy()
            if self.lambda_pred > 0:
                loss_pred_total += loss_pred.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            log_msg = time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] r: %2f, Lr: %5f, T: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
            " LV: %.5f, LAV: %.5f, L: %.5f, ACC: %2.2f%%"  %(lossV/(num), lossAV/(num), loss/(num), 100 * (top1/index))
            if self.lambda_inter > 0:
                log_msg += ", In: %.5f"%(loss_inter_total/num)
            if self.lambda_intra_audio > 0:
                log_msg += ", InA: %.5f"%(loss_intra_a_total/num)
            if self.lambda_intra_visual > 0:
                log_msg += ", InV: %.5f"%(loss_intra_v_total/num)
            if self.lambda_pred > 0:
                log_msg += ", P: %.5f"%(loss_pred_total/num)
            log_msg += " \r"
            sys.stderr.write(log_msg)
            sys.stderr.flush()  

        sys.stdout.write("\n")      

        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        result = str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout)
        mAP = float(result.split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)

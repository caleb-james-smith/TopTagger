import sys
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from MVAcommon import *
try:
    import cv2
except ImportError:
    #sys.path.append("../../opencv/lib/")
    sys.path.append("/uscms_data/d3/pastika/zinv/dev/CMSSW_7_4_8/src/opencv/lib/")
    import cv2
import optparse

parser = optparse.OptionParser("usage: %prog [options]\n")

parser.add_option ('-o', "--opencv", dest='opencv', action='store_true', help="Run using opencv RTrees")
parser.add_option ('-n', "--noRoc", dest='noROC', action='store_true', help="Do not calculate ROC to save time")

options, args = parser.parse_args()

dg = DataGetter()

print "PROCESSING TRAINING DATA"

samplesToRun = ["trainingTuple_division_0_TTbarSingleLep_training.root", "trainingTuple_division_0_ZJetsToNuNu_training.root"]
#samplesToRun = ["trainingTuple_division_0_TTbarSingleLep_training.root"]

doPtWgt = True 

inputData = []
inputAnswer = []
inputWgts = []

c = ROOT.TCanvas("c1","c1",800,800)

for datasetName in samplesToRun:
    dataset = ROOT.TFile.Open(datasetName)

    hPtMatch   = ROOT.TH1D("hPtMatch" + datasetName, "hPtMatch", 50, 0.0, 2000.0)
    hPtMatch.Sumw2()
    hPtNoMatch = ROOT.TH1D("hPtNoMatch" + datasetName, "hPtNoMatch", 50, 0.0, 2000.0)
    hPtNoMatch.Sumw2()

    hWgtMatch   = ROOT.TH1D("hWgtMatch" + datasetName, "hWgtMatch", 50, 0.01, 2000.0)
    hWgtMatch.Sumw2()
    hWgtNoMatch = ROOT.TH1D("hWgtNoMatch" + datasetName, "hWgtNoMatch", 200000, 0.01, 2000.0)
    hWgtNoMatch.Sumw2()

    Nevts = 0
    for event in dataset.slimmedTuple:
#        if event.Njet<4 : continue
        if Nevts >= NEVTS:
            break
        Nevts +=1
        for i in xrange(len(event.genConstiuentMatchesVec)):
            if event.genConstiuentMatchesVec[i] == 3:
                hPtMatch.Fill(event.cand_pt[i], event.sampleWgt)
                hWgtMatch.Fill(event.sampleWgt)
#                hPtMatch.Fill(event.cand_pt[i])
#            elif event.genConstiuentMatchesVec[i] == 0 or event.genConstiuentMatchesVec[i] == 1:
            else:
                hPtNoMatch.Fill(event.cand_pt[i], event.sampleWgt)
                hWgtNoMatch.Fill(event.sampleWgt)
#                hPtNoMatch.Fill(event.cand_pt[i])
    
    Nevts = 0
    for event in dataset.slimmedTuple:
        if Nevts >= NEVTS:
            break
        Nevts +=1
        for i in xrange(len(event.cand_m)):
            nmatch = event.genConstiuentMatchesVec[i]
            if nmatch ==2 or nmatch ==1: continue
            inputData.append(dg.getData(event, i))
            inputAnswer.append(int(nmatch == 3))

            if not doPtWgt:
               inputWgts.append(event.sampleWgt)
            else:
               if nmatch == 3:
                   if hPtMatch.GetBinContent(hPtMatch.FindBin(event.cand_pt[i])) > 10:
                       inputWgts.append(1.0 / hPtMatch.GetBinContent(hPtMatch.FindBin(event.cand_pt[i])) * event.sampleWgt)
#                       inputWgts.append(1.0 / hPtMatch.GetBinContent(hPtMatch.FindBin(event.cand_pt[i])))
                   else:
                       inputWgts.append(0.0)
               else:
                   if hPtNoMatch.GetBinContent(hPtNoMatch.FindBin(event.cand_pt[i])) > 10:
                       inputWgts.append(1.0 / hPtNoMatch.GetBinContent(hPtNoMatch.FindBin(event.cand_pt[i])) * event.sampleWgt)
#                       inputWgts.append(1.0 / hPtNoMatch.GetBinContent(hPtNoMatch.FindBin(event.cand_pt[i])))
                   else:
                       inputWgts.append(0.0)

    c.SetLogy()
    c.SetLogx(False)
    hPtMatch.Draw()
    c.Print("hPtMatch"+datasetName+"_trn.png") 
    hPtNoMatch.Draw()
    c.Print("hPtNoMatch"+datasetName+"_trn.png") 

#    c.SetLogy()
#    c.SetLogx()
#    hWgtMatch.SetMarkerSize(4)
#    hWgtMatch.Draw()
#    c.Print("hWgtMatch"+datasetName+"_trn.png") 
#    hWgtNoMatch.SetMarkerSize(4)
#    hWgtNoMatch.Draw()
#    c.Print("hWgtNoMatch"+datasetName+"_trn.png") 

                
npyInputData = numpy.array(inputData, numpy.float32)
npyInputAnswer = numpy.array(inputAnswer, numpy.float32)
npyInputWgts = numpy.array(inputWgts, numpy.float32)

nSig = npyInputWgts[npyInputAnswer==1].sum()
nBg = npyInputWgts[npyInputAnswer==0].sum()

print "before norm   nSig : %f  nBg : %f" % (nSig, nBg)

#Equalize the relative weights of signal and bg
for i in xrange(len(npyInputAnswer)):
    if npyInputAnswer[i] == 0:
        npyInputWgts[i] *= nSig/nBg
#    if npyInputAnswer[i] == 0:
#       npyInputWgts[i] *= 1./nBg
#    else:
#       npyInputWgts[i] *= 1./nSig

nSig = npyInputWgts[npyInputAnswer==1].sum()
nBg = npyInputWgts[npyInputAnswer==0].sum()

print "after norm   nSig : %f  nBg : %f" % (nSig, nBg)

#randomize input data
perms = numpy.random.permutation(npyInputData.shape[0])
npyInputData = npyInputData[perms]
npyInputAnswer = npyInputAnswer[perms]
npyInputWgts = npyInputWgts[perms]

print "TRAINING MVA"

if options.opencv:
    print "OPENCV ..."
    clf = cv2.ml.RTrees_create()

    n_estimators = 100
    clf.setTermCriteria((cv2.TERM_CRITERIA_COUNT, n_estimators, 0.1))
    #clf.setMaxCategories(2)
    clf.setMaxDepth(12)
    #clf.setMinSampleCount(5)

    #make opencv TrainData container
    cvTrainData = cv2.ml.TrainData_create(npyInputData, cv2.ml.ROW_SAMPLE, npyInputAnswer, sampleWeights = npyInputWgts)

    clf.train(cvTrainData)

    clf.save("TrainingOutput.model")

else:
    print "SCIKIT-LEARN ..."
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs = 4)
#    clf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs = 4)
#    clf = RandomForestClassifier(n_estimators=400, max_depth=15, n_jobs = 4)
#    clf = RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs = 4)
    #clf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs = 4)
    #clf = AdaBoostRegressor(n_estimators=100)
    #clf = GradientBoostingClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=0)
    #clf = GradientBoostingRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=0, loss='ls')
    #clf = DecisionTreeRegressor()
    #clf = DecisionTreeClassifier()
    #clf = svm.SVC()

    clf = clf.fit(npyInputData, npyInputAnswer, npyInputWgts)

    #Dump output from training
    fileObject = open("TrainingOutput.pkl",'wb')
    out = pickle.dump(clf, fileObject)
    fileObject.close()

    # Plot feature importance
    feature_importance = clf.feature_importances_
    feature_names = numpy.array(dg.getList())
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = numpy.argsort(feature_importance)
    
    #try to plot it with matplotlib
    try:
        import matplotlib.pyplot as plt
    
        # make importances relative to max importance
        pos = numpy.arange(sorted_idx.shape[0]) + .5
        #plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, feature_names[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        #plt.show()
        plt.savefig("feature_importance.png")
    except ImportError:
        #I guess no matplotlib is installed, just print to screen?
        featureImportanceandNames = zip(feature_names, feature_importance)
        print [featureImportanceandNames[a] for a in sorted_idx].reverse()

print "TRAINING DONE!"

/*
 * @Author: W.jy
 * @Date: 2021-08-07 19:21:06
 * @LastEditTime: 2022-10-28 09:36:11
 * @FilePath: \c_cpp\test_hmm.cpp
 * @Description:
 */
#ifdef __cplusplus
extern "C" {
#endif
#include "hmm.h"
#ifdef __cplusplus
}
#endif
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <thread>

#include "HMMBaseAlgorithm.h"

using namespace std;
// Train || Test
#define TRAIN_OR_TEST 1
#define TRAIN_TIMES 100

const string trainDataPrefix = "seq_model_0";
const string trainModelPrefix = "train_model_";
const string testDataFile1 = "testing_data1.txt";
const string testDataFile2 = "testing_data2.txt";
const string testDataAnswer = "testing_answer.txt";

void trainModelFunc(const string &modelFile, const string &dataFile, int times,
                    const string &saveFile, int id = 0);
double getProbFromModel(const string &modelFile, const string &sample);
double testModel(void);

int main() {
  vector<double> vecTestResult;
  int operations = 0;
  ofstream accuracyFos;
  accuracyFos.open("accuracyRslt.txt");
  while (vecTestResult.size() < 2 || vecTestResult.back() < 0.7 ||
         abs(vecTestResult.back() - vecTestResult[vecTestResult.size() - 2]) >
             0.001) {
    vector<string> vecTrainData(5);
    vector<string> vecTrainModel(5);
    vector<thread> vecTrainThread;
    for (int seq = 1; seq <= 5; ++seq) {
      vecTrainData[seq - 1] = trainDataPrefix + to_string(seq) + ".txt";
      vecTrainModel[seq - 1] = trainModelPrefix + to_string(seq) + ".txt";
      vecTrainThread.push_back(thread(trainModelFunc, vecTrainModel[seq - 1],
                                      vecTrainData[seq - 1], 1,
                                      vecTrainModel[seq - 1], seq));
      // trainModelFunc(trainModel, trainData, TRAIN_TIMES, trainModel, seq);
    }
    for (auto &it : vecTrainThread) it.join();
    cout << "Train && Test : " << ++operations << endl;
    accuracyFos << testModel() << endl;
  }
  accuracyFos.close();
  return 0;
}

int singleMain() {
  /*
  HMM hmms[5];
  load_models( "modellist.txt", hmms, 5);
  dump_models( hmms, 5);
*/

#if (TRAIN_OR_TEST)
  vector<string> vecTrainData(5);
  vector<string> vecTrainModel(5);
  vector<thread> vecTrainThread;
  for (int seq = 1; seq <= 5; ++seq) {
    vecTrainData[seq - 1] = trainDataPrefix + to_string(seq) + ".txt";
    vecTrainModel[seq - 1] = trainModelPrefix + to_string(seq) + ".txt";
    vecTrainThread.push_back(thread(trainModelFunc, vecTrainModel[seq - 1],
                                    vecTrainData[seq - 1], TRAIN_TIMES,
                                    vecTrainModel[seq - 1], seq));
    // trainModelFunc(trainModel, trainData, TRAIN_TIMES, trainModel, seq);
  }
  for (auto &it : vecTrainThread) it.join();
#else
  int rightCnt = 0, errCnt = 0, totalCnt = 0;
  string tmpStr, tmpAnsStr;
  vector<string> vecModelFile(5);
  vector<HMM> vecHMMModel(5);

  for (int seq = 1; seq <= 5; ++seq)
    vecModelFile[seq - 1] = trainModelPrefix + to_string(seq) + ".txt";

  for (int seq = 1; seq <= 5; ++seq)
    loadHMM(&vecHMMModel[seq - 1], vecModelFile[seq - 1].c_str());

  ifstream fis;
  fis.open(testDataFile2);

  ifstream fisAnswer;
  fisAnswer.open(testDataAnswer);

  while (fis >> tmpStr) {
    ++totalCnt;

    double tmpMaxDouble = 0.0;
    double tmpDouble = 0.0;
    int tmpModelIndex = 0;
    for (int seq = 1; seq <= 5; ++seq) {
      double tmpDebugDouble =
          HMMBaseAlgorithm::decoder(vecHMMModel[seq - 1], tmpStr);
      tmpDouble = max(tmpDouble, tmpDebugDouble);
      if (tmpDouble > tmpMaxDouble) {
        tmpModelIndex = seq;
        tmpMaxDouble = tmpDouble;
      }
    }
    fisAnswer >> tmpAnsStr;
    tmpAnsStr[7] - '0' == tmpModelIndex ? rightCnt++ : errCnt++;
    cout << totalCnt << " "
         << "ACC : " << (double)rightCnt / totalCnt << endl;
  }
  fisAnswer.close();
  fis.close();

  cout << "ACC : " << (double)rightCnt / totalCnt << endl;
#endif
  system("pause");
  return 0;
}

double testModel(void) {
  int rightCnt = 0, errCnt = 0, totalCnt = 0;
  string tmpStr, tmpAnsStr;
  vector<string> vecModelFile(5);
  vector<HMM> vecHMMModel(5);

  for (int seq = 1; seq <= 5; ++seq)
    vecModelFile[seq - 1] = trainModelPrefix + to_string(seq) + ".txt";

  for (int seq = 1; seq <= 5; ++seq)
    loadHMM(&vecHMMModel[seq - 1], vecModelFile[seq - 1].c_str());

  ifstream fis;
  fis.open(testDataFile1);

  ifstream fisAnswer;
  fisAnswer.open(testDataAnswer);

  while (fis >> tmpStr) {
    ++totalCnt;

    double tmpMaxDouble = 0.0;
    double tmpDouble = 0.0;
    int tmpModelIndex = 0;
    for (int seq = 1; seq <= 5; ++seq) {
      double tmpDebugDouble =
          HMMBaseAlgorithm::decoder(vecHMMModel[seq - 1], tmpStr);
      tmpDouble = max(tmpDouble, tmpDebugDouble);
      if (tmpDouble > tmpMaxDouble) {
        tmpModelIndex = seq;
        tmpMaxDouble = tmpDouble;
      }
    }
    fisAnswer >> tmpAnsStr;
    tmpAnsStr[7] - '0' == tmpModelIndex ? rightCnt++ : errCnt++;
    // cout << totalCnt << " "
    // << "ACC : " << (double)rightCnt / totalCnt << endl;
  }
  fisAnswer.close();
  fis.close();
  cout << "ACC : " << (double)rightCnt / totalCnt << endl;
  return (double)rightCnt / totalCnt;
}

void trainModelFunc(const string &modelFile, const string &dataFile, int times,
                    const string &saveFile, int id) {
  HMM hmm_initial;
  // cout << "Load init_model ..." << endl;
  loadHMM(&hmm_initial, modelFile.c_str());
  // dumpHMM(stderr, &hmm_initial);
  // cout << "Load init_model success!" << endl;

  HMMBaseAlgorithm hmmTrain(hmm_initial, 50);

  while (times--) {
    ifstream fis;
    fis.open(dataFile);
    string tmpStr;
    int lineCnt = 0;
    while (fis >> tmpStr) {
      // cout << id ;
      lineCnt++;
      hmmTrain.train(tmpStr);
    }
    fis.close();

    hmmTrain.updateModel();
    save_model(hmm_initial, modelFile.c_str());
    // cout << "Model " << id << " Left " << times << " Times." << endl;
  }
}

double getProbFromModel(const string &modelFile, const string &sample) {
  HMM hmm_initial;
  loadHMM(&hmm_initial, modelFile.c_str());
  return HMMBaseAlgorithm::decoder(hmm_initial, sample);
}
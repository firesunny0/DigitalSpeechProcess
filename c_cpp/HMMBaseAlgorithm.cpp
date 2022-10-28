/*
 * @Author: W.jy
 * @Date: 2021-08-07 19:57:53
 * @LastEditTime: 2022-10-27 22:15:03
 * @FilePath: \c_cpp\HMMBaseAlgorithm.cpp
 * @Description:
 */

#include "HMMBaseAlgorithm.h"

#include <assert.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

std::vector<std::vector<double>> HMMBaseAlgorithm::_vecDelta(
    50, std::vector<double>(5));

HMMBaseAlgorithm::HMMBaseAlgorithm(HMM &hmm, int maxSampleLen)
    : _trainedTimes(0), _hmm(hmm) {
  assert(nullptr != (&_hmm));
  // alloc for vector (without exception)
  int N = _hmm.state_num;

  _vecAlpha.resize(maxSampleLen, vector<double>(N));
  _vecBeta.resize(maxSampleLen, vector<double>(N));
  _vecGamma.resize(maxSampleLen, vector<double>(N));
  _vecEpsilon.resize(maxSampleLen,
                     vector<vector<double>>(N, vector<double>(N)));

  _vecGammaSum_1.resize(_hmm.state_num);
  _vecGammaSum.resize(_hmm.state_num);
  _vec2GammaSum_ok.resize(_hmm.state_num, vector<double>(_hmm.state_num));
  _vec2EpsilonSum.resize(_hmm.state_num, vector<double>(_hmm.state_num));
  // cout << "Ctor : HMMBaseAlgorithm" << endl;
}

HMMBaseAlgorithm::~HMMBaseAlgorithm() {}

double HMMBaseAlgorithm::decoder(HMM &hmm, const string &sample) {
  // Initialization
  for (int i = 0; i < hmm.state_num; ++i)
    _vecDelta[0][i] = hmm.initial[i] * hmm.observation[i][1];
  // Recursion
  for (int t = 1; t < sample.size(); ++t) {
    for (int j = 0; j < hmm.state_num; ++j) {
      double tmpDouble = 0.0;
      for (int i = 0; i < hmm.state_num; ++i)
        tmpDouble = max(tmpDouble, _vecDelta[t - 1][i] * hmm.transition[i][j]);
      _vecDelta[t][j] = tmpDouble * hmm.observation[j][sample[t] - 'A'];
    }
  }
  // get Max probability in _vecDelta[t][i]  (i from 0 to N)
  double tmpDouble = 0.0;
  for (auto i = 0; i < hmm.state_num; ++i)
    tmpDouble = max(tmpDouble, _vecDelta[sample.size() - 1][i]);
  return tmpDouble;
}

void HMMBaseAlgorithm::updateModel() {
  // update params PI of the model
  for (int i = 0; i < _hmm.state_num; i++)
    _hmm.initial[i] = _vecGammaSum_1[i] / _trainedTimes;
  // update transition matrix
  for (int i = 0; i < _hmm.state_num; ++i)
    for (int j = 0; j < _hmm.state_num; ++j)
      _hmm.transition[i][j] = _vec2EpsilonSum[i][j] / _vecGammaSum[i];
  // update emission matrix
  for (int i = 0; i < _hmm.state_num; ++i)
    for (int k = 0; k < _hmm.state_num; ++k)
      _hmm.observation[i][k] = _vec2GammaSum_ok[i][k] / _vecGammaSum[i];
}

void HMMBaseAlgorithm::saveInfoFromSample(const string &sample) {
  for (int i = 0; i < _hmm.state_num; ++i) _vecGammaSum_1[i] += _vecGamma[0][i];

  for (int t = 0; t < sample.size(); ++t)
    for (int i = 0; i < _hmm.state_num; ++i) _vecGammaSum[i] += _vecGamma[t][i];

  for (int t = 0; t < sample.size(); ++t)
    for (int i = 0; i < _hmm.state_num; ++i)
      for (int j = 0; j < _hmm.state_num; ++j)
        _vec2EpsilonSum[i][j] += _vecEpsilon[t][i][j];

  for (int t = 0; t < sample.size(); ++t)
    for (int i = 0; i < _hmm.state_num; ++i)
      _vec2GammaSum_ok[i][getStateFromObserv(sample[t])] += _vecGamma[t][i];
}

bool HMMBaseAlgorithm::train(const string &sample) {
  if (sample.empty()) {
    cout << "Input sample for training is empty!" << endl;
    return false;
  }

  forwardProc(sample);
  backwardProc(sample);
  calGamma(sample);
  calEpsilon(sample);

  saveInfoFromSample(sample);
  ++_trainedTimes;
  // cout << "Train finished : " << _trainedTimes << endl;
  return estimateModel();
}

void HMMBaseAlgorithm::forwardProc(const string &sample) {
  // Initialization
  for (int i = 0; i < _hmm.state_num; ++i)
    _vecAlpha[0][i] = _hmm.initial[i] * getEmission(i, sample[0]);
  // Induction
  for (int t = 0; t < sample.size() - 1; ++t) {
    for (int i = 0; i < _hmm.state_num; ++i) {
      double tmpSumAji = 0.0;
      for (int j = 0; j < _hmm.state_num; ++j)
        tmpSumAji += _vecAlpha[t][j] * _hmm.transition[j][i];
      _vecAlpha[t + 1][i] = tmpSumAji * getEmission(i, sample[t + 1]);
    }
  }
  /* 	// Termination
  double tmpSumP = 0.0;
  for(int i = 0; i < _hmm.state_num; ++i)
          tmpSumP += _vecAlpha[sample.size()-1][i];
  cout << "Forward Proc : sum is " << tmpSumP << endl; */
}

void HMMBaseAlgorithm::backwardProc(const string &sample) {
  // Initialization
  for (int i = 0; i < _hmm.state_num; ++i) _vecBeta[sample.size() - 1][i] = 1;
  // Induction
  for (int t = sample.size() - 2; t >= 0; --t) {
    for (int i = 0; i < _hmm.state_num; ++i) {
      _vecBeta[t][i] = 0.0;
      for (int j = 0; j < _hmm.state_num; ++j)
        _vecBeta[t][i] += _vecBeta[t + 1][j] * _hmm.transition[i][j] *
                          getEmission(j, sample[t + 1]);
    }
  }
  return;
  /* 	// Termination
  double tmpSumP = 0.0;
  for(int i = 0; i < _hmm.state_num; ++i)
          tmpSumP += _vecAlpha[0][i];
  cout << "Backward Proc : sum is " << tmpSumP << endl; */
}

void HMMBaseAlgorithm::calGamma(const string &sample) {
  for (int t = 0; t < sample.size(); ++t) {
    double tmpSumAlphaBeta = 0.0;

    for (int i = 0; i < _hmm.state_num; ++i) {
      // temp values are saved in _vecGamma[t][i]
      _vecGamma[t][i] = _vecAlpha[t][i] * _vecBeta[t][i];
      tmpSumAlphaBeta += _vecGamma[t][i];
    }

    for (int i = 0; i < _hmm.state_num; ++i) _vecGamma[t][i] /= tmpSumAlphaBeta;
  }
}

void HMMBaseAlgorithm::calEpsilon(const string &sample) {
  for (int t = 0; t < sample.size() - 1; ++t) {
    double tmpSumEij = 0.0;

    for (int i = 0; i < _hmm.state_num; ++i) {
      for (int j = 0; j < _hmm.state_num; ++j) {
        _vecEpsilon[t][i][j] = _vecAlpha[t][i] * _hmm.transition[i][j] *
                               getEmission(j, sample[t + 1]) *
                               _vecBeta[t + 1][j];
        tmpSumEij += _vecEpsilon[t][i][j];
      }
      // temp values are saved in _vecGamma[t][i]
    }

    for (int i = 0; i < _hmm.state_num; ++i)
      for (int j = 0; j < _hmm.state_num; ++j)
        _vecEpsilon[t][i][j] /= tmpSumEij;
  }
}

bool HMMBaseAlgorithm::estimateModel(void) const { return true; }

double HMMBaseAlgorithm::getEmission(int state, char observ) const {
  return _hmm.observation[state][observ - 'A'];
}
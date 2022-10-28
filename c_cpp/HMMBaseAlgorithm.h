/*
 * @Author: W.jy
 * @Date: 2021-08-07 19:57:39
 * @LastEditTime: 2021-08-08 21:20:18
 * @FilePath: \dsp_hw1\c_cpp\HMMBaseAlgorithm.h
 * @Description: 
 */
#ifndef __HMMBASEALGORITHM_H
#define __HMMBASEALGORITHM_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "hmm.h"

#ifdef __cplusplus
}
#endif

#include <vector>
#include <string>
#include <algorithm>

/* typedef struct{
   char *model_name;
   int state_num;					//number of state
   int observ_num;					//number of observation
   double initial[MAX_STATE];			//initial prob.
   double transition[MAX_STATE][MAX_STATE];	//transition prob.
   double observation[MAX_OBSERV][MAX_STATE];	//observation prob.
} HMM; */

class HMMBaseAlgorithm final
{
public:
	HMMBaseAlgorithm(HMM &hmm, int maxSampleLen);
	~HMMBaseAlgorithm();	

	// Train by a sample
	bool train(const std::string&);
	// update params after training by all samples 
	void updateModel(void);

	inline bool hasTrained(void) const
	{
		return _trainedTimes > 0;
	}

	// Decoder: Viterbi
	static double decoder(HMM& hmm, const std::string& sample);
	
private:
	static std::vector<std::vector<double> > _vecDelta;
	// hmm + sequences -> matrix Alpha(state number * sequences number)
	void forwardProc(const std::string& sample);
	// hmm + sequences -> matrix Beta(state number * sequences number)
	void backwardProc(const std::string& sample);
	// matrix Alpha && matrix Beta -> matrix Gamma(state number * sequences number)
	void calGamma(const std::string& sample);
	// matrix Alpha && matrix Beta && matrix a && matrix b && sequences -> matrix Epsilon
	// (sequences number - 1) Matrix(state number * state number)
	void calEpsilon(const std::string& sample);
	// recalGammaAndEpsilon
	bool estimateModel(void) const;
	// probability (state mapped to phoneme observ)
	double getEmission(int state, char observ) const;
	// save useful information from a sample 
	void saveInfoFromSample(const std::string&);
	// get state from observe
	inline int getStateFromObserv(char c) const
	{
		return std::max(0, static_cast<int>(c - 'A'));
	}

	std::vector<std::vector<double> > _vecAlpha;
	std::vector<std::vector<double> > _vecBeta;
	std::vector<std::vector<double> > _vecGamma;
	std::vector<std::vector<std::vector<double> > > _vecEpsilon;

	std::vector<double> _vecGammaSum_1;
	std::vector<double> _vecGammaSum;
	std::vector<std::vector<double> > _vec2GammaSum_ok;
	std::vector<std::vector<double> > _vec2EpsilonSum;

	int _trainedTimes;

	HMM & _hmm;
};

#endif

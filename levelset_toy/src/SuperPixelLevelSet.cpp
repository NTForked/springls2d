#include "SuperPixelLevelSet.h"
namespace aly {
	bool SuperPixelLevelSet::stepInternal() {
		float E = superPixels->updateClusters(labelImage, -1);
		float mx = superPixels->updateMaxColor(labelImage, -1);
		if (mSimulationIteration == 0) {
			maxClusterDistance = 0;
			meanClusterDistance = 0;
			int samples = 0;
			for (int n = 0;n < (int)activeList.size();n++) {
				int2 pos = activeList[n];
				if (swapLevelSet(pos).x <= 0.5f) {
					int label = swapLabelImage(pos.x, pos.y).x;
					if (label > 0) {
						samples++;
						float d = superPixels->distance(pos.x, pos.y, label - 1);
						maxClusterDistance = std::max(maxClusterDistance, d);
						meanClusterDistance += d;
					}
				}
			}
			meanClusterDistance /= (float)samples;
		}
		double remaining = mTimeStep;
		double t = 0.0;
		do {
			float timeStep = evolve(std::min(0.5f, (float)remaining));
			t += (double)timeStep;
			remaining = mTimeStep - t;
		} while (remaining > 1E-5f);
		mSimulationTime += t;
		mSimulationIteration++;
		if (cache.get() != nullptr) {
			updateOverlay();
			updateContour();
			Contour2D* contour = getContour();
			contour->setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
			cache->set((int)mSimulationIteration, *contour);
		}
		return (mSimulationTime<mSimulationDuration);
	}
	float SuperPixelLevelSet::evolve(float maxStep) {
#pragma omp parallel for
		for (int i = 0; i < (int)activeList.size(); i++) {
			int2 pos = activeList[i];
			superPixelMotion(pos.x, pos.y, i);
		}
		float timeStep = (float)maxStep;
		if (!clampSpeed) {
			float maxDelta = 0.0f;
			for (float delta : deltaLevelSet) {
				maxDelta = std::max(std::abs(delta), maxDelta);
			}
			const float maxSpeed = 0.999f;
			timeStep = (float)(maxStep * ((maxDelta > maxSpeed) ? (maxSpeed / maxDelta) : maxSpeed));
			//std::cout << "Max Delta " << maxDelta <<" Time Step "<<timeStep<<" Max Step "<<maxStep<< std::endl;
		}
		contourLock.lock();
		if (preserveTopology) {
			for (int nn = 0; nn < 4; nn++) {
#pragma omp parallel for
				for (int i = 0; i < (int)activeList.size(); i++) {
					int2 pos = activeList[i];
					applyForcesTopoRule(pos.x, pos.y, nn, i, timeStep);
				}
			}
		}
		else {
#pragma omp parallel for
			for (int i = 0; i < (int)activeList.size(); i++) {
				int2 pos = activeList[i];
				applyForces(pos.x, pos.y, i, timeStep);
			}
		}
		for (int band = 1; band <= maxLayers; band++) {
#pragma omp parallel for
			for (int i = 0; i < (int)activeList.size(); i++) {
				int2 pos = activeList[i];
				updateDistanceField(pos.x, pos.y, band);
			}
		}
#pragma omp parallel for
		for (int i = 0; i < (int)activeList.size(); i++) {
			int2 pos = activeList[i];
			plugLevelSet(pos.x, pos.y, i);
		}
		requestUpdateContour = false;// (getNumLabels() < 256);
		requestUpdateOverlay = true;
		contourLock.unlock();
#pragma omp parallel for
		for (int i = 0; i < (int)activeList.size(); i++) {
			int2 pos = activeList[i];
			swapLevelSet(pos.x, pos.y) = levelSet(pos.x, pos.y);
			swapLabelImage(pos.x, pos.y) = labelImage(pos.x, pos.y);
		}
		int deleted = deleteElements();
		int added = addElements();
		deltaLevelSet.resize(5 * activeList.size(), 0.0f);
		objectIds.resize(5 * activeList.size(), -1);
		return timeStep;
	}
	void SuperPixelLevelSet::superPixelMotion(int i, int j, size_t gid) {
		if (swapLevelSet(i, j).x > 0.5f) {
			return;
		}
		float4 grad;
		int activeLabels[5];
		float offsets[5];
		activeLabels[0] = swapLabelImage(i, j).x;
		activeLabels[1] = swapLabelImage(i + 1, j).x;
		activeLabels[2] = swapLabelImage(i - 1, j).x;
		activeLabels[3] = swapLabelImage(i, j + 1).x;
		activeLabels[4] = swapLabelImage(i, j - 1).x;
		float maxDiff = 0.0f;
		for (int index = 0;index < 5;index++) {
			int label = activeLabels[index];
			offsets[index] = (label > 0) ? superPixels->distance(i, j, label - 1) : -0.01f;
			if (label != activeLabels[0]) {
				if (offsets[index] > maxDiff) {
					maxDiff = offsets[index];
				}
			}
		}
		for (int index = 0;index < 5;index++) {
			int label = activeLabels[index];
			if (label == 0) {
				objectIds[gid * 5 + index] = 0;
				deltaLevelSet[gid * 5 + index] = 0;
			}
			else {
				float v11 = getSwapLevelSetValue(i, j, label);
				float v00 = getSwapLevelSetValue(i - 1, j - 1, label);
				float v01 = getSwapLevelSetValue(i - 1, j, label);
				float v10 = getSwapLevelSetValue(i, j - 1, label);
				float v21 = getSwapLevelSetValue(i + 1, j, label);
				float v20 = getSwapLevelSetValue(i + 1, j - 1, label);
				float v22 = getSwapLevelSetValue(i + 1, j + 1, label);
				float v02 = getSwapLevelSetValue(i - 1, j + 1, label);
				float v12 = getSwapLevelSetValue(i, j + 1, label);
				float DxNeg = v11 - v01;
				float DxPos = v21 - v11;
				float DyNeg = v11 - v10;
				float DyPos = v12 - v11;
				float DxNegMin = std::min(DxNeg, 0.0f);
				float DxNegMax = std::max(DxNeg, 0.0f);
				float DxPosMin = std::min(DxPos, 0.0f);
				float DxPosMax = std::max(DxPos, 0.0f);
				float DyNegMin = std::min(DyNeg, 0.0f);
				float DyNegMax = std::max(DyNeg, 0.0f);
				float DyPosMin = std::min(DyPos, 0.0f);
				float DyPosMax = std::max(DyPos, 0.0f);
				float GradientSqrPos = DxNegMax * DxNegMax + DxPosMin * DxPosMin + DyNegMax * DyNegMax + DyPosMin * DyPosMin;
				float GradientSqrNeg = DxPosMax * DxPosMax + DxNegMin * DxNegMin + DyPosMax * DyPosMax + DyNegMin * DyNegMin;

				float DxCtr = 0.5f * (v21 - v01);
				float DyCtr = 0.5f * (v12 - v10);

				float DxxCtr = v21 - v11 - v11 + v01;
				float DyyCtr = v12 - v11 - v11 + v10;
				float DxyCtr = (v22 - v02 - v20 + v00) * 0.25f;

				float numer = 0.5f * (DyCtr * DyCtr * DxxCtr - 2 * DxCtr * DyCtr
					* DxyCtr + DxCtr * DxCtr * DyyCtr);
				float denom = DxCtr * DxCtr + DyCtr * DyCtr;
				float kappa = 0;
				const float maxCurvatureForce = 10.0f;
				if (fabs(denom) > 1E-5f) {
					kappa = curvatureParam.toFloat() * numer / denom;
				}
				else {
					kappa = curvatureParam.toFloat() * numer * sign(denom) * 1E5f;
				}
				if (kappa < -maxCurvatureForce) {
					kappa = -maxCurvatureForce;
				}
				else if (kappa > maxCurvatureForce) {
					kappa = maxCurvatureForce;
				}
				// Force should be negative to move level set outwards if pressure is
				// positive
				float force = pressureParam.toFloat() *aly::clamp((maxDiff - offsets[index])/ meanClusterDistance,-1.0f,1.0f);
				float pressure = 0;
				if (force > 0) {
					pressure = -force * std::sqrt(GradientSqrPos);
				}
				else if (force < 0) {
					pressure = -force * std::sqrt(GradientSqrNeg);
				}
				objectIds[gid * 5 + index] = label;
				deltaLevelSet[5 * gid + index] = kappa + pressure;
			}
		}
	}
	void SuperPixelLevelSet::setSuperPixels(const std::shared_ptr<SuperPixels>& superPixels) {
		this->superPixels = superPixels;
		Image1i initLabels = superPixels->getLabelImage();
		{
			auto range = initLabels.range();
		}
		for (int j = 0;j < initLabels.height;j++) {
			for (int i = 0;i < initLabels.width;i++) {
				if (i < 1 || j < 1 || i >= initLabels.width - 1 || j >= initLabels.height - 1) {
					initLabels(i, j) = int1(0);
				}
				else {
					initLabels(i, j).x++;
				}
			}
		}
		setInitial(initLabels);
	}
	SuperPixelLevelSet::SuperPixelLevelSet(const std::shared_ptr<SpringlCache2D>& cache) :MultiActiveContour2D(cache) {

	}
	SuperPixelLevelSet::SuperPixelLevelSet(const std::string& name, const std::shared_ptr<SpringlCache2D>& cache) : MultiActiveContour2D(name, cache) {

	}
}
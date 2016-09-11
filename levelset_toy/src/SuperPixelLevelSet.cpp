#include "SuperPixelLevelSet.h"
namespace aly {
	void SuperPixelLevelSet::setup(const aly::ParameterPanePtr& pane) {
		pane->addNumberField("Pressure Weight", pressureParam, Float(-2.0f), Float(2.0f));
		pane->addNumberField("Curvature Weight", curvatureParam, Float(0.0f), Float(4.0f));
		pane->addNumberField("Prune Interval", pruneInterval, Integer(1), Integer(257));
		pane->addNumberField("Split Interval", splitInterval, Integer(1), Integer(257));
		pane->addNumberField("Split Threshold", splitThreshold,Float(0.0f), Float(200.0f));
		pane->addCheckBox("Preserve Topology", preserveTopology);
		pane->addCheckBox("Clamp Speed", clampSpeed);
		pane->addCheckBox("Dynamic Cluster Centers", updateClusterCenters);
		pane->addCheckBox("Dynamic Compactness", updateCompactness);
	}

	bool SuperPixelLevelSet::updateOverlay() {
		if (requestUpdateOverlay) {
			ImageRGBA& overlay = contour.overlay;
			overlay.resize(labelImage.width, labelImage.height);
#pragma omp parallel for
			for (int j = 0; j <overlay.height; j++) {
				for (int i = 0; i < overlay.width; i++) {
					float d = levelSet(i, j).x;
					int l = labelImage(i, j).x;
					Color c = (l>0)?superPixels->getColorCenter(l-1):Color(0,0,0,0);//getColor(l);
					RGBAf rgba = c.toRGBAf();
					overlay(i, j) = ToRGBA(rgba);
				}
			}
			requestUpdateOverlay = false;
			return true;
		}
		else {
			return false;
		}
	}

	bool SuperPixelLevelSet::init() {
		int2 dims = initialLevelSet.dimensions();
		if (dims.x == 0 || dims.y == 0||superPixels.get()==nullptr)return false;
		*superPixels = initSuperPixels;
		mSimulationDuration = std::max(dims.x, dims.y)*0.5f;
		mSimulationIteration = 0;
		mSimulationTime = 0;
		mTimeStep = 1.0f;
		levelSet.resize(dims.x, dims.y);
		labelImage.resize(dims.x, dims.y);
		swapLevelSet.resize(dims.x, dims.y);
		swapLabelImage.resize(dims.x, dims.y);
#pragma omp parallel for
		for (int i = 0; i < initialLevelSet.size(); i++) {
			float val = clamp(initialLevelSet[i], 0.0f, (maxLayers + 1.0f));
			levelSet[i] = val;
			swapLevelSet[i] = val;
		}
		labelImage = initialLabels;
		swapLabelImage = initialLabels;
		std::set<int> labelSet;
		int L = 1;
		for (int1 l : initialLabels.data) {
			if (l.x != 0) {
				labelSet.insert(l.x);
				L = std::max(L, l.x + 1);
			}
		}

		forceIndexes.resize(L, -1);
		labelList.clear();
		labelList.assign(labelSet.begin(), labelSet.end());
		for (int i = 0;i < (int)labelList.size();i++) {
			forceIndexes[labelList[i]] = i;
		}
		L = (int)labelList.size();
		if (L<256) {
			lineColors.clear();
			lineColors[0] = RGBAf(0.0f, 0.0f, 0.0f, 0.0f);
			int CL = std::min(256, L);
			for (int i = 0;i < L;i++) {
				int l = labelList[i];
				HSV hsv = HSV((l%CL) / (float)CL, 0.7f, 0.7f);
				lineColors[l] = HSVtoColor(hsv);
			}
		}
		else {
			if (lineColors.size() != L + 1) {
				lineColors.clear();
				lineColors[0] = RGBAf(0.0f, 0.0f, 0.0f, 0.0f);
				for (int i = 0;i < L;i++) {
					int l = labelList[i];
					HSV hsv = HSV(RandomUniform(0.0f, 1.0f), RandomUniform(0.5f, 1.0f), RandomUniform(0.5f, 1.0f));
					lineColors[l] = HSVtoColor(hsv);
				}
			}
		}
		rebuildNarrowBand();
		requestUpdateContour = true;
		requestUpdateOverlay = true;
		contour.clusterCenters = superPixels->getPixelCenters();
		contour.clusterColors = superPixels->getColorCenters();
		if (cache.get() != nullptr) {
			updateOverlay();
			updateContour();
			contour.setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
			cache->set((int)mSimulationIteration, contour);
		}
		return true;
	}
	void SuperPixelLevelSet::reinit() {
#pragma omp parallel for
		for (int j = 0;j < labelImage.height;j++) {
			int activeLabels[4];
			for (int i = 0;i < labelImage.width;i++) {
				int currentLabel = labelImage(i, j).x;
				activeLabels[0] = labelImage(i + 1, j).x;
				activeLabels[1] = labelImage(i - 1, j).x;
				activeLabels[2] = labelImage(i, j + 1).x;
				activeLabels[3] = labelImage(i, j - 1).x;
				float val = 1.0f;
				for (int n = 0;n < 4;n++) {
					if (currentLabel < activeLabels[n]) {
						val = 0.01f;
						break;
					}
				}
				levelSet(i, j) = float1(val);
				swapLevelSet(i, j) = float1(val);
			}
		}
		for (int band = 1; band <= 2 * maxLayers; band++) {
#pragma omp parallel for
			for (int j = 0;j < labelImage.height;j++) {
				for (int i = 0;i < labelImage.width;i++) {
					updateDistanceField(i, j, band);
				}
			}
		}
		for (int j = 0;j < labelImage.height;j++) {
			for (int i = 0;i < labelImage.width;i++) {
				if (i < 1 || j < 1 || i >= labelImage.width - 1 || j >= labelImage.height - 1) {
					labelImage(i, j) = int1(0);
				}
				else {
					labelImage(i, j).x++;
				}
			}
		}
		swapLabelImage = labelImage;
#pragma omp parallel for
		for (int i = 0; i < labelImage.size(); i++) {
			float val = clamp(levelSet[i], 0.0f, (maxLayers + 1.0f));
			levelSet[i] = val;
			swapLevelSet[i] = val;
		}
		std::set<int> labelSet;
		int L = 1;
		for (int1 l : labelImage.data) {
			if (l.x != 0) {
				labelSet.insert(l.x);
				L = std::max(L, l.x + 1);
			}
		}
		forceIndexes.resize(L, -1);
		labelList.clear();
		labelList.assign(labelSet.begin(), labelSet.end());
		for (int i = 0;i < (int)labelList.size();i++) {
			forceIndexes[labelList[i]] = i;
		}
		L = (int)labelList.size();
		if (L<256) {
			lineColors.clear();
			lineColors[0] = RGBAf(0.0f, 0.0f, 0.0f, 0.0f);
			int CL = std::min(256, L);
			for (int i = 0;i < L;i++) {
				int l = labelList[i];
				HSV hsv = HSV((l%CL) / (float)CL, 0.7f, 0.7f);
				lineColors[l] = HSVtoColor(hsv);
			}
		}
		else {
			if (lineColors.size() != L + 1) {
				lineColors.clear();
				lineColors[0] = RGBAf(0.0f, 0.0f, 0.0f, 0.0f);
				for (int i = 0;i < L;i++) {
					int l = labelList[i];
					HSV hsv = HSV(RandomUniform(0.0f, 1.0f), RandomUniform(0.5f, 1.0f), RandomUniform(0.5f, 1.0f));
					lineColors[l] = HSVtoColor(hsv);
				}
			}
		}
		contour.clusterCenters = superPixels->getPixelCenters();
		contour.clusterColors = superPixels->getColorCenters();
	}
	bool SuperPixelLevelSet::stepInternal() {
		if ((mSimulationIteration) % splitInterval.toInteger() == 0 && mSimulationIteration>0) {
			superPixels->splitRegions(labelImage, splitThreshold.toFloat(),-1);
			reinit();
		} else 
		if (mSimulationIteration % pruneInterval.toInteger() == 0&&mSimulationIteration>0) {
			superPixels->enforceLabelConnectivity(labelImage);
			reinit();
		}
		else {
			if (updateClusterCenters) {
				float E = superPixels->updateClusters(labelImage, -1);
				contour.clusterCenters = superPixels->getPixelCenters();
				contour.clusterColors = superPixels->getColorCenters();
			}
			if (updateCompactness) {
				float mx = superPixels->updateMaxColor(labelImage, -1);
			}
		}
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
			contour.setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
			cache->set((int)mSimulationIteration, contour);
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
		requestUpdateContour = true;
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
		initSuperPixels = *superPixels;
		Image1i initLabels = superPixels->getLabelImage();
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
	SuperPixelLevelSet::SuperPixelLevelSet(const std::shared_ptr<SpringlCache2D>& cache) :MultiActiveContour2D(cache),updateClusterCenters(true),updateCompactness(true), pruneInterval(Integer(16)), splitInterval(Integer(64)), splitThreshold(Float(40.0f)){

	}
	SuperPixelLevelSet::SuperPixelLevelSet(const std::string& name, const std::shared_ptr<SpringlCache2D>& cache) : MultiActiveContour2D(name, cache), updateClusterCenters(true), updateCompactness(true), pruneInterval(Integer(16)),splitInterval(Integer(64)), splitThreshold(Float(40.0f)) {

	}
}
/*
 * Copyright(C) 2016, Blake C. Lucas, Ph.D. (img.science@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "MultiActiveContour2D.h"
namespace aly {
	void MultiActiveContour2D::rebuildNarrowBand() {
		activeList.clear();
		for (int band = 1; band <= maxLayers; band++) {
#pragma omp parallel for
			for (int i = 0; i < (int)activeList.size(); i++) {
				int2 pos = activeList[i];
				updateDistanceField(pos.x, pos.y, band);
			}
		}
		for (int j = 0; j < swapLevelSet.height; j++) {
			for (int i = 0; i < swapLevelSet.width; i++) {
				if (swapLevelSet(i, j) <= MAX_DISTANCE) {
					activeList.push_back(int2(i, j));
				}
			}
		}
		deltaLevelSet.resize(5 * activeList.size(), 0.0f);
		objectIds.resize(5 * activeList.size(), -1);
	}
	void MultiActiveContour2D::plugLevelSet(int i, int j, size_t index) {
		int label = labelImage(i, j);
		float v11 = levelSet(i, j);
		int activeLabels[8];
		activeLabels[0] = labelImage(i + 1, j);
		activeLabels[1] = labelImage(i - 1, j);
		activeLabels[2] = labelImage(i, j + 1);
		activeLabels[3] = labelImage(i, j - 1);
		activeLabels[4] = labelImage(i - 1, j - 1);
		activeLabels[5] = labelImage(i + 1, j - 1);
		activeLabels[6] = labelImage(i - 1, j + 1);
		activeLabels[7] = labelImage(i + 1, j + 1);
		int count = 0;
		for (int index = 0;index < 8;index++) {
			if (label == activeLabels[index])count++;
		}
		//Pick any label other than this to fill the hole
		if (count == 0) {
			labelImage(i, j) = (i>0) ? activeLabels[1] : activeLabels[0];
			levelSet(i, j) = 3.0f;
		}
	}
	bool MultiActiveContour2D::updateOverlayImage() {
		if (updateOverlay) {
			ImageRGBA& overlay = contour.overlay;
			overlay.resize(labelImage.width, labelImage.height);
			for (int j = 0; j <labelImage.height; j++) {
				for (int i = 0; i < labelImage.width; i++) {
					float d = levelSet(i, j).x;
					int l = labelImage(i, j).x;
					Color c = getColor(l);
					RGBAf rgba = c.toRGBAf();
					rgba.w = 0.5f;
					rgba=aly::mix(RGBAf(0.0f, 0.0f, 0.0f, 1.0f), rgba,clamp(d/1.5f,0.0f,1.0f));
					overlay(i, j) = ToRGBA(rgba);
				}
			}
			updateOverlay = false;
			return true;
		}
		else {
			return false;
		}
	}
	Contour2D* MultiActiveContour2D::getContour() {
		if (updateIsoSurface) {
			std::lock_guard<std::mutex> lockMe(contourLock);
			isoContour.solve(levelSet, labelImage, labelList, contour.vertexes, contour.vertexLabels, contour.indexes, 0.0f, (preserveTopology) ? TopologyRule2D::Connect4 : TopologyRule2D::Unconstrained, Winding::Clockwise);
			updateIsoSurface = false;
		}
		return &contour;
	}
	MultiActiveContour2D::MultiActiveContour2D(const std::shared_ptr<SpringlCache2D>& cache) :Simulation("Active Contour 2D"),
		cache(cache), preserveTopology(false), clampSpeed(false), updateIsoSurface(
			false) {
		advectionParam = Float(1.0f);
		pressureParam = Float(0.0f);
		targetPressureParam = Float(0.5f);
		curvatureParam = Float(0.3f);
	}
	MultiActiveContour2D::MultiActiveContour2D(const std::string& name, const std::shared_ptr<SpringlCache2D>& cache) : Simulation(name),
		cache(cache), preserveTopology(false), clampSpeed(false), updateIsoSurface(
			false) {
		advectionParam = Float(1.0f);
		pressureParam = Float(0.0f);
		targetPressureParam = Float(0.5f);
		curvatureParam = Float(0.3f);
	}


	void MultiActiveContour2D::setup(const aly::ParameterPanePtr& pane) {
		pane->addNumberField("Target Pressure", targetPressureParam, Float(0.0f), Float(1.0f));
		pane->addNumberField("Pressure Weight", pressureParam, Float(-2.0f), Float(2.0f));
		pane->addNumberField("Advection Weight", advectionParam, Float(-1.0f), Float(1.0f));
		pane->addNumberField("Curvature Weight", curvatureParam, Float(0.0f), Float(4.0f));
		pane->addCheckBox("Preserve Topology", preserveTopology);
		pane->addCheckBox("Clamp Speed", clampSpeed);
	}
	void MultiActiveContour2D::cleanup() {
		if (cache.get() != nullptr)cache->clear();
	}
	void MultiActiveContour2D::setInitial(const Image1i& labels) {
		this->initialLabels = labels;
		this->swapLabelImage = labels;
		this->labelImage = labels;
		levelSet.resize(labels.width, labels.height);
		swapLevelSet.resize(labels.width, labels.height);
#pragma omp parallel for
		for (int j = 0;j < labels.height;j++) {
			int activeLabels[4];
			for (int i = 0;i < labels.width;i++) {
				int currentLabel = labels(i, j).x;
				activeLabels[0] = labels(i + 1, j).x;
				activeLabels[1] = labels(i - 1, j).x;
				activeLabels[2] = labels(i, j + 1).x;
				activeLabels[3] = labels(i, j - 1).x;
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
		for (int band = 1; band <= 2*maxLayers; band++) {
#pragma omp parallel for
			for (int j = 0;j < labels.height;j++) {
				for (int i = 0;i < labels.width;i++) {
					updateDistanceField(i,j, band);
				}
			}
		}
		initialLevelSet = levelSet;
		WriteImageToRawFile(GetDesktopDirectory() + ALY_PATH_SEPARATOR + "init_labels.xml",labelImage);
		WriteImageToRawFile(GetDesktopDirectory() + ALY_PATH_SEPARATOR + "init_distfield.xml", levelSet);

	}

	bool MultiActiveContour2D::init() {
		int2 dims = initialLevelSet.dimensions();
		if (dims.x == 0 || dims.y == 0)return false;
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
		forceIndexes.resize(L,-1);
		labelList.clear();
		labelList.assign(labelSet.begin(), labelSet.end());
		for (int i = 0;i < (int)labelList.size();i++) {
			forceIndexes[labelList[i]] = i;
		}
		lineColors.clear();
		L = (int)labelList.size();
		lineColors[0] = RGBAf(0.0f, 0.0f, 0.0f, 0.0f);
		if(L<256){
			int CL = std::min(256, L);
			for (int i = 0;i < L;i++) {
				int l = labelList[i];
				HSV hsv = HSV((l%CL) / (float)CL, 0.7f, 0.7f);
				lineColors[l] = HSVtoColor(hsv);
			}
		}
		else {
			for (int i = 0;i < L;i++) {
				int l = labelList[i];
				HSV hsv = HSV(RandomUniform(0.0f, 1.0f), RandomUniform(0.5f, 1.0f), RandomUniform(0.5f, 1.0f));
				lineColors[l] = HSVtoColor(hsv);
			}
		}
		rebuildNarrowBand();
		updateIsoSurface = (getNumLabels()<256);
		updateOverlay = true;
		if (cache.get() != nullptr) {
			Contour2D* contour = getContour();
			contour->setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
			cache->set((int)mSimulationIteration, *contour);
		}
		return true;
	}
	void MultiActiveContour2D::pressureAndAdvectionMotion(int i, int j, size_t gid) {
		float v11 = swapLevelSet(i, j).x;
		if (v11 > 0.5f) {
			for (int index = 0;index < 5;index++) {
				deltaLevelSet[5 * gid + index] = 0;
			}
			return;
		}
		int activeLabels[5];
		activeLabels[0] = swapLabelImage(i, j);
		activeLabels[1] = swapLabelImage(i + 1, j);
		activeLabels[2] = swapLabelImage(i - 1, j);
		activeLabels[3] = swapLabelImage(i, j + 1);
		activeLabels[4] = swapLabelImage(i, j - 1);
		int label;
		float2 vec = vecFieldImage(i, j);
		float forceX = advectionParam.toFloat() * vec.x;
		float forceY = advectionParam.toFloat() * vec.y;
		float pressureValue = pressureImage(i, j).x;
		for (int index = 0;index < 5;index++) {
			label = activeLabels[index];
			if (label == 0) {
				objectIds[5 * gid + index] = 0;
				deltaLevelSet[5 * gid + index] = 0;
			}
			else {
				objectIds[5 * gid + index] = label;
				if (forceIndexes[label] < 0) {
					deltaLevelSet[5 * gid + index] = 0.999f;
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

					float DxNegMin = min(DxNeg, 0.0f);
					float DxNegMax = max(DxNeg, 0.0f);
					float DxPosMin = min(DxPos, 0.0f);
					float DxPosMax = max(DxPos, 0.0f);
					float DyNegMin = min(DyNeg, 0.0f);
					float DyNegMax = max(DyNeg, 0.0f);
					float DyPosMin = min(DyPos, 0.0f);
					float DyPosMax = max(DyPos, 0.0f);
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
						kappa = curvatureParam.toFloat()* numer / denom;
					}
					else {
						kappa = curvatureParam.toFloat()* numer * sign(denom) * 1E5f;
					}
					if (kappa < -maxCurvatureForce) {
						kappa = -maxCurvatureForce;
					}
					else if (kappa > maxCurvatureForce) {
						kappa = maxCurvatureForce;
					}

					// Level set force should be the opposite sign of advection force so it
					// moves in the direction of the force.



					float advection = 0;

					// Dot product force with upwind gradient
					if (forceX > 0) {
						advection = forceX * DxNeg;
					}
					else if (forceX < 0) {
						advection = forceX * DxPos;
					}
					if (forceY > 0) {
						advection += forceY * DyNeg;
					}
					else if (forceY < 0) {
						advection += forceY * DyPos;
					}

					// Force should be negative to move level set outwards if pressure is
					// positive
					float force = pressureParam.toFloat() * pressureValue;
					float pressure = 0;
					if (force > 0) {
						pressure = -force * std::sqrt(GradientSqrPos);
					}
					else if (force < 0) {
						pressure = -force * std::sqrt(GradientSqrNeg);
					}

					deltaLevelSet[5 * gid + index] = -advection + kappa + pressure;
				}
			}
		}
	}
	void MultiActiveContour2D::advectionMotion(int i, int j, size_t gid) {
		float v11 = swapLevelSet(i, j).x;
		if (v11 > 0.5f) {
			for (int index = 0;index < 5;index++) {
				deltaLevelSet[5 * gid + index] = 0;
			}
			return;
		}
		int activeLabels[5];

		activeLabels[0] = swapLabelImage(i, j);
		activeLabels[1] = swapLabelImage(i + 1, j);
		activeLabels[2] = swapLabelImage(i - 1, j);
		activeLabels[3] = swapLabelImage(i, j + 1);
		activeLabels[4] = swapLabelImage(i, j - 1);
		int label;

		float2 vec = vecFieldImage(i, j);
		float forceX = advectionParam.toFloat() * vec.x;
		float forceY = advectionParam.toFloat() * vec.y;

		for (int index = 0;index < 5;index++) {
			label = activeLabels[index];
			if (label == 0) {
				objectIds[5 * gid + index] = 0;
				deltaLevelSet[5 * gid + index] = 0;
			}
			else {
				objectIds[5 * gid + index] = label;
				if (forceIndexes[label] < 0) {
					deltaLevelSet[5 * gid + index] = 0.999f;
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
					if (std::abs(denom) > 1E-3f) {
						kappa = curvatureParam.toFloat() * numer / denom;
					}
					else {
						kappa = curvatureParam.toFloat() * numer * sign(denom) * 1E3f;
					}
					if (kappa < -maxCurvatureForce) {
						kappa = -maxCurvatureForce;
					}
					else if (kappa > maxCurvatureForce) {
						kappa = maxCurvatureForce;
					}

					// Level set force should be the opposite sign of advection force so it
					// moves in the direction of the force.

					float advection = 0;
					// Dot product force with upwind gradient
					if (forceX > 0) {
						advection = forceX * DxNeg;
					}
					else if (forceX < 0) {
						advection = forceX * DxPos;
					}
					if (forceY > 0) {
						advection += forceY * DyNeg;
					}
					else if (forceY < 0) {
						advection += forceY * DyPos;
					}
					deltaLevelSet[5 * gid + index] = -advection + kappa;
				}
			}
		}
	}
	void MultiActiveContour2D::applyForces(int i, int j, size_t gid, float timeStep) {
		if (swapLevelSet(i, j).x>0.5f)return;
		float minValue1 = 1E10f;
		float minValue2 = 1E10f;
		int minLabel1 = -1;
		int minLabel2 = -1;
		int mask = 0;
		float delta;
		float update = 0;
		for (int l = 0;l < 5;l++) {
			mask = objectIds[5 * gid + l];
			if (clampSpeed) {
				delta = timeStep*clamp(deltaLevelSet[5 * gid + l], -1.0f, 1.0f);
			}
			else {
				delta = timeStep*deltaLevelSet[5 * gid + l];
			}
			if (mask != -1) {
				update = getLevelSetValue(i, j, mask) + delta;
				if (mask != minLabel1&&mask != minLabel2) {
					if (update < minValue1) {
						minValue2 = minValue1;
						minLabel2 = minLabel1;
						minValue1 = update;
						minLabel1 = mask;
					}
					else if (update < minValue2) {
						minValue2 = update;
						minLabel2 = mask;
					}
				}
			}
		}
		if (minLabel2 >= 0) {
			if (minValue1 == minValue2) {
				labelImage(i, j).x = min(minLabel1, minLabel2);
			}
			else {
				labelImage(i, j).x = minLabel1;
			}
			levelSet(i, j).x = std::abs(0.5f * (float)(minValue1 - minValue2));
		}
		else if (minValue1 < 1E10f) {
			labelImage(i, j) = minLabel1;
			levelSet(i, j) = std::abs(minValue1);
		}
	}
	bool MultiActiveContour2D::getBitValue(int i) {
		const char lut4_8[] = { 123, -13, -5, -13, -69, 51, -69, 51, -128, -13, -128, -13, 0, 51, 0, 51, -128, -13, -128, -13, -69, -52, -69, -52, -128, -13,
				-128, -13, -69, -52, -69, -52, -128, 0, -128, 0, -69, 51, -69, 51, 0, 0, 0, 0, 0, 51, 0, 51, -128, -13, -128, -13, -69, -52, -69, -52, -128,
				-13, -128, -13, -69, -52, -69, -52, 123, -13, -5, -13, -69, 51, -69, 51, -128, -13, -128, -13, 0, 51, 0, 51, -128, -13, -128, -13, -69, -52,
				-69, -52, -128, -13, -128, -13, -69, -52, -69, -52, -128, 0, -128, 0, -69, 51, -69, 51, 0, 0, 0, 0, 0, 51, 0, 51, -128, -13, -128, -13, -69,
				-52, -69, -52, -128, -13, -128, -13, -69, -52, -69, -52 };
		return ((((uint8_t)lut4_8[63 - (i >> 3)]) & (1 << (i % 8))) > 0);
	}

	int MultiActiveContour2D::deleteElements() {
		std::vector<int2> newList;
		for (int i = 0; i < activeList.size(); i++) {
			int2 pos = activeList[i];
			float val = swapLevelSet(pos.x, pos.y);
			if (std::abs(val) <= MAX_DISTANCE) {
				newList.push_back(pos);
			}
			else {
				val = sign(val) * (MAX_DISTANCE + 0.5f);
				levelSet(pos.x, pos.y) = val;
				swapLevelSet(pos.x, pos.y) = val;
			}
		}
		int diff = (int)(activeList.size() - newList.size());
		activeList = newList;
		return diff;
	}
	int MultiActiveContour2D::addElements() {
		const int xShift[4] = { -1, 1, 0, 0 };
		const int yShift[4] = { 0, 0,-1, 1 };
		std::vector<int2> newList;
		int sz = (int)activeList.size();
		float INDICATOR = (float)std::max(levelSet.width, levelSet.height);
		for (int offset = 0; offset < 4; offset++) {
			int xOff = xShift[offset];
			int yOff = yShift[offset];
			for (int n = 0; n < sz; n++) {
				int2 pos = activeList[n];
				int2 pos2 = int2(pos.x + xOff, pos.y + yOff);
				float val1 = std::abs(levelSet(pos.x, pos.y));
				float val2 = std::abs(levelSet(pos2.x, pos2.y));
				if (val1 <= MAX_DISTANCE - 1.0f &&
					val2 >= MAX_DISTANCE&&
					val2 < INDICATOR) {
					levelSet(pos2.x, pos2.y) = INDICATOR + offset;
				}
			}
		}
		for (int offset = 0; offset < 4; offset++) {
			int xOff = xShift[offset];
			int yOff = yShift[offset];
			for (int n = 0; n < sz; n++) {
				int2 pos = activeList[n];
				int2 pos2 = int2(pos.x + xOff, pos.y + yOff);
				float val1 = levelSet(pos.x, pos.y);
				float val2 = levelSet(pos2.x, pos2.y);
				if (std::abs(val1) <= MAX_DISTANCE - 1.0f && val2 == INDICATOR + offset) {
					activeList.push_back(pos2);
					val2 = swapLevelSet(pos2.x, pos2.y);
					val2 = sign(val2) * MAX_DISTANCE;
					swapLevelSet(pos2.x, pos2.y) = val2;
					levelSet(pos2.x, pos2.y) = val2;
				}
			}
		}
		return (int)(activeList.size() - sz);
	}
	void MultiActiveContour2D::applyForcesTopoRule(int i, int j, int offset, size_t gid, float timeStep) {
		const int xShift[4] = { 0, 0, 1, 1 };
		const int yShift[4] = { 0, 1, 0, 1 };
		int xOff = xShift[offset];
		int yOff = yShift[offset];
		if (i % 2 != xOff || j % 2 != yOff)return;
		if (swapLevelSet(i, j).x>0.5f)return;
		float minValue1 = 1E10f;
		float minValue2 = 1E10f;
		int minLabel1 = -1;
		int minLabel2 = -1;
		int mask = 0;
		int oldLabel = labelImage(i,j);
		float delta;
		float update = 0;
		for (int l = 0;l < 5;l++) {
			mask = objectIds[5 * gid+l];
			if (clampSpeed) {
				delta = timeStep*clamp(deltaLevelSet[5 * gid+l], -1.0f, 1.0f);
			} else {
				delta = timeStep*deltaLevelSet[5 * gid+l];
			}
			if (mask != -1) {
				update = getLevelSetValue(i, j,mask) + delta;
				if (mask != minLabel1&&mask != minLabel2) {
					if (update < minValue1) {
						minValue2 = minValue1;
						minLabel2 = minLabel1;
						minValue1 = update;
						minLabel1 = mask;
					}
					else if (update < minValue2) {
						minValue2 = update;
						minLabel2 = mask;
					}
				}
			}
		}
		mask = 0;
		mask |= ((labelImage( i - 1, j - 1).x== oldLabel) ? (1 << 0) : 0);
		mask |= ((labelImage( i - 1, j + 0).x== oldLabel) ? (1 << 1) : 0);
		mask |= ((labelImage( i - 1, j + 1).x== oldLabel) ? (1 << 2) : 0);
		mask |= ((labelImage( i + 0, j - 1).x== oldLabel) ? (1 << 3) : 0);
		mask |= ((labelImage( i + 0, j + 0).x== oldLabel) ? (1 << 4) : 0);
		mask |= ((labelImage( i + 0, j + 1).x== oldLabel) ? (1 << 5) : 0);
		mask |= ((labelImage( i + 1, j - 1).x== oldLabel) ? (1 << 6) : 0);
		mask |= ((labelImage( i + 1, j + 0).x== oldLabel) ? (1 << 7) : 0);
		mask |= ((labelImage( i + 1, j + 1).x== oldLabel) ? (1 << 8) : 0);

		if (!getBitValue(mask)) {
			levelSet(i,j) = 1.0f;
			return;
		}

		if (minLabel2 >= 0) {
			if (minValue1 == minValue2) {
				labelImage(i,j).x = std::min(minLabel1, minLabel2);
			}
			else {
				labelImage(i, j).x = minLabel1;
			}
			levelSet(i,j).x = std::abs(0.5f * (float)(minValue1 - minValue2));
		}
		else if (minValue1<1E10f) {
			labelImage(i,j).x = minLabel1;
			levelSet(i, j).x = std::abs(minValue1);
		}
		
	}
	void MultiActiveContour2D::pressureMotion(int i, int j, size_t gid) {
		float v11 = swapLevelSet(i, j).x;
		if (v11 > 0.5f) {
			for (int index = 0;index < 5;index++) {
				deltaLevelSet[5 * gid + index] = 0;
			}
			return;
		}
		float4 grad;
		int activeLabels[5];
		activeLabels[0] = swapLabelImage(i, j);
		activeLabels[1] = swapLabelImage(i + 1, j);
		activeLabels[2] = swapLabelImage(i - 1, j);
		activeLabels[3] = swapLabelImage(i, j + 1);
		activeLabels[4] = swapLabelImage(i, j - 1);
		int label;
		float pressureValue = pressureImage(i, j).x;
		for (int index = 0;index < 5;index++) {
			label = activeLabels[index];
			if (label == 0) {
				objectIds[gid * 5 + index] = 0;
				deltaLevelSet[gid * 5 + index] = 0;
			}
			else {
				objectIds[gid * 5 + index] = label;
				if (forceIndexes[label] < 0) {
					deltaLevelSet[gid * 5 + index] = 0.999f;
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

					float DxNegMin = min(DxNeg, 0.0f);
					float DxNegMax = max(DxNeg, 0.0f);
					float DxPosMin = min(DxPos, 0.0f);
					float DxPosMax = max(DxPos, 0.0f);
					float DyNegMin = min(DyNeg, 0.0f);
					float DyNegMax = max(DyNeg, 0.0f);
					float DyPosMin = min(DyPos, 0.0f);
					float DyPosMax = max(DyPos, 0.0f);
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
					float force = pressureParam.toFloat() *pressureValue;
					float pressure = 0;
					if (force > 0) {
						pressure = -force * std::sqrt(GradientSqrPos);
					}
					else if (force < 0) {
						pressure = -force * std::sqrt(GradientSqrNeg);
					}
					deltaLevelSet[gid * 5 + index] = kappa + pressure;
				}
			}
		}
	}
	float MultiActiveContour2D::getLevelSetValue(int i, int j, int l) const {
		if (labelImage(i, j).x == l) {
			return -levelSet(i, j);
		}
		else {
			return levelSet(i, j);
		}
	}
	float MultiActiveContour2D::getSwapLevelSetValue(int i, int j, int l) const {
		if (swapLabelImage(i, j).x == l) {
			return -swapLevelSet(i, j);
		}
		else {
			return swapLevelSet(i, j);
		}
	}
	void MultiActiveContour2D::updateDistanceField(int i, int j, int band) {
		float v11;
		float v01;
		float v12;
		float v10;
		float v21;
		float activeLevelSet = swapLevelSet(i, j).x;
		if (std::abs(activeLevelSet) <= 0.5f) {
			return;
		}
		int label = labelImage(i, j);
		v01 = getLevelSetValue(i - 1, j, label);
		v12 = getLevelSetValue(i, j + 1, label);
		v10 = getLevelSetValue(i, j - 1, label);
		v21 = getLevelSetValue(i + 1, j, label);
		if (levelSet(i, j) > band - 0.5f) {
			v11 = 1E10f;
			v11 = std::min(std::abs(v01 - 1), v11);
			v11 = std::min(std::abs(v12 - 1), v11);
			v11 = std::min(std::abs(v10 - 1), v11);
			v11 = std::min(std::abs(v21 - 1), v11);
			levelSet(i, j) = v11;
		}
	}
	float MultiActiveContour2D::evolve(float maxStep) {
		if (pressureImage.size() > 0) {
			if (vecFieldImage.size() > 0) {
#pragma omp parallel for
				for (int i = 0; i < (int)activeList.size(); i++) {
					int2 pos = activeList[i];
					pressureAndAdvectionMotion(pos.x, pos.y, i);
				}
			}
			else {
#pragma omp parallel for
				for (int i = 0; i < (int)activeList.size(); i++) {
					int2 pos = activeList[i];
					pressureMotion(pos.x, pos.y, i);
				}
			}
		}
		else if (vecFieldImage.size() > 0) {
#pragma omp parallel for
			for (int i = 0; i < (int)activeList.size(); i++) {
				int2 pos = activeList[i];
				advectionMotion(pos.x, pos.y, i);
			}
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
		updateIsoSurface = (getNumLabels()<256);
		updateOverlay = true;
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
	bool MultiActiveContour2D::stepInternal() {
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
			Contour2D* contour = getContour();
			contour->setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
			cache->set((int)mSimulationIteration, *contour);
		}
		return (mSimulationTime<mSimulationDuration);
	}
	void MultiActiveContour2D::rescale(aly::Image1f& pressureForce) {
		float minValue = 1E30f;
		float maxValue = -1E30f;
		if (!std::isnan(targetPressureParam.toFloat())) {
			for (int i = 0; i < pressureForce.size(); i++) {
				float val = pressureForce[i] - targetPressureParam.toFloat();
				minValue = std::min(val, minValue);
				maxValue = std::max(val, maxValue);
			}
		}
		float normMin = (std::abs(minValue) > 1E-4) ? 1 / std::abs(minValue) : 1;
		float normMax = (std::abs(maxValue) > 1E-4) ? 1 / std::abs(maxValue) : 1;
#pragma omp parallel for
		for (int i = 0; i < pressureForce.size(); i++) {
			float val = pressureForce[i] - targetPressureParam.toFloat();
			if (val < 0) {
				pressureForce[i] = (float)(val * normMin);
			}
			else {
				pressureForce[i] = (float)(val * normMax);
			}
		}
	}
}

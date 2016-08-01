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
#include "ActiveContour2D.h"

namespace aly {

	void ActiveContour2D::rebuildNarrowBand() {
		activeList.clear();
		for (int band = 1; band <= maxLayers; band++) {
#pragma omp parallel for
			for (int i = 0; i < (int) activeList.size(); i++) {
				int2 pos = activeList[i];
				updateDistanceField(pos.x, pos.y, band, i);
			}
		}
		for (int j = 0; j < swapLevelSet.height; j++) {
			for (int i = 0; i < swapLevelSet.width; i++) {
				if (std::abs(swapLevelSet(i, j)) <= MAX_DISTANCE) {
					activeList.push_back(int2(i, j));
				}
			}
		}
		deltaLevelSet.clear();
		deltaLevelSet.resize(activeList.size(), 0.0f);
	}
	void ActiveContour2D::plugLevelSet(int i, int j, size_t index) {
		float v11;
		float v01;
		float v12;
		float v10;
		float v21;
		v11 = levelSet(i, j);
		float sgn = sign(v11);
		v11 = sgn * v11;
		v01 = sgn * levelSet(i - 1, j);
		v12 = sgn * levelSet(i, j + 1);
		v10 = sgn * levelSet(i, j - 1);
		v21 = sgn * levelSet(i + 1, j);
		if (v11 > 0 && v11 < 0.5f && v01 > 0 && v12 > 0 && v10 > 0 && v21 > 0) {
			levelSet(i, j) = sgn * MAX_DISTANCE;
		}
	}
	const Contour2D& ActiveContour2D::getContour() {
		if (updateIsoSurface) {
			std::lock_guard<std::mutex> lockMe(contourLock);
			isoContour.solve(levelSet, contour.vertexes, contour.indexes, 0.0f, (preserveTopology) ? TopologyRule2D::Connect4 : TopologyRule2D::Unconstrained);
			updateIsoSurface = false;
		}
		return contour;
	}
	ActiveContour2D::ActiveContour2D(const std::shared_ptr<SpringlCache2D>& cache) :Simulation("Active Contour 2D"),
			cache(cache),advectionWeight(1.0f), pressureWeight(1.0), curvatureWeight(0.1f), targetPressure(NAN), maxLayers(3), preserveTopology(false), clampSpeed(false), updateIsoSurface(
					false), resamplingInterval(5){

	}
	void ActiveContour2D::setup(const aly::ParameterPanePtr& pane){
	
	}
	void ActiveContour2D::cleanup(){
		cache->clear();
	}
	bool ActiveContour2D::init() {
		int2 dims = initialLevelSet.dimensions();
		mSimulationDuration=std::max(dims.x,dims.y);
		mSimulationIteration=0;
		mSimulationTime=0;
		mTimeStep=0.5;
		if(dims.x==0||dims.y==0)return false;
		levelSet.resize(dims.x, dims.y);
		swapLevelSet.resize(dims.x, dims.y);
#pragma omp parallel for
		for (int i = 0; i < initialLevelSet.size(); i++) {
			float val = clamp(initialLevelSet[i], -(maxLayers + 1.0f), (maxLayers + 1.0f));
			levelSet[i] = val;
			swapLevelSet[i] = val;
		}
		if (pressureImage.size() != 0) {
			rescale(pressureImage);
		}
		rebuildNarrowBand();
		updateIsoSurface = true;
		if (cache.get() != nullptr) {
			Contour2D contour = getContour();
			//WriteContourToFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin", contour);
			contour.setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
			cache->set((int)mSimulationIteration, contour);
			
		}
		return true;
	}
	void ActiveContour2D::pressureAndAdvectionMotion(int i, int j, size_t gid) {
		float v11 = swapLevelSet(i, j).x;
		float2 grad;
		if (v11 > 0.5f || v11 < -0.5f) {
			deltaLevelSet[gid] = 0;
			return;
		}

		float v00 = swapLevelSet(i - 1, j - 1).x;
		float v01 = swapLevelSet(i - 1, j).x;
		float v10 = swapLevelSet(i, j - 1).x;
		float v21 = swapLevelSet(i + 1, j).x;
		float v20 = swapLevelSet(i + 1, j - 1).x;
		float v22 = swapLevelSet(i + 1, j + 1).x;
		float v02 = swapLevelSet(i - 1, j + 1).x;
		float v12 = swapLevelSet(i, j + 1).x;

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

		float numer = 0.5f * (DyCtr * DyCtr * DxxCtr - 2 * DxCtr * DyCtr * DxyCtr + DxCtr * DxCtr * DyyCtr);
		float denom = DxCtr * DxCtr + DyCtr * DyCtr;
		float kappa = 0;

		const float maxCurvatureForce = 10.0f;
		if (fabs(denom) > 1E-5f) {
			kappa = curvatureWeight * numer / denom;
		} else {
			kappa = curvatureWeight * numer * sign(denom) * 1E5f;
		}
		if (kappa < -maxCurvatureForce) {
			kappa = -maxCurvatureForce;
		} else if (kappa > maxCurvatureForce) {
			kappa = maxCurvatureForce;
		}
		float force = pressureWeight * pressureImage(i, j).x;
		float pressure = 0;
		if (force > 0) {
			pressure = -force * std::sqrt(GradientSqrPos);
		} else if (force < 0) {
			pressure = -force * std::sqrt(GradientSqrNeg);
		}

		// Level set force should be the opposite sign of advection force so it
		// moves in the direction of the force.

		float2 vec = vecFieldImage(i, j);
		float forceX = advectionWeight * vec.x;
		float forceY = advectionWeight * vec.y;
		float advection = 0;

		// Dot product force with upwind gradient
		if (forceX > 0) {
			advection = forceX * DxNeg;
		} else if (forceX < 0) {
			advection = forceX * DxPos;
		}
		if (forceY > 0) {
			advection += forceY * DyNeg;
		} else if (forceY < 0) {
			advection += forceY * DyPos;
		}
		deltaLevelSet[gid] = -advection + kappa + pressure;
	}
	void ActiveContour2D::advectionMotion(int i, int j, size_t gid) {
		float v11 = swapLevelSet(i, j).x;
		float2 grad;
		if (v11 > 0.5f || v11 < -0.5f) {
			deltaLevelSet[gid] = 0;
			return;
		}

		float v00 = swapLevelSet(i - 1, j - 1).x;
		float v01 = swapLevelSet(i - 1, j).x;
		float v10 = swapLevelSet(i, j - 1).x;
		float v21 = swapLevelSet(i + 1, j).x;
		float v20 = swapLevelSet(i + 1, j - 1).x;
		float v22 = swapLevelSet(i + 1, j + 1).x;
		float v02 = swapLevelSet(i - 1, j + 1).x;
		float v12 = swapLevelSet(i, j + 1).x;

		float DxNeg = v11 - v01;
		float DxPos = v21 - v11;
		float DyNeg = v11 - v10;
		float DyPos = v12 - v11;

		float DxCtr = 0.5f * (v21 - v01);
		float DyCtr = 0.5f * (v12 - v10);

		float DxxCtr = v21 - v11 - v11 + v01;
		float DyyCtr = v12 - v11 - v11 + v10;
		float DxyCtr = (v22 - v02 - v20 + v00) * 0.25f;

		float numer = 0.5f * (DyCtr * DyCtr * DxxCtr - 2 * DxCtr * DyCtr * DxyCtr + DxCtr * DxCtr * DyyCtr);
		float denom = DxCtr * DxCtr + DyCtr * DyCtr;
		float kappa = 0;

		const float maxCurvatureForce = 10.0f;
		if (fabs(denom) > 1E-5f) {
			kappa = curvatureWeight * numer / denom;
		} else {
			kappa = curvatureWeight * numer * sign(denom) * 1E5f;
		}
		if (kappa < -maxCurvatureForce) {
			kappa = -maxCurvatureForce;
		} else if (kappa > maxCurvatureForce) {
			kappa = maxCurvatureForce;
		}

		// Level set force should be the opposite sign of advection force so it
		// moves in the direction of the force.

		float2 vec = vecFieldImage(i, j);
		float forceX = advectionWeight * vec.x;
		float forceY = advectionWeight * vec.y;
		float advection = 0;

		// Dot product force with upwind gradient
		if (forceX > 0) {
			advection = forceX * DxNeg;
		} else if (forceX < 0) {
			advection = forceX * DxPos;
		}
		if (forceY > 0) {
			advection += forceY * DyNeg;
		} else if (forceY < 0) {
			advection += forceY * DyPos;
		}

		deltaLevelSet[gid] = -advection + kappa;
	}
	void ActiveContour2D::applyForces(int i, int j, size_t index, float timeStep) {
		float delta;
		float old = swapLevelSet(i, j);
		if (std::abs(old) > 0.5f)
			return;
		if (clampSpeed) {
			delta = timeStep * clamp(deltaLevelSet[index], -1.0f, 1.0f);
		} else {
			delta = timeStep * deltaLevelSet[index];
		}
		old += delta;
		levelSet(i, j) = float1(old);
	}
	bool ActiveContour2D::getBitValue(int i) {
		const char lut4_8[] = { 123, -13, -5, -13, -69, 51, -69, 51, -128, -13, -128, -13, 0, 51, 0, 51, -128, -13, -128, -13, -69, -52, -69, -52, -128, -13,
				-128, -13, -69, -52, -69, -52, -128, 0, -128, 0, -69, 51, -69, 51, 0, 0, 0, 0, 0, 51, 0, 51, -128, -13, -128, -13, -69, -52, -69, -52, -128,
				-13, -128, -13, -69, -52, -69, -52, 123, -13, -5, -13, -69, 51, -69, 51, -128, -13, -128, -13, 0, 51, 0, 51, -128, -13, -128, -13, -69, -52,
				-69, -52, -128, -13, -128, -13, -69, -52, -69, -52, -128, 0, -128, 0, -69, 51, -69, 51, 0, 0, 0, 0, 0, 51, 0, 51, -128, -13, -128, -13, -69,
				-52, -69, -52, -128, -13, -128, -13, -69, -52, -69, -52 };
		return ((((uint8_t)lut4_8[63 - (i >> 3)]) & (1 << (i % 8))) > 0);
	}
	int ActiveContour2D::deleteElements() {
		std::vector<int2> newList;
		for (int i = 0; i < activeList.size(); i++) {
			int2 pos = activeList[i];
			float val = swapLevelSet(pos.x, pos.y);
			if (std::abs(val) <= MAX_DISTANCE) {
				newList.push_back(pos);
			} else {
				val = sign(val) * (MAX_DISTANCE + 0.5f);
				levelSet(pos.x, pos.y) = val;
				swapLevelSet(pos.x, pos.y) = val;
			}
		}
		int diff = (int)(activeList.size() - newList.size());
		activeList = newList;
		return diff;
	}
	int ActiveContour2D::addElements() {
		const int xShift[4] =  {-1, 1, 0, 0};
		const int yShift[4] =  { 0, 0,-1, 1};
		std::vector<int2> newList;
		int sz = (int)activeList.size();
		float INDICATOR=(float)std::max(levelSet.width,levelSet.height);
		for (int offset = 0; offset < 4; offset++) {
			int xOff = xShift[offset];
			int yOff = yShift[offset];
			for (int n = 0; n < sz; n++) {
				int2 pos = activeList[n];
				int2 pos2 = int2(pos.x + xOff, pos.y + yOff);
				float val1 = std::abs(levelSet(pos.x, pos.y));
				float val2 = std::abs(levelSet(pos2.x, pos2.y));
				if (	val1 <= MAX_DISTANCE - 1.0f &&
						val2 >= MAX_DISTANCE&&
						val2 < INDICATOR) {
					levelSet(pos2.x, pos2.y) =INDICATOR + offset;
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
				if (std::abs(val1) <= MAX_DISTANCE - 1.0f && val2==INDICATOR+offset) {
					activeList.push_back(pos2);
					val2 = swapLevelSet(pos2.x, pos2.y);
					val2= sign(val2) * MAX_DISTANCE;
					swapLevelSet(pos2.x, pos2.y) = val2;
					levelSet(pos2.x, pos2.y) = val2;
				}
			}
		}
		return (int)(activeList.size() - sz);
	}
	void ActiveContour2D::applyForcesTopoRule(int i, int j, int offset, size_t index, float timeStep) {
		float v11 = swapLevelSet(i, j).x;
		if (std::abs(v11) > 0.5f) {
			levelSet(i, j) = v11;
			return;
		}
		float delta;
		if (clampSpeed) {
			delta = timeStep * aly::clamp(deltaLevelSet[index], -1.0f, 1.0f);
		} else {
			delta = timeStep * deltaLevelSet[index];
		}
		const int xShift[4] = { 0, 0, 1, 1 };
		const int yShift[4] = { 0, 1, 0, 1 };
		int xOff = xShift[offset];
		int yOff = yShift[offset];
		if (i % 2 != xOff || j % 2 != yOff)
			return;
		float oldValue = swapLevelSet(i, j);
		float newValue = oldValue + delta;
		int mask = 0;
		if (newValue * oldValue <= 0) {
			mask |= ((levelSet(i - 1, j - 1).x < 0) ? (1 << 0) : 0);
			mask |= ((levelSet(i - 1, j + 0).x < 0) ? (1 << 1) : 0);
			mask |= ((levelSet(i - 1, j + 1).x < 0) ? (1 << 2) : 0);
			mask |= ((levelSet(i + 0, j - 1).x < 0) ? (1 << 3) : 0);
			mask |= ((levelSet(i + 0, j + 0).x < 0) ? (1 << 4) : 0);
			mask |= ((levelSet(i + 0, j + 1).x < 0) ? (1 << 5) : 0);
			mask |= ((levelSet(i + 1, j - 1).x < 0) ? (1 << 6) : 0);
			mask |= ((levelSet(i + 1, j + 0).x < 0) ? (1 << 7) : 0);
			mask |= ((levelSet(i + 1, j + 1).x < 0) ? (1 << 8) : 0);
			if (!getBitValue(mask)) {
				newValue = sign(oldValue);
			}
		}
		levelSet(i, j) = newValue;
	}
	void ActiveContour2D::pressureMotion(int i, int j, size_t gid) {
		float v11 = swapLevelSet(i, j).x;
		float2 grad;
		if (std::abs(v11) > 0.5f) {
			deltaLevelSet[gid] = 0;
			return;
		}
		float v00 = swapLevelSet(i - 1, j - 1).x;
		float v01 = swapLevelSet(i - 1, j).x;
		float v10 = swapLevelSet(i, j - 1).x;
		float v21 = swapLevelSet(i + 1, j).x;
		float v20 = swapLevelSet(i + 1, j - 1).x;
		float v22 = swapLevelSet(i + 1, j + 1).x;
		float v02 = swapLevelSet(i - 1, j + 1).x;
		float v12 = swapLevelSet(i, j + 1).x;
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

		float DxCtr = 0.5f * (v21 - v01);
		float DyCtr = 0.5f * (v12 - v10);

		float DxxCtr = v21 - v11 - v11 + v01;
		float DyyCtr = v12 - v11 - v11 + v10;
		float DxyCtr = (v22 - v02 - v20 + v00) * 0.25f;

		float numer = 0.5f * (DyCtr * DyCtr * DxxCtr - 2 * DxCtr * DyCtr * DxyCtr + DxCtr * DxCtr * DyyCtr);
		float denom = DxCtr * DxCtr + DyCtr * DyCtr;
		float kappa = 0;
		const float maxCurvatureForce = 10.0f;
		if (std::abs(denom) > 1E-5f) {
			kappa = curvatureWeight * numer / denom;
		} else {
			kappa = curvatureWeight * numer * sign(denom) * 1E5f;
		}
		if (kappa < -maxCurvatureForce) {
			kappa = -maxCurvatureForce;
		} else if (kappa > maxCurvatureForce) {
			kappa = maxCurvatureForce;
		}
		float force = pressureWeight * pressureImage(i, j).x;
		float pressure = 0;
		if (force > 0) {
			float GradientSqrPos = DxNegMax * DxNegMax + DxPosMin * DxPosMin + DyNegMax * DyNegMax + DyPosMin * DyPosMin;
			pressure = -force * std::sqrt(GradientSqrPos);
		} else if (force < 0) {
			float GradientSqrNeg = DxPosMax * DxPosMax + DxNegMin * DxNegMin + DyPosMax * DyPosMax + DyNegMin * DyNegMin;
			pressure = -force * std::sqrt(GradientSqrNeg);
		}
		deltaLevelSet[gid] = kappa + pressure;
	}
	void ActiveContour2D::updateDistanceField(int i, int j, int band, size_t index) {
		float v11;
		float v01;
		float v12;
		float v10;
		float v21;
		float activeLevelSet = swapLevelSet(i, j).x;
		if (std::abs(activeLevelSet) <= 0.5f) {
			return;
		}
		v11 = levelSet(i, j);
		float oldVal = v11;
		v01 = levelSet(i - 1, j);
		v12 = levelSet(i, j + 1);
		v10 = levelSet(i, j - 1);
		v21 = levelSet(i + 1, j);
		if (v11 < -band + 0.5f) {
			v11 = -1E30f;
			v11 = (v01 > 1) ? v11 : max(v01, v11);
			v11 = (v12 > 1) ? v11 : max(v12, v11);
			v11 = (v10 > 1) ? v11 : max(v10, v11);
			v11 = (v21 > 1) ? v11 : max(v21, v11);
			v11 -= 1.0f;
		} else if (v11 > band - 0.5f) {
			v11 = 1E30f;
			v11 = (v01 < -1) ? v11 : min(v01, v11);
			v11 = (v12 < -1) ? v11 : min(v12, v11);
			v11 = (v10 < -1) ? v11 : min(v10, v11);
			v11 = (v21 < -1) ? v11 : min(v21, v11);
			v11 += 1.0f;
		}

		if (oldVal * v11 > 0) {
			levelSet(i, j) = v11;
		} else {
			levelSet(i, j) = oldVal;
		}
	}
	bool ActiveContour2D::stepInternal() {
		if (pressureImage.size() > 0) {
			if (vecFieldImage.size() > 0) {
#pragma omp parallel for
				for (int i = 0; i < (int) activeList.size(); i++) {
					int2 pos = activeList[i];
					pressureAndAdvectionMotion(pos.x, pos.y, i);
				}
			} else {
#pragma omp parallel for
				for (int i = 0; i < (int) activeList.size(); i++) {
					int2 pos = activeList[i];
					pressureMotion(pos.x, pos.y, i);
				}
			}
		} else if (vecFieldImage.size() > 0) {
#pragma omp parallel for
			for (int i = 0; i < (int) activeList.size(); i++) {
				int2 pos = activeList[i];
				advectionMotion(pos.x, pos.y, i);
			}
		}
		float timeStep =(float) mTimeStep;
		if (!clampSpeed) {
			float maxDelta = 0.0f;
			for (float delta : deltaLevelSet) {
				maxDelta = std::max(std::abs(delta), maxDelta);
			}
			const float maxSpeed = 0.999f;
			timeStep = (float)(mTimeStep * ((maxDelta > maxSpeed) ? (maxSpeed / maxDelta) : maxSpeed));
		}
		contourLock.lock();
		if (preserveTopology) {
			for (int nn = 0; nn < 4; nn++) {
#pragma omp parallel for
				for (int i = 0; i < (int) activeList.size(); i++) {
					int2 pos = activeList[i];
					applyForcesTopoRule(pos.x, pos.y, nn, i, timeStep);
				}
			}
		} else {
#pragma omp parallel for
			for (int i = 0; i < (int) activeList.size(); i++) {
				int2 pos = activeList[i];
				applyForces(pos.x, pos.y, i, timeStep);
			}
		}
		for (int band = 1; band <= maxLayers; band++) {
#pragma omp parallel for
			for (int i = 0; i < (int) activeList.size(); i++) {
				int2 pos = activeList[i];
				updateDistanceField(pos.x, pos.y, band, i);
			}
		}
#pragma omp parallel for
		for (int i = 0; i < (int) activeList.size(); i++) {
			int2 pos = activeList[i];
			plugLevelSet(pos.x, pos.y, i);
		}
		updateIsoSurface=true;
		contourLock.unlock();

#pragma omp parallel for
		for (int i = 0; i < (int) activeList.size(); i++) {
			int2 pos = activeList[i];
			swapLevelSet(pos.x, pos.y) = levelSet(pos.x, pos.y);
		}
		int deleted = deleteElements();
		int added = addElements();
		deltaLevelSet.clear();
		deltaLevelSet.resize(activeList.size(), 0.0f);
		mSimulationTime+=timeStep;
		mSimulationIteration++;
		if (cache.get() != nullptr) {
			Contour2D contour = getContour();
			//WriteContourToFile(, contour);
			contour.setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
			cache->set((int)mSimulationIteration, contour);
		}
		return (added+deleted != 0);
	}
	void ActiveContour2D::rescale(aly::Image1f& pressureForce) {
		float minValue = 1E30f;
		float maxValue = -1E30f;
		if (!std::isnan(targetPressure)) {
			for (int i = 0; i < pressureForce.size(); i++) {
				float val = pressureForce[i] - targetPressure;
				minValue = std::min(val, minValue);
				maxValue = std::max(val, maxValue);
			}
		}
		float normMin = (std::abs(minValue) > 1E-4) ? 1 / std::abs(minValue) : 1;
		float normMax = (std::abs(maxValue) > 1E-4) ? 1 / std::abs(maxValue) : 1;
#pragma omp parallel for
		for (int i = 0; i < pressureForce.size(); i++) {
			float val = pressureForce[i] - targetPressure;
			if (val < 0) {
				pressureForce[i] = (float) (val * normMin);
			} else {
				pressureForce[i] = (float) (val * normMax);
			}
		}
	}
}

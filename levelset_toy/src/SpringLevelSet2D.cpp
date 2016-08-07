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
#include "SpringLevelSet2D.h"
#include "AlloyApplication.h"
namespace aly {
	float SpringLevelSet2D::MIN_ANGLE_TOLERANCE = (float)(ALY_PI * 20 / 180.0f);
	float SpringLevelSet2D::NEAREST_NEIGHBOR_DISTANCE = 1.0f;
	float SpringLevelSet2D::PARTICLE_RADIUS = 0.05f;
	float SpringLevelSet2D::REST_RADIUS = 0.1f;
	float SpringLevelSet2D::SPRING_CONSTANT = 0.3f;
	float SpringLevelSet2D::EXTENT = 0.5f;
	float SpringLevelSet2D::SHARPNESS = 5.0f;
	SpringLevelSet2D::SpringLevelSet2D(const std::shared_ptr<SpringlCache2D>& cache) :ActiveContour2D("Spring Level Set 2D", cache) {
	}
	void SpringLevelSet2D::setSpringls(const Vector2f& particles, const Vector2f& points) {
		contour.particles = particles;
		contour.correspondence = particles;
		contour.points = points;
		contour.updateNormals();
	}
	void SpringLevelSet2D::updateUnsignedLevelSet(float maxDistance) {
		unsignedLevelSet = unsignedShader->solve(contour, maxDistance);
	}
	void SpringLevelSet2D::updateNearestNeighbors(float maxDistance) {
		matcher.reset(new Matcher2f(contour.points));
		nearestNeighbors.clear();
		nearestNeighbors.resize(contour.points.size(), std::list<uint32_t>());
		int N = (int)contour.points.size();
#pragma omp parallel for
		for (int i = 0;i < N;i += 2) {
			float2 pt0 = contour.points[i];
			float2 pt1 = contour.points[i + 1];
			std::vector<std::pair<size_t, float>> result;
			matcher->closest(pt0, maxDistance, result);
			for (auto pr : result) {
				if (pr.first != i&&pr.first != i + 1) {
					nearestNeighbors[i].push_back((uint32_t)pr.first);
					break;
				}
			}
			matcher->closest(pt1, maxDistance, result);
			for (auto pr : result) {
				if (pr.first != i&&pr.first != i + 1) {
					nearestNeighbors[i + 1].push_back((uint32_t)pr.first);
					break;
				}
			}
		}
	}
	int SpringLevelSet2D::fill() {
		{
			std::lock_guard<std::mutex> lockMe(contourLock);
			isoContour.solve(levelSet, contour.vertexes, contour.indexes, 0.0f, (preserveTopology) ? TopologyRule2D::Connect4 : TopologyRule2D::Unconstrained, Winding::Clockwise);
			updateIsoSurface = false;
		}
		int fillCount = 0;
		for (std::list<uint32_t> curve : contour.indexes) {
			size_t count = 0;
			uint32_t first = 0, prev = 0;
			if (curve.size() > 1) {
				for (uint32_t idx : curve) {
					if (count != 0) {
						float2 pt = 0.5f*(contour.vertexes[prev] + contour.vertexes[idx]);
						if (unsignedLevelSet(pt.x, pt.y).x >1.25f*EXTENT) {
							contour.particles.push_back(pt);
							contour.points.push_back(contour.vertexes[prev]);
							contour.points.push_back(contour.vertexes[idx]);
							fillCount++;
						}
						if (idx == first) break;
					}
					else {
						first = idx;
					}
					count++;
					prev = idx;
				}
			}
		}
		if (fillCount > 0) {
			contour.updateNormals();
			contour.setDirty(true);
		}
		return fillCount;
	}
	void SpringLevelSet2D::contract() {

	}
	void SpringLevelSet2D::computeForce(size_t idx, float2& f1, float2& f2, float2& f) {
		f1 = float2(0.0f);
		f2 = float2(0.0f);
		f = float2(0.0f);
		float2 p = contour.particles[idx];
		float2 p1 = contour.points[2 * idx];
		float2 p2 = contour.points[2 * idx + 1];

		if (pressureImage.size() > 0) {
			float2 v1 = normalize(contour.points[2 * idx + 1] - contour.points[2 * idx]);
			float2 norm = float2(-v1.y, v1.x);
			float2 pres = pressureWeight*norm*pressureImage(p.x, p.y).x;
			f = pres;
			f1 = f;
			f2 = f;
		}
		if (vecFieldImage.size() > 0) {
			float2 vec = vecFieldImage(p.x, p.y)*advectionWeight;
			f += vec;
			vec = vecFieldImage(p1.x, p1.y);
			f1 += vec;
			vec = vecFieldImage(p2.x, p2.y);
			f2 += vec;
		}
	}
	float SpringLevelSet2D::updateSignedLevelSet(float maxStep) {
#pragma omp parallel for
		for (int i = 0; i < (int)activeList.size(); i++) {
			int2 pos = activeList[i];
			distanceFieldMotion(pos.x, pos.y, i);
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
				updateDistanceField(pos.x, pos.y, band, i);
			}
		}
#pragma omp parallel for
		for (int i = 0; i < (int)activeList.size(); i++) {
			int2 pos = activeList[i];
			plugLevelSet(pos.x, pos.y, i);
		}
		updateIsoSurface = true;
		contourLock.unlock();

#pragma omp parallel for
		for (int i = 0; i < (int)activeList.size(); i++) {
			int2 pos = activeList[i];
			swapLevelSet(pos.x, pos.y) = levelSet(pos.x, pos.y);
		}
		int deleted = deleteElements();
		int added = addElements();
		deltaLevelSet.clear();
		deltaLevelSet.resize(activeList.size(), 0.0f);
	}
	float SpringLevelSet2D::advect(float maxStep) {
		Vector2f f(contour.particles.size());
		Vector2f f1(contour.particles.size());
		Vector2f f2(contour.particles.size());
#pragma omp parallel for
		for (int i = 0;i < (int)f.size();i++) {
			computeForce(i, f1[i], f2[i], f[i]);
		}
		float maxForce = 0.0f;
		for (int i = 0;i < (int)f.size();i++) {
			maxForce = std::max(maxForce, lengthSqr(f[i]));
		}
		maxForce = std::sqrt(maxForce);
		float timeStep = (maxForce > 1.0f) ? maxStep / (maxForce) : maxStep;
#pragma omp parallel for
		for (int i = 0;i < (int)f.size();i++) {
			contour.points[2 * i] += timeStep*f1[i];
			contour.points[2 * i + 1] += timeStep*f2[i];
			contour.particles[i] += timeStep*f[i];
		}
		contour.updateNormals();
		contour.setDirty(true);
		return timeStep;
	}
	void SpringLevelSet2D::relax(float timeStep) {
		Vector2f updates(contour.points.size());
#pragma omp parallel for
		for (int i = 0;i < (int)contour.particles.size();i++) {
			relax(i, timeStep, updates[2 * i], updates[2 * i + 1]);
		}
		contour.points = updates;
		contour.updateNormals();
		contour.setDirty(true);
	}
	void SpringLevelSet2D::relax(size_t idx, float timeStep, float2& f1, float2& f2) {
		const float maxForce = 0.999f;
		float2 particlePt = contour.particles[idx];
		float2 tangets[2];
		float2 motion[2];
		float springForce[2];
		float len, w, tlen, dotProd;
		float2 startVelocity;
		float2 start, dir;
		float resultantMoment = 0.0f;
		for (int i = 0; i < 2; i++) {
			size_t eid = idx * 2 + i;
			start = contour.points[eid];
			tangets[i] = (start - particlePt);
			tlen = length(tangets[i]);
			if (tlen > 1E-6f) tangets[i] /= tlen;
			startVelocity = float2(0, 0);
			for (uint32_t nbr : nearestNeighbors[eid]) {
				dir = (contour.points[nbr] - start);
				len = length(dir);
				w = atanh(maxForce*clamp(((len - 2 * PARTICLE_RADIUS) / (EXTENT + 2 * PARTICLE_RADIUS)), -1.0f, 1.0f));
				startVelocity += (w * dir);
			}
			motion[i] = timeStep*startVelocity*SHARPNESS;
			springForce[i] = timeStep*SPRING_CONSTANT*(2 * PARTICLE_RADIUS - tlen);
			resultantMoment += crossMag(motion[i], tangets[i]);
		}

		float cosa = std::cos(resultantMoment);
		float sina = std::sin(resultantMoment);

		std::pair<float2, float2> update;
		start = contour.points[idx * 2] - particlePt;
		dotProd = std::max(length(start) + dot(motion[0], tangets[0]) + springForce[0], 0.001f);
		start = dotProd*tangets[0];
		f1 = float2(start.x*cosa + start.y*sina, -start.x*sina + start.y*cosa) + particlePt;

		start = contour.points[idx * 2 + 1] - particlePt;
		dotProd = std::max(length(start) + dot(motion[1], tangets[1]) + springForce[1], 0.001f);
		start = dotProd*tangets[1];
		f2 = float2(start.x*cosa + start.y*sina, -start.x*sina + start.y*cosa) + particlePt;
	}
	float2 SpringLevelSet2D::getScaledGradientValue(int i, int j) {
		float v21 = unsignedLevelSet( i + 1, j).x;
		float v12 = unsignedLevelSet( i, j + 1).x;
		float v10 = unsignedLevelSet( i, j - 1).x;
		float v01 = unsignedLevelSet( i - 1, j).x;
		float v11 = unsignedLevelSet( i, j).x;
		float2 grad;
		grad.x = 0.5f*(v21 - v01);
		grad.y = 0.5f*(v12 - v10);
		float len = max(1E-6f, length(grad));
		return -(v11*grad / len);
	}
	void SpringLevelSet2D::distanceFieldMotion(int i, int j, size_t gid) {
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
			if (std::abs(denom) > 1E-5f) {
				kappa = curvatureWeight * numer / denom;
			}
			else {
				kappa = curvatureWeight * numer * sign(denom) * 1E5f;
			}
			if (kappa < -maxCurvatureForce) {
				kappa = -maxCurvatureForce;
			}
			else if (kappa > maxCurvatureForce) {
				kappa = maxCurvatureForce;
			}
			grad =getScaledGradientValue( i, j);
			float advection = 0;
			// Dot product force with upwind gradient
			if (grad.x > 0) {
				advection = grad.x * DxNeg;
			}
			else if (grad.x < 0) {
				advection = grad.x * DxPos;
			}
			if (grad.y > 0) {
				advection += grad.y * DyNeg;
			}
			else if (grad.y < 0) {
				advection += grad.y * DyPos;
			}
			deltaLevelSet[gid] = -advection + kappa;
	}
	bool SpringLevelSet2D::init() {
		ActiveContour2D::init();
		contour.points.clear();
		contour.particles.clear();
		for (std::list<uint32_t> curve : contour.indexes) {
			size_t count = 0;
			uint32_t first = 0, prev = 0;
			if (curve.size() > 1) {
				for (uint32_t idx : curve) {
					if (count != 0) {
						contour.particles.push_back(0.5f*(contour.vertexes[prev] + contour.vertexes[idx]));
						contour.points.push_back(contour.vertexes[prev]);
						contour.points.push_back(contour.vertexes[idx]);
						if (idx == first) break;
					}
					else {
						first = idx;
					}
					count++;
					prev = idx;
				}
			}
		}
		contour.correspondence = contour.particles;
		contour.updateNormals();
		contour.setDirty(true);
		if (cache.get() != nullptr) {
			Contour2D* contour = getContour();
			contour->setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
		}
		if (unsignedShader.get() == nullptr) {
			unsignedShader.reset(new UnsignedDistanceShader(true, AlloyApplicationContext()));
			unsignedShader->init(initialLevelSet.width, initialLevelSet.height);
			
		}
		relax();
		updateNearestNeighbors();
		updateUnsignedLevelSet();
		cache->set((int)mSimulationIteration, contour);
		return true;
	}
	void SpringLevelSet2D::relax() {
		const int maxIterations = 8;
		const float timeStep = 0.1f;
		updateNearestNeighbors();
		for (int i = 0;i < maxIterations;i++) {
			relax(timeStep);
		}
	}
	void SpringLevelSet2D::cleanup() {
		ActiveContour2D::cleanup();
	}
	bool SpringLevelSet2D::stepInternal() {
		double remaining = mTimeStep;
		double t = 0.0;
		const int evolveIterations = 8;
		do {
			float timeStep = advect(std::min(0.33333f,(float)remaining));
			t += (double)timeStep;
			relax();
			updateUnsignedLevelSet();
			for (int i = 0;i < evolveIterations;i++) {
				updateSignedLevelSet();
			}
			if (fill()>0) {
				relax();
			}
			contract();
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

	void SpringLevelSet2D::setup(const aly::ParameterPanePtr& pane) {
		ActiveContour2D::setup(pane);
	}
}
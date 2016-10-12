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
#include "MultiSpringLevelSet2D.h"
#include "AlloyApplication.h"
namespace aly {

	float MultiSpringLevelSet2D::MIN_ANGLE_TOLERANCE = (float)(ALY_PI * 20 / 180.0f);
	float MultiSpringLevelSet2D::NEAREST_NEIGHBOR_DISTANCE = std::sqrt(2.0f)*0.5f;
	float MultiSpringLevelSet2D::PARTICLE_RADIUS = 0.05f;
	float MultiSpringLevelSet2D::REST_RADIUS = 0.1f;
	float MultiSpringLevelSet2D::SPRING_CONSTANT = 0.3f;
	float MultiSpringLevelSet2D::EXTENT = 0.5f;
	float MultiSpringLevelSet2D::SHARPNESS = 5.0f;
	
	MultiSpringLevelSet2D::MultiSpringLevelSet2D(const std::shared_ptr<SpringlCache2D>& cache) :MultiActiveManifold2D("Multi Spring Level Set 2D", cache), resampleEnabled(true) {
	}
	void MultiSpringLevelSet2D::setSpringls(const Vector2f& particles, const Vector2f& points) {
		contour.particles = particles;
		contour.correspondence = particles;
		contour.points = points;
		contour.updateNormals();
	}
	void MultiSpringLevelSet2D::updateUnsignedLevelSet(float maxDistance) {
		unsignedLevelSet = unsignedShader->solve(contour, maxDistance);
	}
	void MultiSpringLevelSet2D::refineContour(bool signedIso) {
		//Optimize location of level set to remove jitter and improve spacing of iso-vertexes.
		int N = (int)contour.vertexes.size();
		Vector2f delta(N);
		float maxDelta;
		int iter = 0;
		do {
			maxDelta = 0.0f;
			for (std::list<uint32_t> curve : contour.indexes) {
				uint32_t cur = 0, prev = 0, next = 0;
				if (curve.size() > 1) {
					auto prevIter = curve.begin();
					auto curIter = prevIter;
					curIter++;
					auto nextIter = curIter;
					nextIter++;
					for (; nextIter != curve.end();curIter++, nextIter++, prevIter++) {
						cur = *curIter;
						prev = *prevIter;
						next = *nextIter;
						float2 curPt = contour.vertexes[cur];
						float2 nextPt = contour.vertexes[next];
						float2 prevPt = contour.vertexes[prev];
						float2 tan1 = (nextPt - curPt);
						float d1 = length(tan1);
						float2 tan2 = (curPt - prevPt);
						float d2 = length(tan2);
						float2 tan = normalize(tan1 + tan2);
						float2 norm(-tan.y, tan.x);
						float2 d = 0.1f*dot(norm, getScaledGradientValue(curPt.x, curPt.y, signedIso))*norm + 0.5f*tan*(d1 - d2) / (d1 + d2);
						maxDelta = std::max(lengthSqr(d), maxDelta);
						delta[cur] = d;
					}
				}
			}
			maxDelta = std::sqrt(maxDelta);
			contour.vertexes += delta;
			iter++;
		} while (iter < 10);
		//Add line search to find zero crossing.
	}
	float2 MultiSpringLevelSet2D::traceUnsigned(float2 pt) {
		float disp = 0.0f;
		int iter = 0;
		const float timeStep = 0.5f;
		float2 grad;
		float v11, v21, v12, v10, v01;
		do {
			v21 = unsignedLevelSet(pt.x + 1, pt.y).x;
			v12 = unsignedLevelSet(pt.x, pt.y + 1).x;
			v10 = unsignedLevelSet(pt.x, pt.y - 1).x;
			v01 = unsignedLevelSet(pt.x - 1, pt.y).x;
			v11 = unsignedLevelSet(pt.x, pt.y).x;

			grad.x = 0.5f*(v21 - v01);
			grad.y = 0.5f*(v12 - v10);
			grad *= v11;
			disp = length(grad);
			pt = pt - timeStep*grad;
			iter++;
		} while (disp > 1E-5f&&iter < 30);
		return pt;
	}

	float2 MultiSpringLevelSet2D::traceInitial(float2 pt) {
		float disp = 0.0f;
		int iter = 0;
		const float timeStep = 0.5f;
		float2 grad;
		float v11, v21, v12, v10, v01;
		do {
			v21 = std::abs(initialLevelSet(pt.x + 1, pt.y).x);
			v12 = std::abs(initialLevelSet(pt.x, pt.y + 1).x);
			v10 = std::abs(initialLevelSet(pt.x, pt.y - 1).x);
			v01 = std::abs(initialLevelSet(pt.x - 1, pt.y).x);
			v11 = std::abs(initialLevelSet(pt.x, pt.y).x);

			grad.x = 0.5f*(v21 - v01);
			grad.y = 0.5f*(v12 - v10);
			grad *= v11;
			disp = length(grad);
			pt = pt - timeStep*grad;
			iter++;
		} while (disp > 1E-5f&&iter < 30);
		return pt;
	}
	void MultiSpringLevelSet2D::updateNearestNeighbors(float maxDistance) {
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
				if ((int)pr.first != i && (int)pr.first != i + 1) {
					nearestNeighbors[i].push_back((uint32_t)pr.first);
					break;
				}
			}
			matcher->closest(pt1, maxDistance, result);
			for (auto pr : result) {
				if ((int)pr.first != i && (int)pr.first != i + 1) {
					nearestNeighbors[i + 1].push_back((uint32_t)pr.first);
					break;
				}
			}
		}
	}
	int MultiSpringLevelSet2D::fill() {
		{
			std::lock_guard<std::mutex> lockMe(contourLock);
			isoContour.solve(levelSet, labelImage, contour.vertexes, contour.vertexLabels, contour.indexes, 0.0f, (preserveTopology) ? TopologyRule2D::Connect4 : TopologyRule2D::Unconstrained, Winding::Clockwise);
			refineContour(false);
			requestUpdateContour = false;
		}
		int fillCount = 0;
		for (std::list<uint32_t> curve : contour.indexes) {
			size_t count = 0;
			uint32_t first = 0, prev = 0;
			if (curve.size() > 1) {
				for (uint32_t idx : curve) {
					if (count != 0) {
						float2 pt = 0.5f*(contour.vertexes[prev] + contour.vertexes[idx]);
						if (unsignedLevelSet(pt.x, pt.y).x > 0.5f*(NEAREST_NEIGHBOR_DISTANCE + EXTENT)) {
							contour.particles.push_back(pt);
							for (Vector2f& vel : contour.velocities) {
								vel.push_back(float2(0.0f));
							}
							contour.points.push_back(contour.vertexes[prev]);
							contour.points.push_back(contour.vertexes[idx]);
							contour.correspondence.push_back(float2(std::numeric_limits<float>::infinity()));
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

		return fillCount;
	}
	void MultiSpringLevelSet2D::updateTracking(float maxDistance) {
		int tries = 0;
		int invalid = 0;
		do {
			invalid = 0;
			matcher.reset(new Matcher2f(oldPoints));
			std::vector<int> retrack;
			for (size_t i = 0;i < contour.particles.size();i++) {
				if (std::isinf(contour.correspondence[i].x)) {
					retrack.push_back((int)i);
				}
			}
			int N = (int)retrack.size();
			for (int i = 0;i < N;i++) {
				int pid = retrack[i];
				int eid1 = pid * 2;
				int eid2 = pid * 2 + 1;
				float2 pt0 = contour.points[eid1];
				float2 pt1 = contour.points[eid2];
				std::vector<std::pair<size_t, float>> result;
				float2 q1(std::numeric_limits<float>::infinity());
				float2 q2(std::numeric_limits<float>::infinity());
				matcher->closest(pt0, maxDistance, result);
				std::array<float2, 4> velocities;
				for (auto pr : result) {
					q1 = oldCorrespondences[pr.first / 2];
					if (!std::isinf(q1.x)) {
						for (int nn = 0;nn < 4;nn++) {
							velocities[nn] = oldVelocities[nn][pr.first / 2];
						}
						break;
					}
				}
				result.clear();
				matcher->closest(pt1, maxDistance, result);
				for (auto pr : result) {
					q2 = oldCorrespondences[pr.first / 2];
					if (!std::isinf(q2.x)) {
						for (int nn = 0;nn < 4;nn++) {
							velocities[nn] += oldVelocities[nn][pr.first / 2];
						}
						break;
					}
				}

				if (!std::isinf(q1.x)) {
					if (!std::isinf(q2.x)) {
						for (int nn = 0;nn < 4;nn++) {
							contour.velocities[nn][pid] = 0.5f*velocities[nn];
							oldVelocities[nn].push_back(0.5f*velocities[nn]);
						}

						q1 = traceInitial(0.5f*(q1 + q2));
						contour.correspondence[pid] = q1;
						oldCorrespondences.push_back(q1);

						oldPoints.push_back(pt0);
						oldPoints.push_back(pt1);
					}
					else {
						for (int nn = 0;nn < 4;nn++) {
							contour.velocities[nn][pid] = velocities[nn];
							oldVelocities[nn].push_back(velocities[nn]);
						}
						contour.correspondence[pid] = q1;
						oldCorrespondences.push_back(q1);
						oldPoints.push_back(pt0);
						oldPoints.push_back(pt1);
					}
				}
				else if (!std::isinf(q2.x)) {
					for (int nn = 0;nn < 4;nn++) {
						contour.velocities[nn][pid] = velocities[nn];
						oldVelocities[nn].push_back(velocities[nn]);
					}
					contour.correspondence[pid] = q2;
					oldCorrespondences.push_back(q2);
					oldPoints.push_back(pt0);
					oldPoints.push_back(pt1);
				}
				else {
					invalid++;
				}
			}
			tries++;
		} while (invalid > 0 && tries<4);
	}
	int MultiSpringLevelSet2D::contract() {
		int contractCount = 0;
		Vector2f particles;
		Vector2f points;
		Vector2f normals;
		Vector2f correspondence;
		std::array<Vector2f, 4> velocities;
		particles.data.reserve(contour.particles.size());
		points.data.reserve(contour.points.size());
		normals.data.reserve(contour.normals.size());
		for (int i = 0;i<(int)contour.particles.size();i++) {
			float2 pt = contour.particles[i];
			float d1 = distance(contour.points[2 * i + 1], pt);
			float d2 = distance(contour.points[2 * i], pt);
			if (std::abs(levelSet(pt.x, pt.y).x) <= 1.25f*EXTENT
				&&d1>3.0f*PARTICLE_RADIUS
				&&d2 > 3.0f*PARTICLE_RADIUS
				&&d1 < 1.5f
				&&d2 < 1.5f
				) {
				particles.push_back(pt);
				points.push_back(contour.points[2 * i]);
				points.push_back(contour.points[2 * i + 1]);
				normals.push_back(contour.normals[i]);
				for (int nn = 0;nn < 4;nn++) {
					velocities[nn].push_back(contour.velocities[nn][i]);
				}
				correspondence.push_back(contour.correspondence[i]);
			}
			else {
				contractCount++;
			}
		}
		if (contractCount > 0) {
			contour.points = points;
			contour.normals = normals;
			contour.particles = particles;
			contour.velocities = velocities;
			contour.correspondence = correspondence;
			contour.setDirty(true);
		}
		return contractCount;
	}
	void MultiSpringLevelSet2D::computeForce(size_t idx, float2& f1, float2& f2, float2& f) {
		f1 = float2(0.0f);
		f2 = float2(0.0f);
		f = float2(0.0f);
		float2 p = contour.particles[idx];
		float2 p1 = contour.points[2 * idx];
		float2 p2 = contour.points[2 * idx + 1];
		if (pressureImage.size() > 0 && pressureParam.toFloat() != 0.0f) {
			float2 v1 = normalize(contour.points[2 * idx + 1] - contour.points[2 * idx]);
			float2 norm = float2(-v1.y, v1.x);
			float2 pres = pressureParam.toFloat()*norm*pressureImage(p.x, p.y).x;
			f = pres;
			f1 = f;
			f2 = f;
		}

		float2x2 M, A;
		if (vecFieldImage.size() > 0 && advectionParam.toFloat() != 0.0f) {
			float w = advectionParam.toFloat();
			f1 += vecFieldImage(p1.x, p1.y)*w;
			f2 += vecFieldImage(p2.x, p2.y)*w;
			f += vecFieldImage(p.x, p.y)*w;
		}
		float2 k1, k2, k3, k4;
		k4 = contour.velocities[2][idx];
		k3 = contour.velocities[1][idx];
		k2 = contour.velocities[0][idx];
		k1 = f;

		contour.velocities[3][idx] = k4;
		contour.velocities[2][idx] = k3;
		contour.velocities[1][idx] = k2;
		contour.velocities[0][idx] = k1;
		if (mSimulationIteration >= 4) {
			f = (1.0f / 6.0f)*(k1 + 2.0f * k2 + 2.0f * k3 + k4);
		}
		else if (mSimulationIteration == 3) {
			f = (1.0f / 4.0f)*(k1 + 2.0f * k2 + k3);
		} if (mSimulationIteration == 2) {
			f = (1.0f / 2.0f)*(k1 + k2);
		}
		float2 v1 = p1 + f1;
		float2 v2 = p2 + f2;
		float2 v = p + f;
		//Correction keeps particle and points on same line segment
		float2 t = normalize(v2 - v1);
		float2 correction = v - (v1 + dot(t, v - v1)*t);
		f1 += correction;
		f2 += correction;
	}
	void MultiSpringLevelSet2D::updateSignedLevelSet(float maxStep) {
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
				updateDistanceField(pos.x, pos.y, band);
			}
		}
#pragma omp parallel for
		for (int i = 0; i < (int)activeList.size(); i++) {
			int2 pos = activeList[i];
			plugLevelSet(pos.x, pos.y, i);
		}
		requestUpdateContour = true;
		contourLock.unlock();

#pragma omp parallel for
		for (int i = 0; i < (int)activeList.size(); i++) {
			int2 pos = activeList[i];
			swapLevelSet(pos.x, pos.y) = levelSet(pos.x, pos.y);
		}
		deleteElements();
		addElements();
		deltaLevelSet.clear();
		deltaLevelSet.resize(activeList.size(), 0.0f);
	}
	float MultiSpringLevelSet2D::advect(float maxStep) {
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
	void MultiSpringLevelSet2D::relax(float timeStep) {
		Vector2f updates(contour.points.size());
#pragma omp parallel for
		for (int i = 0;i < (int)contour.particles.size();i++) {
			relax(i, timeStep, updates[2 * i], updates[2 * i + 1]);
		}
		contour.points = updates;
		contour.updateNormals();
		contour.setDirty(true);
	}
	void MultiSpringLevelSet2D::relax(size_t idx, float timeStep, float2& f1, float2& f2) {
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
	float2 MultiSpringLevelSet2D::getScaledGradientValue(int i, int j) {
		float v21 = unsignedLevelSet(i + 1, j).x;
		float v12 = unsignedLevelSet(i, j + 1).x;
		float v10 = unsignedLevelSet(i, j - 1).x;
		float v01 = unsignedLevelSet(i - 1, j).x;
		float v11 = unsignedLevelSet(i, j).x;
		float2 grad;
		grad.x = 0.5f*(v21 - v01);
		grad.y = 0.5f*(v12 - v10);
		float len = max(1E-6f, length(grad));
		return -(v11*grad / len);
	}
	float2 MultiSpringLevelSet2D::getScaledGradientValue(float i, float j, bool signedIso) {
		float2 grad;
		float v21;
		float v12;
		float v10;
		float v01;
		float v11;
		if (signedIso) {
			v21 = std::abs(levelSet(i + 1, j).x);
			v12 = std::abs(levelSet(i, j + 1).x);
			v10 = std::abs(levelSet(i, j - 1).x);
			v01 = std::abs(levelSet(i - 1, j).x);
			v11 = std::abs(levelSet(i, j).x);
		}
		else {
			v21 = unsignedLevelSet(i + 1, j).x;
			v12 = unsignedLevelSet(i, j + 1).x;
			v10 = unsignedLevelSet(i, j - 1).x;
			v01 = unsignedLevelSet(i - 1, j).x;
			v11 = unsignedLevelSet(i, j).x;
		}
		grad.x = 0.5f*(v21 - v01);
		grad.y = 0.5f*(v12 - v10);
		float len = max(1E-6f, length(grad));
		return -(v11*grad / len);
	}
	void MultiSpringLevelSet2D::distanceFieldMotion(int i, int j, size_t gid) {
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
		grad = getScaledGradientValue(i, j);
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
	bool MultiSpringLevelSet2D::init() {
		MultiActiveManifold2D::init();
		refineContour(true);
		contour.points.clear();
		contour.particles.clear();
		contour.particleLabels.clear();
		for (Vector2f& vel : contour.velocities) {
			vel.clear();
		}
		for (std::list<uint32_t> curve : contour.indexes) {
			size_t count = 0;
			uint32_t first = 0, prev = 0;
			if (curve.size() > 1) {
				for (uint32_t idx : curve) {
					if (count != 0) {
						contour.particles.push_back(0.5f*(contour.vertexes[prev] + contour.vertexes[idx]));
						contour.particleLabels.push_back(int1(contour.vertexLabels[idx]));
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
		for (Vector2f& vel : contour.velocities) {
			vel.resize(contour.particles.size());
		}
		contour.correspondence = contour.particles;
		contour.updateNormals();
		contour.setDirty(true);
		if (cache.get() != nullptr) {
			Manifold2D* contour = getContour();
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
	void MultiSpringLevelSet2D::relax() {
		const int maxIterations = 4;
		const float timeStep = 0.1f;
		updateNearestNeighbors();
		for (int i = 0;i < maxIterations;i++) {
			relax(timeStep);
		}
	}
	void MultiSpringLevelSet2D::cleanup() {
		MultiActiveManifold2D::cleanup();
	}
	bool MultiSpringLevelSet2D::stepInternal() {
		double remaining = mTimeStep;
		double t = 0.0;
		const int evolveIterations = 8;
		do {
			float timeStep = advect(std::min(0.33333f, (float)remaining));
			t += (double)timeStep;
			if (resampleEnabled) {
				relax();
			}
			updateUnsignedLevelSet();
			for (int i = 0;i < evolveIterations;i++) {
				updateSignedLevelSet();
			}
			if (resampleEnabled) {
				oldPoints = contour.points;
				oldCorrespondences = contour.correspondence;
				oldVelocities = contour.velocities;
				contract();
				updateNearestNeighbors();
				int fillCount = 0;
				do {
					updateUnsignedLevelSet();
					fillCount = fill();
					relax();
				} while (fillCount > 0); //Continue filling until all gaps are closed
				contour.updateNormals();
				contour.setDirty(true);
				updateTracking();
			}
			else {
				std::lock_guard<std::mutex> lockMe(contourLock);
				isoContour.solve(levelSet, labelImage, contour.vertexes, contour.vertexLabels, contour.indexes, 0.0f, (preserveTopology) ? TopologyRule2D::Connect4 : TopologyRule2D::Unconstrained, Winding::Clockwise);
				refineContour(false);
				contour.updateNormals();
				contour.setDirty(true);
				requestUpdateContour = false;
			}
			remaining = mTimeStep - t;
		} while (remaining > 1E-5f);
		mSimulationTime += t;
		mSimulationIteration++;
		if (cache.get() != nullptr) {
			Manifold2D* contour = getContour();
			refineContour(false);
			contour->setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
			cache->set((int)mSimulationIteration, *contour);
		}
		return (mSimulationTime < mSimulationDuration);
	}

	void MultiSpringLevelSet2D::setup(const aly::ParameterPanePtr& pane) {
		MultiActiveManifold2D::setup(pane);
		pane->addCheckBox("Re-sampling", resampleEnabled);
	}
}

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
	float SpringLevelSet2D::NEAREST_NEIGHBOR_DISTANCE = 1.5f;
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
		nearestNeighbors.resize(contour.points.size(),std::list<uint32_t>());
		int N = (int)contour.points.size();
#pragma omp parallel for
		for (int i = 0;i < N;i += 2) {
			float2 pt0 = contour.points[i];
			float2 pt1 = contour.points[i + 1];
			std::vector<std::pair<size_t, float>> result;
			matcher->closest(pt0, maxDistance, result);
			for (auto pr : result) {
				if(pr.first != i&&pr.first != i + 1){
					nearestNeighbors[i].push_back((uint32_t)pr.first);
					break;
				}
			}
			matcher->closest(pt1, maxDistance, result);
			for (auto pr : result) {
				if (pr.first != i&&pr.first != i + 1) {
					nearestNeighbors[i+1].push_back((uint32_t)pr.first);
					break;
				}
			}
		}
	}
	void SpringLevelSet2D::relax( float timeStep) {
		Vector2f updates(contour.points.size());
#pragma omp parallel for
		for (int i = 0;i < (int)contour.particles.size();i++) {
			auto pr = relax(i, timeStep);
			updates[2 * i] = pr.first;
			updates[2 * i + 1] = pr.second;
		}
		contour.points = updates;
		contour.setDirty(true);
	}
	std::pair<float2,float2> SpringLevelSet2D::relax(size_t idx,float timeStep) {
		const float maxForce = 0.999f;
		float2 particlePt = contour.particles[idx];
		float2 tangets[2];
		float2 motion[2];
		float springForce[2];
		float len, w, tlen, dotProd;
		float2 startVelocity;
		float2 start,dir;
		float resultantMoment=0.0f;
		for (int i = 0; i < 2; i++) {
			size_t eid = idx * 2 + i;
			start = contour.points[eid];
			tangets[i] = (start - particlePt);
			tlen = length(tangets[i]);
			if (tlen> 1E-6f) tangets[i] *= (1.0f / tlen);
			startVelocity = float2(0, 0);
			for (uint32_t nbr:nearestNeighbors[eid]) {
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
		start =contour.points[idx*2] -particlePt;
		dotProd = std::max(length(start)+ dot(motion[0], tangets[0])+ springForce[0],0.001f);
		start = dotProd*tangets[0];	
		update.first = float2(start.x*cosa + start.y*sina, -start.x*sina + start.y*cosa) + particlePt;

		start = contour.points[idx * 2+1] - particlePt;
		dotProd = std::max(length(start) + dot(motion[1], tangets[1]) + springForce[1], 0.001f);
		start = dotProd*tangets[1];
		update.second = float2(start.x*cosa + start.y*sina, -start.x*sina + start.y*cosa) + particlePt;
		return update;
	}
	bool SpringLevelSet2D::init() {
		ActiveContour2D::init();
		if (contour.particles.size() == 0) {
			contour.points.clear();
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
				Contour2D contour = getContour();
				contour.setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "contour" << std::setw(4) << std::setfill('0') << mSimulationIteration << ".bin");
				cache->set((int)mSimulationIteration, contour);
			}
		}
		if (unsignedShader.get() == nullptr) {
			unsignedShader.reset(new UnsignedDistanceShader(true, AlloyApplicationContext()));
			unsignedShader->init(initialLevelSet.width, initialLevelSet.height);
			updateUnsignedLevelSet();
			relax();
			updateNearestNeighbors();
			cache->set((int)mSimulationIteration, contour);
		}
		return true;
	}
	void SpringLevelSet2D::relax() {
		const int maxIterations = 8;
		const float timeStep = 0.1f;
		for (int i = 0;i < maxIterations ;i++) {
			updateNearestNeighbors();
			relax(timeStep);
		}
	}
	void SpringLevelSet2D::cleanup() {
		ActiveContour2D::cleanup();
	}
	bool SpringLevelSet2D::stepInternal() {
		return ActiveContour2D::stepInternal();
	}
	void SpringLevelSet2D::setup(const aly::ParameterPanePtr& pane) {
		ActiveContour2D::setup(pane);
	}
}
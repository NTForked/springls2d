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
#include "SLIC.h"
#include <AlloyImageProcessing.h>
namespace aly {
	SuperPixels::SuperPixels() :perturbSeeds(true), numLabels(0), bonusThreshold(10.0f), bonus(1.5f), errorThreshold(0.01f){
	}
	void SuperPixels::initializeSeeds(int K) {
		size_t sz = labImage.width*labImage.height;
		float step = std::sqrt((float)(sz) / (float)(K));
		float xoff = (step / 2.0f);
		float yoff = (step / 2.0f);
		colorCenters.clear();
		pixelCenters.clear();
		for (int y = 0; y < labImage.height; y++)
		{
			float yy = std::floor(y*step + yoff);
			if (yy > labImage.height - 1) break;
			for (int x = 0; x < labImage.width; x++)
			{
				float xx = (x*step + ((y & 0x1) ? (2.0f*xoff) : xoff));//hex grid
				if (xx > labImage.width - 1) break;
				colorCenters.push_back(labImage(xx, yy));
				pixelCenters.push_back(float2(xx, yy));
			}
		}
	}
	void SuperPixels::enforceLabelConnectivity() //the number of superpixels desired by the user
	{
		const int dx4[4] = { -1,  0,  1,  0 };
		const int dy4[4] = { 0, -1,  0,  1 };
		int K = (int)colorCenters.size();
		const int SUPSZ = (labImage.width*labImage.height) / K;
		Image1i newLabels(labelImage.width, labelImage.height);
		newLabels.set(int1(-1));
		int label = 0;
		Vector2i vec(labImage.width* labImage.height);//temporary buffer with worse case memory storage
		int adjlabel = 0;//adjacent label
		for (int j = 0; j < labImage.height; j++)
		{
			for (int i = 0; i < labImage.width; i++)
			{
				if (newLabels(i, j).x < 0)
				{
					newLabels(i, j).x = label;
					//--------------------
					// Start a new segment
					//--------------------
					vec[0] = int2(i, j);
					//-------------------------------------------------------
					// Quickly find an adjacent label for use later if needed
					//-------------------------------------------------------
					for (int n = 0; n < 4; n++)
					{
						int2 v = vec[0];
						int x = v.x + dx4[n];
						int y = v.y + dy4[n];
						if ((x >= 0 && x < labImage.width) && (y >= 0 && y < labImage.height))
						{
							int l = newLabels(x, y).x;
							if (l >= 0) adjlabel = l;
						}
					}
					int count = 1;
					for (int c = 0; c < count; c++)
					{
						for (int n = 0; n < 4; n++)
						{
							int2 v = vec[c];
							int x = v.x + dx4[n];
							int y = v.y + dy4[n];
							if ((x >= 0 && x < labImage.width) && (y >= 0 && y < labImage.height))
							{
								if (0 > newLabels(x, y).x && labelImage(i, j).x == labelImage(x, y).x)
								{
									vec[count] = int2(x, y);
									newLabels(x, y) = label;
									count++;
								}
							}

						}
					}
					//-------------------------------------------------------
					// If segment size is less then a limit, assign an
					// adjacent label found before, and decrement label count.
					//-------------------------------------------------------
					if (count <= SUPSZ >> 2)
					{
						for (int c = 0; c < count; c++)
						{
							int2 v = vec[c];
							newLabels(v.x, v.y) = adjlabel;
						}
						label--;
					}
					label++;
				}
			}
		}
		labelImage = newLabels;
		numLabels = label;
		updateClusters(labelImage);
		updateMaxColor(labelImage);
	}
	void SuperPixels::refineSeeds(const Image1f& magImage) {
		const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
		const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };
#pragma omp parallel for
		for (int n = 0; n < (int)pixelCenters.size(); n++)
		{
			float2 center = pixelCenters[n];
			float2 bestCenter = center;
			float bestMag = magImage(center).x;
			for (int i = 0; i < 8; i++)
			{
				float nx = center.x + dx8[i];//new x
				float ny = center.y + dy8[i];//new y
				float mag = magImage(nx, ny).x;
				if (mag < bestMag) {
					bestMag = mag;
					bestCenter = float2((float)nx, (float)ny);
				}
			}
			if (bestCenter != center)
			{
				colorCenters[n] = labImage(bestCenter);
				pixelCenters[n] = bestCenter;
			}
		}
	}
	void SuperPixels::optimize(int iterations) {
		S = std::sqrt((labImage.width*labImage.height) / (float)(colorCenters.size())) + 2.0f;//adding a small value in the even the S size is too small.
		int numk = (int)colorCenters.size();
		numLabels = numk;
		float offset = S;
		if (S < bonusThreshold) offset = S*bonus;
		colorMean.resize(numk);
		pixelMean.resize(numk);
		clustersize.resize(numk, 0);
		Image2f distImage(labImage.width, labImage.height);
		Image1f scoreImage(labImage.width, labImage.height);
		labelImage.set(int1(-1));
		distImage.set(float2(1E10f));
		scoreImage.set(float1(1E10f));
		maxlab.resize(numk, 0.0f);
		float invxywt = 1.0f / (S*S);//NOTE: this is different from how usual SLIC/LKM works, but in original code implementation
		for (int iter = 0;iter < iterations;iter++)
		{
			scoreImage.set(float1(1E10f));
			if (iter > 0) {
				updateMaxColor(labelImage);
			}
			for (int n = 0; n < numk; n++)
			{
				float2 pixelCenter = pixelCenters[n];
				float3 colorCenter = colorCenters[n];
				int xMin = std::max(0, (int)std::floor(pixelCenter.x - offset));
				int xMax = std::min(labImage.width - 1, (int)std::ceil(pixelCenter.x + offset));
				int yMin = std::max(0, (int)std::floor(pixelCenter.y - offset));
				int yMax = std::min(labImage.height - 1, (int)std::ceil(pixelCenter.y + offset));
				float ml = (maxlab[n] > 0.0f) ? 1.0f / maxlab[n] : 0.0f;
				for (int y = yMin; y <= yMax; y++) {
					for (int x = xMin; x <= xMax; x++) {
						float3 c = labImage(x, y);
						float distLab = lengthSqr(c - colorCenter);
						float distPixel = lengthSqr(float2((float)x, (float)y) - pixelCenter);
						float dist = distLab*ml + distPixel * invxywt;
						float last = scoreImage(x, y).x;
						distImage(x, y) = float2(distLab, distPixel);
						if (dist < last)
						{
							scoreImage(x, y).x = dist;
							labelImage(x, y).x = n;
						}
						else if (dist == last&&n < labelImage(x, y).x) {//Tie breaker, use smaller label id
							scoreImage(x, y).x = dist;
							labelImage(x, y).x = n;
						}
					}
				}
			}
			float E = updateClusters(labelImage);
			if (E < errorThreshold)break;
		}
	}
	float SuperPixels::updateMaxColor(const Image1i& labelImage, int labelOffset) {
		maxlab.resize(numLabels);
		maxlab.assign(numLabels, 1.0f);
		float maxx = 0.0f;
		for (int j = 0;j < labImage.height;j++) {
			for (int i = 0; i < labImage.width; i++) {
				int idx = labelImage(i, j).x+labelOffset;
				if (idx >= 0) {
					if (idx >= numLabels) {
						std::cout << "Index exceeds max" << idx << std::endl;
					}
					float3 c = labImage(i, j);
					float3 colorCenter = colorCenters[idx];
					float distLab = lengthSqr(c - colorCenter);
					maxx = std::max(distLab, maxx);
					if (distLab > maxlab[idx]) {
						maxlab[idx] = distLab;
					}
				}
			}
		}
		maxx=std::sqrt(maxx);
		return maxx;
	}
	float SuperPixels::updateClusters(const Image1i& labelImage,int labelOffset) {
		colorMean.resize(numLabels);
		pixelMean.resize(numLabels);
		colorCenters.resize(numLabels);
		pixelCenters.resize(numLabels);
		clustersize.resize(numLabels);
		colorMean.set(float3(0.0f));
		pixelMean.set(float2(0.0f));
		clustersize.assign(clustersize.size(), 0);
		for (int j = 0;j < labImage.height;j++) {
			for (int i = 0; i < labImage.width; i++) {
				int idx = labelImage(i, j).x+ labelOffset;
				if (idx >= 0) {
					if (idx >= numLabels) {
						throw std::runtime_error("Invalid cluster id.");
					}
					colorMean[idx] += labImage(i, j);
					pixelMean[idx] += float2((float)i, (float)j);
					clustersize[idx]++;
				}
			}
		}
		//Recalculate centers
		float E = 0.0f;
#pragma omp parallel for reduction(+:E)
		for (int k = 0; k < numLabels; k++) {
			if (clustersize[k] <= 0) clustersize[k] = 1;
			float inv = 1.0f / (float)(clustersize[k]);
			float3 newColor = colorMean[k] * inv;
			float2 newPixel = pixelMean[k] * inv;
			colorCenters[k] = newColor;
			E += aly::distance(newPixel, pixelCenters[k]);
			pixelCenters[k] = newPixel;
		}
		E /= (float)numLabels;
		return E;
	}
	float SuperPixels::distance(int x, int y, int label) const {
		if (label >= 0&&label<numLabels) {
			float invxywt = 1.0f / (S*S);//NOTE: this is different from how usual SLIC/LKM works, but in original code implementation
			float3 c = labImage(x, y);
			float2 pixelCenter = pixelCenters[label];
			float3 colorCenter = colorCenters[label];
			float distLab = lengthSqr(c - colorCenter);
			float distPixel = lengthSqr(float2((float)x, (float)y) - pixelCenter);
			float ml = (maxlab[label] > 0.0f) ? 1.0f / maxlab[label] : 0.0f;
			float dist = std::sqrt(distLab*ml + distPixel * invxywt);
			return dist;
		}
		else {
			throw std::runtime_error("Invalid cluster id.");
			return 0;
		}
	}
	void SuperPixels::solve(const ImageRGBA& image, int K, int iterations) {
		labImage.resize(image.width, image.height);
		labelImage.resize(image.width, image.height);
#pragma omp parallel for
		for (int j = 0;j < image.height;j++) {
			for (int i = 0;i < image.width;i++) {
				labImage(i, j) = RGBtoLAB(ToRGBf(image(i, j)));
			}
		}
		initializeSeeds(K);
		if (perturbSeeds)
		{
			Image1f magImage;
			gradientMagnitude(magImage);
			refineSeeds(magImage);
		}
		optimize(iterations);
		enforceLabelConnectivity();
	}
	void SuperPixels::solve(const ImageRGBAf& image, int K, int iterations) {
		labImage.resize(image.width, image.height);
		labelImage.resize(image.width, image.height);
#pragma omp parallel for
		for (int j = 0;j < image.height;j++) {
			for (int i = 0;i < image.width;i++) {
				labImage(i, j) = RGBtoLAB(image(i, j).xyz());
			}
		}
		initializeSeeds(K);
		if (perturbSeeds)
		{
			Image1f magImage;
			gradientMagnitude(magImage);
			refineSeeds(magImage);
		}
		optimize(iterations);
		enforceLabelConnectivity();
	}
	void SuperPixels::gradientMagnitude(Image1f& magImage)
	{
		ImageRGBf Gx, Gy;
		Gradient5x5(labImage, Gx, Gy);
		magImage.resize(labImage.width, labImage.height);
#pragma omp parallel for
		for (int j = 0; j < labImage.height; j++) {
			for (int i = 0; i < labImage.width; i++) {
				float3 gx = Gx(i, j);
				float3 gy = Gy(i, j);
				magImage(i, j) = float1(std::sqrt(lengthSqr(gx) + lengthSqr(gy)));
			}
		}
	}
}

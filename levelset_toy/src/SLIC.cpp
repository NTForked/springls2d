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
	SuperPixels::SuperPixels():perturbSeeds(true),numLabels(0) {
	}
	void SuperPixels::initializeSeeds(int K) {
		size_t sz = labImage.width*labImage.height;
		float step = std::sqrt((float)(sz) / (float)(K));

		float xoff =(step / 2.0f);
		float yoff =(step / 2.0f);
		int n=0;
		int r=0;
		colorCenters.clear();
		pixelCenters.clear();
		for (int y = 0; y < labImage.height; y++)
		{
			int yy = (int)std::floor(y*step + yoff);
			if (yy > labImage.height - 1) break;
			for (int x = 0; x < labImage.width; x++)
			{
				int xx =(int)std::floor(x*step + (((int)xoff) << (r & 0x1)));//hex grid
				if (xx > labImage.width - 1) break;
				colorCenters.push_back(labImage(xx, yy));
				pixelCenters.push_back(float2((float)xx,(float) yy));
				n++;
			}
			r++;
		}
	}
	void SuperPixels::enforceLabelConnectivity() //the number of superpixels desired by the user
	{
		//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
		//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
		const int dx4[4] = { -1,  0,  1,  0 };
		const int dy4[4] = { 0, -1,  0,  1 };
		int K = (int)colorCenters.size();
		const int SUPSZ =(labImage.width*labImage.height) / K;
		Image1i newLabels(labelImage.width,labelImage.height);
		newLabels.set(int1(-1));
		int label=0;
		Vector2i vec(labImage.width* labImage.height);
		int oindex=0;
		int adjlabel=0;//adjacent label
		for (int j = 0; j < labImage.height; j++)
		{
			for (int k = 0; k < labImage.width; k++)
			{
				if (newLabels[oindex].x<0)
				{
					newLabels[oindex].x = label;
					//--------------------
					// Start a new segment
					//--------------------
					vec[0] = int2(k, j);
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
							int nindex = y*labImage.width + x;
							int l = newLabels[nindex].x;
							if (l >= 0) adjlabel =l;
						}
					}
					int count=1;
					for (int c = 0; c < count; c++)
					{
						for (int n = 0; n < 4; n++)
						{
							int2 v = vec[c];
							int x = v.x + dx4[n];
							int y = v.y + dy4[n];

							if ((x >= 0 && x < labImage.width) && (y >= 0 && y < labImage.height))
							{
								int nindex = y*labImage.width + x;

								if (0 > newLabels[nindex].x && labelImage[oindex] == labelImage[nindex])
								{
									vec[count] = int2(x,y);
									newLabels[nindex] = label;
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
							int ind = v.y * labImage.width + v.x;
							newLabels[ind] = adjlabel;
						}
						label--;
					}
					label++;
				}
				oindex++;
			}
		}
		numLabels = label;
	}
	void SuperPixels::refineSeeds(const Image1f& magImage) {
		const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
		const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };
#pragma omp parallel for
		for (int n = 0; n <  (int)pixelCenters.size(); n++)
		{
			float2 center = pixelCenters[n];
			float2 bestCenter = center;
			float bestMag = magImage(center).x;
			for (int i = 0; i < 8; i++)
			{
				float nx = center.x + dx8[i];//new x
				float ny =center.y + dy8[i];//new y
				float mag = magImage(nx, ny).x;
				if (mag< bestMag){
					bestMag = mag;
					bestCenter = float2((float)nx,(float)ny);
				}
			}
			if (bestCenter != center)
			{
				colorCenters[n] = labImage(bestCenter);
				pixelCenters[n] = bestCenter;
			}
		}
	}
	void SuperPixels::optimize(int NUMITR) {
		float STEP = std::sqrt((labImage.width*labImage.height) /(float)(colorCenters.size())) + 2.0f;//adding a small value in the even the STEP size is too small.
		int numk = (int)colorCenters.size();
		float offset = STEP;
		if (STEP < 10) offset = STEP*1.5f;
		//----------------
		Vector3f colorSigma(numk);
		Vector2f pixelSigma(numk);
		std::vector<int> clustersize(numk,0);
		std::vector<float> inv(numk, 0);//to store 1/clustersize[k] values
		Image2f distImage(labImage.width, labImage.height);
		Image1f scoreImage(labImage.width, labImage.height);
		labelImage.set(int1(-1));
		distImage.set(float2(1E10f));
		scoreImage.set(float1(1E10f));

		std::vector<float> maxlab(numk, 10 * 10);//THIS IS THE VARIABLE VALUE OF M, just start with 10
		std::vector<float> maxxy(numk, STEP*STEP);//THIS IS THE VARIABLE VALUE OF M, just start with 10
		float invxywt = 1.0f / (STEP*STEP);//NOTE: this is different from how usual SLIC/LKM works
		for (int numitr = 0;numitr < NUMITR;numitr++)
		{
			scoreImage.set(float1(1E10f));
			for (int n = 0; n < numk; n++)
			{
				float2 pixelCenter = pixelCenters[n];
				float3 colorCenter = colorCenters[n];
				int y1 = std::max(0,(int)(pixelCenter.y - offset));
				int y2 = std::min(labImage.height-1,(int)(pixelCenter.y + offset));
				int x1 = std::max(0, (int)(pixelCenter.x - offset));
				int x2 = std::min(labImage.width-1, (int)(pixelCenter.x + offset));
				for (int y = y1; y < y2; y++)
				{
					for (int x = x1; x < x2; x++)
					{
						float3 c = labImage(x, y);
						float distLab = lengthSqr(c - colorCenter);
						float distPixel = lengthSqr(float2((float)x, (float)y) - pixelCenter);
						float dist = distLab/ maxlab[n] + distPixel * invxywt;
						float last = scoreImage(x, y).x;
						distImage(x, y) = float2(distLab, distPixel);
						if (dist < last)
						{
							scoreImage(x, y).x = dist;
							labelImage(x, y) = n;
						}
					}
				}
			}

			for (int j = 0;j < labImage.height;j++){
				for (int i = 0; i < labImage.width; i++){
					float2 dist = distImage(i, j);
					if (maxlab[labelImage(i,j).x] < dist.x) maxlab[labelImage(i, j).x] = dist.x;
					if (maxxy[labelImage(i, j).x] < dist.y) maxxy[labelImage(i, j).x] = dist.y;
				}
			}
			//-----------------------------------------------------------------
			// Recalculate the centroid and store in the seed values
			//-----------------------------------------------------------------
			colorSigma.set(float3(0.0f));
			pixelSigma.set(float2(0.0f));
			clustersize.assign(clustersize.size(), 0);
			for (int j = 0;j < labImage.height;j++) {
				for (int i = 0; i < labImage.width; i++) {
					int idx = labelImage(i, j).x;
					colorSigma[idx] += labImage(i, j);
					pixelSigma[idx] += float2((float)i, (float)j);
					clustersize[idx]++;
				}
			}
			for (int k = 0; k < numk; k++){
				if (clustersize[k] <= 0) clustersize[k] = 1;
				inv[k] = 1.0f / (float)(clustersize[k]);//computing inverse now to multiply, than divide later
			}
			for (int k = 0; k < numk; k++)
			{
				colorCenters[k] = colorSigma[k] * inv[k];
				pixelCenters[k] = pixelSigma[k] * inv[k];
			}
		}
	}
	void SuperPixels::solve(const ImageRGBAf& image,int K) {
		labImage.resize(image.width, image.height);
		labelImage.resize(image.width, image.height);
		
#pragma omp parallel for
		for (int j = 0;j < image.height;j++) {
			for (int i = 0;i < image.width;i++) {
				labImage(i,j)=RGBtoLAB(image(i, j).xyz());
			}
		}
		initializeSeeds(K);
		if (perturbSeeds)
		{
			Image1f magImage;
			gradientMagnitude(magImage);
			refineSeeds(magImage);
		}
		optimize();
		enforceLabelConnectivity();
	}
	void SuperPixels::gradientMagnitude(Image1f& magImage)
	{
		magImage.resize(labImage.width, labImage.height);
#pragma omp parallel for
		for (int j = 0; j < labImage.height; j++){
			for (int i = 0; i < labImage.width; i++){
				RGBf v10 = labImage(i, j - 1);
				RGBf v12 = labImage(i, j + 1);
				RGBf v01 = labImage(i-1, j);
				RGBf v21 = labImage(i+1, j);
				float dx = length(v21 - v01);
				float dy = length(v12 - v10);
				magImage(i,j) = float1(dx + dy);
			}
		}
	}
}

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

//  This class is an implementation of the SLIC Superpixel algorithm by Achanta et al. [PAMI'12,vol. 34, num. 11, pp. 2274-2282].

#ifndef _SLIC_H
#define _SLIC_H

#include <vector>
#include <AlloyMath.h>
#include <AlloyVector.h>
#include <AlloyImage.h>
#include <AlloyUnits.h>
#include <array>
using namespace std;
namespace aly {
	int ComputeConnectedComponents(const Image1i& labels, Image1i& outLabels, std::vector<int> &compCounts);
	int RemoveSmallConnectedComponents(const Image1i& labels, Image1i& outImage, int minSize);
	int MakeLabelsUnique(Image1i& outImage);
	class SuperPixels
	{
	private:
		ImageRGBf labImage;
		Image1i labelImage;
		Vector3f colorCenters;
		Vector2f pixelCenters;
		std::vector<int> clustersize;
		Vector3f colorMean;
		Vector2f pixelMean;
		std::vector<float> maxlab;
		bool perturbSeeds;
		int numLabels;
		float bonusThreshold;
		float bonus;
		float errorThreshold;
		float S;
		void initializeSeeds(int K);
		void refineSeeds(const Image1f& magImage);
		void gradientMagnitude(Image1f& magImage);
		void optimize(int NUMITR=10);
		void enforceLabelConnectivity();
	public:
		void solve(const ImageRGBAf& image,int K,int iterations=128);
		void solve(const ImageRGBA& image, int K, int iterations = 128);
		float updateClusters(const Image1i& labelImage,int labelOffset=0);
		float updateMaxColor(const Image1i& labelImage, int labelOffset = 0);
		float distance(int x, int y,int label) const;
		float distanceColor(int x, int y, int label) const;

		float2 getPixelCenter(int label) const {
			return pixelCenters[label];
		}
		RGBAf getColorCenter(int label) const {
			return LABAtoRGBA(float4(colorCenters[label],1.0f));
		}
		void setPerturbSeeds(bool p) {
			perturbSeeds = p;
		}
		Vector3f& getColorCenters() {
			return colorCenters;
		}
		const Vector3f& getColorCenters() const {
			return colorCenters;
		}
		Vector2f& getPixelCenters() {
			return pixelCenters;
		}
		const Vector2f& getPixelCenters() const{
			return pixelCenters;
		}
		Image1i& getLabelImage() {
			return labelImage;
		}
		const Image1i& getLabelImage() const {
			return labelImage;
		}
		const Image3f& getColorImage() const {
			return labImage;
		}
		Image3f& getColorImage() {
			return labImage;
		}
		int getNumLabels() const {
			return numLabels;
		}
		SuperPixels();
	};
}
#endif

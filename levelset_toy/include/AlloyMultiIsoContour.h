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

#ifndef ALLOYMULTIISOCONTOUR_H_
#define ALLOYMULTIISOCONTOUR_H_
#include "AlloyImage.h"
#include "AlloyVector.h"
#include "AlloyIsoContour.h"
#include <memory>
#include <list>
#include <map>
namespace aly {
	struct MultiContourData {
		std::map<uint64_t, EdgeSplitPtr> splits;
		std::list<EdgePtr> edges;
		int vertCount=0;
	};
class MultiIsoContour {
protected:

	const int a2fVertex1Offset[4][2] =
			{ { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
	const int a2fVertex2Offset[4][2] =
			{ { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0, 0 } };
	const int afSquareValue4[16][4] = { { 4, 4, 4, 4 }, // 0000 0
			{ 3, 0, 4, 4 }, // 0001 1
			{ 0, 1, 4, 4 }, // 0010 2
			{ 3, 1, 4, 4 }, // 0011 3
			{ 1, 2, 4, 4 }, // 0100 4
			{ 0, 1, 2, 3 }, // 0101 5
			{ 0, 2, 4, 4 }, // 0110 6
			{ 3, 2, 4, 4 }, // 0111 7
			{ 2, 3, 4, 4 }, // 1000 8
			{ 2, 0, 4, 4 }, // 1001 9
			{ 1, 2, 3, 0 }, // 1010 10
			{ 2, 1, 4, 4 }, // 1011 11
			{ 1, 3, 4, 4 }, // 1100 12
			{ 1, 0, 4, 4 }, // 1101 13
			{ 0, 3, 4, 4 }, // 1110 14
			{ 4, 4, 4, 4 } // 1111 15
	};
	const int afSquareValue8[16][4] = { { 4, 4, 4, 4 }, // 0000 0
			{ 3, 0, 4, 4 }, // 0001 1
			{ 0, 1, 4, 4 }, // 0010 2
			{ 3, 1, 4, 4 }, // 0011 3
			{ 1, 2, 4, 4 }, // 0100 4
			{ 2, 1, 0, 3 }, // 0101 5
			{ 0, 2, 4, 4 }, // 0110 6
			{ 3, 2, 4, 4 }, // 0111 7
			{ 2, 3, 4, 4 }, // 1000 8
			{ 2, 0, 4, 4 }, // 1001 9
			{ 1, 0, 3, 2 }, // 1010 10s
			{ 2, 1, 4, 4 }, // 1011 11
			{ 1, 3, 4, 4 }, // 1100 12
			{ 1, 0, 4, 4 }, // 1101 13
			{ 0, 3, 4, 4 }, // 1110 14
			{ 4, 4, 4, 4 } // 1111 15
	};
	float isoLevel = 0.0f;
	bool nudgeLevelSet;
	const float LEVEL_SET_TOLERANCE;
	TopologyRule2D rule = TopologyRule2D::Unconstrained;
	int rows=0, cols=0;
	const Image1f* img;
	const Image1i* label;
	float getValue(int x, int y,int l);
	int getLabel(int x, int y);
	float fGetOffset(uint2 v1, uint2 v2,int l);
	EdgeSplitPtr createSplit(std::map<uint64_t, EdgeSplitPtr>& splits, int p1x,int p1y, int p2x, int p2y, int l, int& vertCount);
	void addEdge(std::map<uint64_t, EdgeSplitPtr>& splits,std::list<EdgePtr>& edges, int p1x, int p1y, int p2x, int p2y,int p3x, int p3y, int p4x, int p4y,int l, int& vertCount);
	void processSquare(int x, int y,int l, std::map<uint64_t, EdgeSplitPtr>& splits,std::list<EdgePtr>& edges,int & vertCount);
	bool orient(const Image1f& img, const EdgeSplit& split1,const EdgeSplit& split2, Edge& edge);
public:
	MultiIsoContour(bool nudgeLevelSet = true, float levelSetTolerance = 1E-3f) :
			nudgeLevelSet(nudgeLevelSet), LEVEL_SET_TOLERANCE(
					levelSetTolerance), img(nullptr) {

	}
	virtual ~MultiIsoContour() {
	}

	void solve(const Image1f& levelset, const Image1i& labels, Vector2f& points, Vector1i& vertexLabels,
		std::vector<std::list<uint32_t>>& indexes, float isoLevel = 0.0f,
		const TopologyRule2D& rule = TopologyRule2D::Unconstrained,
		const Winding& winding = Winding::CounterClockwise);

};
}
#endif 

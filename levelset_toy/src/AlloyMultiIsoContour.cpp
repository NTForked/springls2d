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

#include "AlloyMultiIsoContour.h"
#include <map>
#include <list>
#include <algorithm>
#include <set>
namespace aly {
	bool MultiIsoContour::orient(const Image1f& img, const EdgeSplit& split1, const EdgeSplit& split2, Edge& edge) {
		float2 pt1 = split1.pt2d;
		float2 pt2 = split2.pt2d;
		float1x2 norm = img.gradient(0.5f * (pt1.x + pt2.x), 0.5f * (pt2.y + pt2.y));
		if (norm.x * (pt2.y - pt1.y) + norm.y * (pt1.x - pt2.x) > 0.0f) {
			std::swap(edge.x, edge.y);
			return false;
		}
		return true;
	}


	void MultiIsoContour::solve(const Image1f& levelset, const Image1i& labels, Vector2f& points, Vector1i& vertexLabels, std::vector<std::list<uint32_t>>& lines, float isoLevel, const TopologyRule2D& topoRule, const Winding& winding) {
		rows = levelset.width;
		cols = levelset.height;
		this->isoLevel = isoLevel;
		rule = topoRule;
		img = &levelset;
		label = &labels;

		size_t vertexOffset = 0;
		points.clear();
		lines.clear();
		vertexLabels.clear();
		std::map<int, MultiContourData> metadata;
		std::set<int> labelSet;
		for (int i = 0; i < rows - 1; i++) {
			for (int j = 0; j < cols - 1; j++) {
				labelSet.clear();
				labelSet.insert(getLabel(i, j));
				labelSet.insert(getLabel(i + 1, j));
				labelSet.insert(getLabel(i, j + 1));
				labelSet.insert(getLabel(i + 1, j + 1));
				for (int l : labelSet) {
					if (l != 0) {
						MultiContourData& data = metadata[l];
						processSquare(i, j, l, data.splits, data.edges,data.vertCount);
					}
				}
			}
		}

		for (auto pr : metadata) {
			int currentLabel = pr.first;
			MultiContourData& data = metadata[currentLabel];
			std::map<uint64_t, EdgeSplitPtr>& splits = data.splits;
			vertexOffset = points.size();
			std::vector<EdgeSplitPtr> pts(splits.size());
			for (const std::pair<uint64_t, EdgeSplitPtr>& split : splits) {
				pts[split.second->vid] = split.second;
			}
			points.resize(vertexOffset + splits.size());
			vertexLabels.resize(vertexOffset + splits.size());
			uint32_t index = 0;
			for (const std::pair<uint64_t, EdgeSplitPtr>& split : splits) {
				index = split.second->vid;
				points[vertexOffset + index] = split.second->pt2d;
				vertexLabels[vertexOffset + index] = currentLabel;
			}
			if (pts.size() == 0)continue;
			EdgeSplitPtr lastSplit = pts.front();
			bool firstPass = true;
			std::list<uint32_t> curvePath;
			index = 0;
			while (lastSplit.get() != nullptr) {
				curvePath.push_back(lastSplit->vid);
				EdgePtr e1 = lastSplit->e1;
				EdgePtr e2 = lastSplit->e2;
				float val = getValue(lastSplit->pt1.x, lastSplit->pt1.y, currentLabel);
				if ((val > 0 && winding == Winding::CounterClockwise)
					|| (val <= 0 && winding == Winding::Clockwise)) {
					// Swap edges so they are correctly ordered. This technique may
					// fail if a vertex lies exactly on the iso-level.
					std::swap(e1, e2);
				}
				// March around contour
				if (e1.get() != nullptr) {
					//if (!firstPass) {
					lastSplit->e1.reset();
					//}
					if (e1->x == lastSplit->vid) {
						lastSplit = pts[e1->y];
					}
					else {
						lastSplit = pts[e1->x];
					}
					if (!firstPass) {
						if (lastSplit->e1.get() == e1.get()) {
							lastSplit->e1.reset();
						}
						if (lastSplit->e2.get() == e1.get()) {
							lastSplit->e2.reset();
						}
					}
					firstPass = false;
				}
				else if (e2.get() != nullptr) {
					//if (!firstPass) {
					lastSplit->e2.reset();
					//}
					firstPass = false;
					if (e2->x == lastSplit->vid) {
						lastSplit = pts[e2->y];
					}
					else {
						lastSplit = pts[e2->x];
					}
					if (!firstPass) {
						if (lastSplit->e1.get() == e2.get()) {
							lastSplit->e1.reset();
						}
						if (lastSplit->e2.get() == e2.get()) {
							lastSplit->e2.reset();
						}
					}
					firstPass = false;
				}
				else {
					// Start new contour
					lastSplit.reset();
					//Add vertex offset to contour
					for (uint32_t& idx : curvePath) {
						idx += (uint32_t)vertexOffset;
					}
					lines.push_back(curvePath);
					curvePath.clear();
					firstPass = true;
					index = 0;
					while (index < pts.size()) {
						EdgeSplitPtr tmp = pts[index++];
						if (tmp->e1.get() != nullptr && tmp->e2.get() != nullptr) {
							lastSplit = tmp;
							break;
						}
					}
				}
			}
		}
		img = nullptr;
		label = nullptr;
	}

	/*
	* Geometric Tools, LLC Copyright (c) 1998-2010 Distributed under the Boost
	* Software License, Version 1.0. http://www.boost.org/LICENSE_1_0.txt
	* http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
	*
	* File Version: 4.10.0 (2009/11/18)
	*/
	void MultiIsoContour::processSquare(int i, int j, int l, std::map<uint64_t, EdgeSplitPtr>& splits, std::list<EdgePtr>& edges,int& vertCount) {
		float iF00 = getValue(i, j, l);
		float iF10 = getValue(i + 1, j, l);
		float iF01 = getValue(i, j + 1, l);
		float iF11 = getValue(i + 1, j + 1, l);
		bool signFlip = false;
		if (iF00 != 0) {
			if (iF00 < 0) {
				iF00 = -iF00;
				iF10 = -iF10;
				iF11 = -iF11;
				iF01 = -iF01;
				signFlip = true;
			}

			if (iF10 > 0) {
				if (iF11 > 0) {
					if (iF01 > 0) {
						// ++++
						return;
					}
					else {
						// +++-
						addEdge(splits, edges, i, j + 1, i + 1, j + 1, i,
							j + 1, i, j, l,vertCount);
					}
				}
				else if (iF11 < 0) {
					if (iF01 > 0) {
						// ++-+
						addEdge(splits, edges, i + 1, j, i + 1, j + 1, i + 1,
							j + 1, i, j + 1, l,vertCount);
					}
					else if (iF01 < 0) {
						// ++--
						addEdge(splits, edges, i, j + 1, i, j, i + 1, j, i + 1,
							j + 1, l,vertCount);
					}
					else {
						// ++-0
						addEdge(splits, edges, i, j + 1, i, j + 1, i + 1, j,
							i + 1, j + 1, l,vertCount);
					}
				}
				else {
					if (iF01 > 0) {
						// ++0+
						return;
					}
					else if (iF01 < 0) {
						// ++0-
						addEdge(splits, edges, i + 1, j + 1, i + 1, j + 1, i,
							j, i, j + 1, l,vertCount);
					}
					else {
						// ++00
						addEdge(splits, edges, i + 1, j + 1, i + 1, j + 1, i,
							j + 1, i, j + 1, l,vertCount);
					}
				}
			}
			else if (iF10 < 0) {
				if (iF11 > 0) {
					if (iF01 > 0) {
						// +-++
						addEdge(splits, edges, i, j, i + 1, j, i + 1, j + 1,
							i + 1, j, l,vertCount);
					}
					else if (iF01 < 0) {
						// +-+-
						// Ambiguous Case
						if (rule == TopologyRule2D::Unconstrained) {
							float iD0 = iF00 - iF10;
							float iXN0 = iF00 * (i + 1) - iF10 * i;
							float iD3 = iF11 - iF01;
							float iXN1 = iF11 * i - iF01 * (i + 1);
							float iDet = 0;
							if (iD0 * iD3 > 0) {
								iDet = iXN1 * iD0 - iXN0 * iD3;
							}
							else {
								iDet = iXN0 * iD3 - iXN1 * iD0;
							}
							if (iDet > 0) {
								addEdge(splits, edges, i + 1, j + 1, i, j + 1,
									i + 1, j + 1, i + 1, j, l,vertCount);
								addEdge(splits, edges, i, j, i + 1, j, i, j, i,
									j + 1, l,vertCount);
							}
							else {
								addEdge(splits, edges, i + 1, j + 1, i, j + 1,
									i, j, i, j + 1, l,vertCount);
								addEdge(splits, edges, i, j, i + 1, j, i + 1,
									j + 1, i + 1, j, l,vertCount);
							}
						}
						else if (rule == TopologyRule2D::Connect4) {
							if (signFlip) {
								addEdge(splits, edges, i + 1, j + 1, i, j + 1,
									i + 1, j + 1, i + 1, j, l,vertCount);
								addEdge(splits, edges, i, j, i + 1, j, i, j, i,
									j + 1, l,vertCount);
							}
							else {
								addEdge(splits, edges, i + 1, j + 1, i, j + 1,
									i, j, i, j + 1, l,vertCount);
								addEdge(splits, edges, i, j, i + 1, j, i + 1,
									j + 1, i + 1, j, l,vertCount);
							}
						}
						else if (rule == TopologyRule2D::Connect8) {
							if (signFlip) {
								addEdge(splits, edges, i + 1, j + 1, i, j + 1,
									i, j, i, j + 1, l,vertCount);
								addEdge(splits, edges, i, j, i + 1, j, i + 1,
									j + 1, i + 1, j, l,vertCount);
							}
							else {
								addEdge(splits, edges, i + 1, j + 1, i, j + 1,
									i + 1, j + 1, i + 1, j, l,vertCount);
								addEdge(splits, edges, i, j, i + 1, j, i, j, i,
									j + 1, l,vertCount);
							}
						}
					}
					else {
						// +-+0
						addEdge(splits, edges, i, j, i + 1, j, i + 1, j + 1,
							i + 1, j, l,vertCount);
					}
				}
				else if (iF11 < 0) {
					if (iF01 > 0) {
						// +--+
						addEdge(splits, edges, i, j, i + 1, j, i + 1, j + 1, i,
							j + 1, l,vertCount);
					}
					else if (iF01 < 0) {
						// +---
						addEdge(splits, edges, i, j + 1, i, j, i, j, i + 1, j, l,vertCount);
					}
					else {
						// +--0
						addEdge(splits, edges, i, j + 1, i, j + 1, i, j, i + 1,
							j, l,vertCount);
					}
				}
				else {
					if (iF01 > 0) {
						// +-0+
						addEdge(splits, edges, i + 1, j + 1, i + 1, j + 1, i,
							j, i + 1, j, l,vertCount);
					}
					else if (iF01 < 0) {
						// +-0-
						addEdge(splits, edges, i, j + 1, i, j, i, j, i + 1, j, l,vertCount);
					}
					else {
						// +-00
						addEdge(splits, edges, i + 1, j + 1, i + 1, j + 1, i,
							j + 1, i + 1, j + 1, l,vertCount);
						addEdge(splits, edges, i, j + 1, i + 1, j + 1, i,
							j + 1, i, j + 1, l,vertCount);
						addEdge(splits, edges, i, j + 1, i + 1, j + 1, i, j,
							i + 1, j, l,vertCount);
					}
				}
			}
			else {
				if (iF11 > 0) {
					if (iF01 > 0) {
						// +0++
					}
					else if (iF01 < 0) {
						// +0+-
						addEdge(splits, edges, i, j + 1, i + 1, j + 1, i,
							j + 1, i, j, l,vertCount);
					}
				}
				else if (iF11 < 0) {
					if (iF01 > 0) {
						// +0-+
						addEdge(splits, edges, i + 1, j, i + 1, j, i, j + 1,
							i + 1, j + 1, l,vertCount);
					}
					else if (iF01 < 0) {
						// +0--
						addEdge(splits, edges, i + 1, j, i + 1, j, i, j, i,
							j + 1, l,vertCount);
					}
					else {
						// +0-0
						addEdge(splits, edges, i + 1, j, i + 1, j, i, j + 1, i,
							j + 1, l,vertCount);
					}
				}
				else {
					if (iF01 > 0) {
						// +00+
						addEdge(splits, edges, i + 1, j, i + 1, j, i + 1,
							j + 1, i + 1, j + 1, l,vertCount);
					}
					else if (iF01 < 0) {
						// +00-
						addEdge(splits, edges, i + 1, j, i + 1, j, i + 1, j,
							i + 1, j + 1, l,vertCount);
						addEdge(splits, edges, i + 1, j, i + 1, j + 1, i + 1,
							j + 1, i + 1, j + 1, l,vertCount);
						addEdge(splits, edges, i + 1, j, i + 1, j + 1, i, j, i,
							j + 1, l,vertCount);
					}
					else {
						// +000
						addEdge(splits, edges, i, j + 1, i, j + 1, i, j, i, j, l,vertCount);
						addEdge(splits, edges, i, j, i, j, i + 1, j, i + 1, j, l,vertCount);
					}
				}
			}
		}
		else if (iF10 != 0) {
			// convert to case 0+**
			if (iF10 < 0) {
				iF10 = -iF10;
				iF11 = -iF11;
				iF01 = -iF01;
			}

			if (iF11 > 0) {
				if (iF01 > 0) {
					// 0+++
				}
				else if (iF01 < 0) {
					// 0++-
					addEdge(splits, edges, i, j, i, j, i, j + 1, i + 1, j + 1, l,vertCount);
				}
				else {
					// 0++0
					addEdge(splits, edges, i, j + 1, i, j + 1, i, j, i, j, l,vertCount);
				}
			}
			else if (iF11 < 0) {
				if (iF01 > 0) {
					// 0+-+
					addEdge(splits, edges, i + 1, j, i + 1, j + 1, i + 1,
						j + 1, i, j + 1, l,vertCount);
				}
				else if (iF01 < 0) {
					// 0+--
					addEdge(splits, edges, i, j, i, j, i + 1, j, i + 1, j + 1, l,vertCount);
				}
				else {
					// 0+-0
					addEdge(splits, edges, i, j, i, j, i, j, i, j + 1, l,vertCount);
					addEdge(splits, edges, i, j, i, j + 1, i, j + 1, i, j + 1, l,vertCount);
					addEdge(splits, edges, i, j, i, j + 1, i + 1, j, i + 1,
						j + 1, l,vertCount);
				}
			}
			else {
				if (iF01 > 0) {
					// 0+0+
				}
				else if (iF01 < 0) {
					// 0+0-
					addEdge(splits, edges, i, j, i, j, i + 1, j + 1, i + 1,
						j + 1, l,vertCount);
				}
				else {
					// 0+00
					addEdge(splits, edges, i + 1, j + 1, i + 1, j + 1, i,
						j + 1, i, j + 1, l,vertCount);
					addEdge(splits, edges, i, j + 1, i, j + 1, i, j, i, j, l,vertCount);
				}
			}
		}
		else if (iF11 != 0) {
			// convert to case 00+*
			if (iF11 < 0) {
				iF11 = -iF11;
				iF01 = -iF01;
			}

			if (iF01 > 0) {
				// 00++
				addEdge(splits, edges, i, j, i, j, i + 1, j, i + 1, j, l,vertCount);
			}
			else if (iF01 < 0) {
				// 00+-
				addEdge(splits, edges, i, j, i, j, i, j, i + 1, j, l,vertCount);
				addEdge(splits, edges, i, j, i + 1, j, i + 1, j, i + 1, j, l,vertCount);
				addEdge(splits, edges, i, j, i + 1, j, i, j + 1, i + 1, j + 1, l,vertCount);
			}
			else {
				// 00+0
				addEdge(splits, edges, i + 1, j, i + 1, j, i + 1, j + 1, i + 1,
					j + 1, l,vertCount);
				addEdge(splits, edges, i + 1, j + 1, i + 1, j + 1, i, j + 1, i,
					j + 1, l,vertCount);
			}
		}
		else if (iF01 != 0) {
			// cases 000+ or 000-
			addEdge(splits, edges, i, j, i, j, i + 1, j, i + 1, j, l,vertCount);
			addEdge(splits, edges, i + 1, j, i + 1, j, i + 1, j + 1, i + 1,
				j + 1, l,vertCount);
		}
		else {
			// case 0000
			addEdge(splits, edges, i, j, i, j, i + 1, j, i + 1, j, l,vertCount);
			addEdge(splits, edges, i + 1, j, i + 1, j, i + 1, j + 1, i + 1,
				j + 1, l,vertCount);
			addEdge(splits, edges, i + 1, j + 1, i + 1, j + 1, i, j + 1, i,
				j + 1, l,vertCount);
			addEdge(splits, edges, i, j + 1, i, j + 1, i, j, i, j, l,vertCount);
		}
	}
	float MultiIsoContour::getValue(int i, int j, int l) {
		float sgn = (((*label)(i, j).x == l) ? -1.0f : 1.0f);
		float val = (*img)(i, j)*sgn - isoLevel;
		if (nudgeLevelSet) {
			if (val < 0) {
				val = std::min(val, -LEVEL_SET_TOLERANCE);
			}
			else {
				val = std::max(val, LEVEL_SET_TOLERANCE);
			}
		}
		return val;
	}
	int MultiIsoContour::getLabel(int i, int j) {
		return (*label)(i, j).x;
	}
	float MultiIsoContour::fGetOffset(uint2 v1, uint2 v2, int l) {
		float fValue1 = getValue(v1.x, v1.y, l);
		float fValue2 = getValue(v2.x, v2.y, l);
		double fDelta = fValue2 - fValue1;
		if (fDelta == 0.0) {
			return 0.5f;
		}
		return (float)(-fValue1 / fDelta);
	}

	void MultiIsoContour::addEdge(std::map<uint64_t, EdgeSplitPtr>& splits, std::list<EdgePtr>& edges, int p1x, int p1y, int p2x, int p2y, int p3x, int p3y, int p4x, int p4y, int l, int& vertCount) {
		EdgeSplitPtr split1 = createSplit(splits, p1x, p1y, p2x, p2y, l, vertCount);
		EdgeSplitPtr split2 = createSplit(splits, p3x, p3y, p4x, p4y, l, vertCount);
		EdgePtr edge = EdgePtr(new Edge(split1->vid, split2->vid));
		if (split1->e1.get() == nullptr) {
			split1->e1 = edge;
		}
		else {
			split1->e2 = edge;
		}
		if (split2->e1.get() == nullptr) {
			split2->e1 = edge;
		}
		else {
			split2->e2 = edge;
		}
		edges.push_back(edge);
	}
	EdgeSplitPtr MultiIsoContour::createSplit(std::map<uint64_t, EdgeSplitPtr>& splits, int p1x, int p1y, int p2x, int p2y, int l, int& vertCount) {
		EdgeSplitPtr split = EdgeSplitPtr(new EdgeSplit(uint2(p1x, p1y), uint2(p2x, p2y)));
		uint64_t hash = split->hashValue(rows, cols);
		if (splits.find(hash) != splits.end()) {
			EdgeSplitPtr foundSplit = splits[hash];
			return foundSplit;
		}
		else {
			split->vid = vertCount++;
			float2 pt2d(0.0f);
			float fOffset = fGetOffset(split->pt1, split->pt2, l);
			float fInvOffset = 1.0f - fOffset;
			pt2d.x = (fInvOffset * p1x + fOffset * p2x);
			pt2d.y = (fInvOffset * p1y + fOffset * p2y);
			split->pt2d = pt2d;
			splits[split->hashValue(rows, cols)] = split;
			return split;
		}
	}
}
